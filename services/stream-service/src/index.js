const { createServer } = require('http');
const { Server } = require('socket.io');
const { createAdapter } = require('@socket.io/redis-adapter');
const { createClient } = require('redis');
const { initConsumer, getSequenceCounter, cleanup } = require('./kafkaConsumer');

const PORT = process.env.PORT || 3000;
const REDIS_URL = process.env.REDIS_URL || 'redis://redis:6379';
const INSTANCE_ID = process.env.HOSTNAME || `instance-${Math.random().toString(36).substr(2, 9)}`;

let isShuttingDown = false;
let connectionCount = 0;
let messageCount = 0;
let startTime = Date.now();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function connectRedisWithRetry(client, name) {
  let attempt = 0;
  while (!isShuttingDown) {
    try {
      attempt += 1;
      await client.connect();
      console.log(`${name} connected`);
      return;
    } catch (err) {
      const waitMs = Math.min(1000 * 2 ** Math.min(attempt, 5), 15000);
      console.error(`${name} connection failed (attempt ${attempt}), retrying in ${waitMs}ms:`, err?.message || err);
      await sleep(waitMs);
    }
  }
}

// Enhanced HTTP Server with detailed health check
const httpServer = createServer((req, res) => {
  if (req.url === '/health') {
    const status = isShuttingDown ? 503 : 200;
    const health = {
      instance_id: INSTANCE_ID,
      alive: !isShuttingDown,
      status: isShuttingDown ? 'shutting_down' : 'healthy',
      connections: connectionCount,
      messages_processed: messageCount,
      sequence_counter: getSequenceCounter(),
      uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
      memory_usage: {
        rss: Math.round(process.memoryUsage().rss / 1024 / 1024) + 'MB',
        heapUsed: Math.round(process.memoryUsage().heapUsed / 1024 / 1024) + 'MB'
      },
      timestamp: new Date().toISOString()
    };

    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(health, null, 2));
    return;
  }

  res.writeHead(404);
  res.end('Not Found');
});

async function start() {
  // Setup Redis Adapter
  const pubClient = createClient({ url: REDIS_URL });
  const subClient = pubClient.duplicate();

  pubClient.on('error', (err) => {
    console.error('Redis pub client error:', err?.message || err);
  });
  subClient.on('error', (err) => {
    console.error('Redis sub client error:', err?.message || err);
  });

  await connectRedisWithRetry(pubClient, 'Redis pub client');
  await connectRedisWithRetry(subClient, 'Redis sub client');
  console.log('Redis adapter connected');

  const io = new Server(httpServer, {
    cors: { origin: '*', methods: ['GET'] },
    path: '/socket.io',
    transports: ['websocket'],
    perMessageDeflate: false,
    httpCompression: false,
    adapter: createAdapter(pubClient, subClient)
  });

  // Connection tracking with hash key logging
  io.on('connection', (socket) => {
    connectionCount++;
    const hashKey = socket.handshake.query.token || 'no-token';
    console.log(`[CONNECTION] Client ${socket.id} with hash_key: ${hashKey.substring(0, 20)}... -> Instance: ${INSTANCE_ID}. Total: ${connectionCount}`);

    // Subscribe to symbols
    socket.on('subscribe', (symbol) => {
      if (symbol && typeof symbol === 'string') {
        const room = symbol.toUpperCase();
        socket.join(room);

        // Also join interval-specific rooms
        const intervals = ['1M', '5M', '15M', '1H'];
        intervals.forEach(interval => {
          socket.join(`${room}_${interval}`);
        });

        console.log(`[SUBSCRIBE] Client ${socket.id} joined room: ${room}`);

        const roomSize = io.sockets.adapter.rooms.get(room)?.size || 0;
        console.log(`[SUBSCRIBE] Room ${room} now has ${roomSize} client(s)`);
      }
    });

    // Join user room for notifications
    socket.on('join_user_room', (userId) => {
      if (userId && typeof userId === 'string') {
        socket.join(`user_${userId}`);
        console.log(`[USER_ROOM] Client ${socket.id} joined user room: user_${userId}`);
      }
    });

    socket.on('unsubscribe', (symbol) => {
      if (symbol && typeof symbol === 'string') {
        const room = symbol.toUpperCase();
        socket.leave(room);

        // Also leave interval-specific rooms
        const intervals = ['1M', '5M', '15M', '1H'];
        intervals.forEach(interval => {
          socket.leave(`${room}_${interval}`);
        });

        console.log(`[UNSUBSCRIBE] Client ${socket.id} left room: ${room}`);
      }
    });

    socket.on('disconnect', (reason) => {
      connectionCount--;
      console.log(`[DISCONNECT] Client ${socket.id} disconnected. Reason: ${reason}. Total: ${connectionCount}`);
    });
  });

  // Start Kafka Consumer
  try {
    await initConsumer(io);
  } catch (e) {
    console.error('Kafka Consumer init failed', e);
  }

  // Graceful shutdown
  const shutdown = async () => {
    console.log('Graceful shutdown initiated...');
    isShuttingDown = true;

    // 1. Stop accepting new connections
    httpServer.close();

    // 2. Notify all clients
    io.emit('server_shutdown', {
      reason: 'maintenance',
      message: 'Server đang bảo trì, vui lòng kết nối lại sau 5 giây',
      reconnectAfter: 5000
    });

    console.log('Notified all clients about shutdown');

    // 3. Wait for clients to disconnect gracefully
    console.log('Waiting 10s for clients to disconnect...');
    await new Promise(resolve => setTimeout(resolve, 10000));

    // 4. Force close remaining connections
    console.log(`Closing remaining ${connectionCount} connections...`);
    io.close();

    // 5. Cleanup Kafka and Redis
    await cleanup();
    await pubClient.disconnect();
    await subClient.disconnect();

    console.log('Shutdown complete');
    process.exit(0);
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);

  httpServer.listen(PORT, () => {
    console.log(`Stream Service running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
    console.log(`Instance ID: ${INSTANCE_ID}`);
    console.log(`Consistent Hashing: Enabled (hash key: JWT token)`);
  });
}

start().catch(console.error);
