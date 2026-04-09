import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Listen on 0.0.0.0 for Docker access
    port: 5173,
    proxy: {
      // Local dev: proxy to Kong Gateway on localhost:8000
      '/api': 'http://localhost:8000',
      '/auth': 'http://localhost:8000',
      '/socket.io': {
        target: 'http://localhost:8000', // Kong → haproxy:8005 → stream-service
        ws: true,
        rewrite: (path) => path.replace(/^\/socket.io/, '/socket.io')
      }
    }
  }
});
