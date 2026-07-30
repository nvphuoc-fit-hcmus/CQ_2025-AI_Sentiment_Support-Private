import os
import json
import logging
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

LOG = logging.getLogger("crawler.kafka")

KAFKA_BROKER = os.getenv("KAFKA_BROKERS", "kafka:9092")
TOPIC = os.getenv("NEWS_RAW_TOPIC", "news_raw")
ANALYZED_TOPIC = os.getenv("NEWS_ANALYZED_TOPIC", "news_analyzed")

# Allow passing additional producer config via env (comma-separated key=val)
producer_config = {
    "bootstrap.servers": KAFKA_BROKER,
    # default acks to all for durability
    "acks": os.getenv("KAFKA_ACKS", "all")
}
_extra = os.getenv("KAFKA_PRODUCER_CONFIG")
if _extra:
    for pair in _extra.split(','):
        if '=' in pair:
            k, v = pair.split('=', 1)
            producer_config[k.strip()] = v.strip()

producer = Producer(producer_config)

def create_startup_topics():
    """Explicitly create topics on startup to avoid consumer errors."""
    admin_client = AdminClient({'bootstrap.servers': KAFKA_BROKER})
    # Create news_raw topic with 1 partition and replication factor 1 (since we have 1 broker)
    # You can adjust num_partitions and replication_factor as needed.
    new_topics = [
        NewTopic(TOPIC, num_partitions=1, replication_factor=1),
        NewTopic(ANALYZED_TOPIC, num_partitions=1, replication_factor=1),
    ]
    
    # Call create_topics to asynchronously create topics.
    fs = admin_client.create_topics(new_topics)

    # Wait for each operation to finish.
    for topic, f in fs.items():
        try:
            f.result()  # The result itself is None
            LOG.info("Topic {} created".format(topic))
        except Exception as e:
            # If topic already exists, it will raise an error, which is fine.
            LOG.info("Topic {} creation failed (might already exist): {}".format(topic, e))

def _delivery(err, msg):
    if err:
        LOG.error("Delivery failed: %s", err)
    else:
        LOG.debug("Delivered message to %s [%s]@%s", msg.topic(), msg.partition(), msg.offset())


def produce_news(data: dict, flush: bool = False):
    """Produce a news JSON to Kafka topic.

    data should be JSON-serializable (dict with source, url, title, content, published_at)
    """
    try:
        payload = json.dumps(data, ensure_ascii=False)
        producer.produce(TOPIC, payload.encode("utf-8"), callback=_delivery)
        # Extraction already includes a normalized sentiment score, so publish
        # it directly for core-service persistence as well. There is no
        # separate raw-to-analyzed worker in this deployment.
        producer.produce(ANALYZED_TOPIC, payload.encode("utf-8"), callback=_delivery)
        # poll to serve delivery callbacks
        producer.poll(0)
    except Exception as e:
        LOG.exception("Failed to produce message: %s", e)
        raise
    if flush:
        try:
            producer.flush(10)
        except Exception:
            LOG.exception("flush failed")


def close_producer(timeout: int = 10):
    try:
        LOG.info("Flushing Kafka producer before shutdown")
        producer.flush(timeout)
    except Exception:
        LOG.exception("Error flushing producer")
