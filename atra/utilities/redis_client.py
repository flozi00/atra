from redis_cache import RedisCache
from redis import StrictRedis
import os

redis_client = StrictRedis(
    host=os.getenv("CACHE_HOST", "localhost"), decode_responses=True
)
cache = RedisCache(redis_client=redis_client)
