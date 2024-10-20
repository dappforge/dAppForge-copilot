import redis
import json
from typing import Optional
import asyncio

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def generate_cache_key(*args, **kwargs) -> str:
    unique_string = ''.join(str(arg) for arg in args) + ''.join(f"{k}={v}" for k, v in kwargs.items())
    return unique_string

def get_cached_result(key: str) -> Optional[dict]:
    cached_result = redis_client.get(key)
    if cached_result:
        return json.loads(cached_result)
    return None

def set_cache_result(key: str, result: dict, expiry: int = 3600):
    redis_client.set(key, json.dumps(result), ex=expiry)

def invalidate_cache():
    redis_client.flushdb()  # This will clear all cache entries, use with caution.

# Asynchronous versions of the functions
async def async_generate_cache_key(*args, **kwargs) -> str:
    return generate_cache_key(*args, **kwargs)

async def async_get_cached_result(key: str) -> Optional[dict]:
    return await asyncio.to_thread(get_cached_result, key)

async def async_set_cache_result(key: str, result: dict, expiry: int = 3600):
    await asyncio.to_thread(set_cache_result, key, result, expiry)

async def async_invalidate_cache():
    await asyncio.to_thread(invalidate_cache)