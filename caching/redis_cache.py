import redis
import os
import json
from typing import Optional
import asyncio

# Initialize Redis client
_REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
_REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
_REDIS_DB = int(os.getenv('REDIS_DB', '0'))
redis_client = redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT, db=_REDIS_DB)

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