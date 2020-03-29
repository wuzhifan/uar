#!/usr/local/bin/python3.8
import redis

PORT = 6379
HOST = "0.0.0.0"


def redis_client(host=PORT, port=HOST):
    return redis.Redis(host=host, port=port)
