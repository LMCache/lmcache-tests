# Redis-Sentinel-Docker-Compose

An example setup for using Redis Sentinel with Docker Compose.

For more information and an explanation, see: https://www.feedthedev.com/2020/07/13/using-redis-sentinel-with-docker-compose/

# Usage
By default, it will start 3 sentinel nodes on localhost:26379, localhost:26380, localhost:26381
```bash
# Start the cluster
docker compose up --build

# Start the cluster in background
docker compose up -d

# Stop the cluster
docker compose down
```

