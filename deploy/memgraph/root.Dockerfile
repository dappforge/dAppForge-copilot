FROM memgraph/memgraph-mage:3.4

# Switch to root user
USER root

# Optionally expose Memgraph default ports
EXPOSE 7687 7444 3000

# Start Memgraph as root (CMD/ENTRYPOINT is inherited from base image)