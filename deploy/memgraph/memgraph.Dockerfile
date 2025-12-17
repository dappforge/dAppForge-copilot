FROM memgraph/memgraph-mage:3.4

# Switch to memgraph user
USER memgraph

# Optionally expose Memgraph default ports
EXPOSE 7687 7444 3000

# Start Memgraph as memgraph (CMD/ENTRYPOINT is inherited from base image)