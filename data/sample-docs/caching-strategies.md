# Caching Strategies

A cache is a fast storage layer that holds copies of data so future requests can be served without recomputing or re-fetching the original data. Caches sit between an application and its slower data source, usually a database, and they trade freshness for speed. The most common cache stores are in-memory systems such as Redis and Memcached, and content delivery networks (CDNs) that cache static assets close to users.

## Cache-aside (lazy loading)

In the cache-aside pattern, the application first checks the cache. On a cache hit, it returns the cached value immediately. On a cache miss, it reads the data from the database, writes it into the cache, and then returns it. Cache-aside is simple and only caches data that is actually requested, but the first request for any key is always slow, and stale data can linger until its time-to-live expires. It is the most widely used caching pattern for read-heavy workloads.

## Write-through and write-behind

In write-through caching, every write goes to the cache and to the database at the same time, so the cache is always consistent with the database. The cost is higher write latency, because the application waits for both writes. In write-behind caching, also called write-back, writes go to the cache first and are flushed to the database asynchronously in the background. Write-behind gives very low write latency, but data can be lost if the cache crashes before the flush. Write-through is preferred when consistency matters more than write speed.

## TTL and eviction

Every cached entry should have a time-to-live (TTL), the number of seconds after which the entry expires and must be re-fetched. A short TTL keeps data fresh but increases load on the database; a long TTL reduces load but risks serving stale data. When a cache fills up, an eviction policy decides which entries to remove. Least recently used (LRU) evicts the entry that has not been accessed for the longest time, while least frequently used (LFU) evicts the entry with the fewest accesses. LRU is the default in most systems, including Redis, because it adapts well to changing access patterns.

## Cache stampede and thundering herd

A cache stampede, also called a thundering herd, happens when a popular cached entry expires and many concurrent requests all miss the cache at once, flooding the database with the same expensive query. Common defences are request coalescing, where only the first request recomputes the value while the others wait, and probabilistic early expiration, where the TTL is jittered so entries do not all expire at the same moment. Another defence is a stale-while-revalidate policy, which serves the old cached value while a background refresh updates it.

## Where to cache

Caching can happen at several layers: the browser cache, a CDN for static assets, an application-level cache such as Redis for database query results and session data, and a database query cache. Each layer has its own TTL and invalidation rules. The hardest part of caching is cache invalidation — deciding when a cached value is no longer valid. The safest rule is to invalidate a cache entry whenever the underlying data changes, rather than relying on TTL alone for correctness.
