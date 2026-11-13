# Database Indexing

A database index is a data structure that lets a database find rows quickly without scanning the whole table. Without an index, the database performs a full table scan, reading every row to find the ones that match a query. With an index, it can jump straight to the matching rows. Indexes make reads much faster, but they make writes slower, because every insert, update, and delete must also update each index on the table.

## B-tree indexes

The B-tree is the default index type in most relational databases, including PostgreSQL and MySQL. A B-tree keeps keys in sorted order, which makes it efficient for equality lookups, range queries such as "price between 10 and 50", and sorting by the indexed column. A B-tree index on a user id column makes `WHERE user_id = 42` fast, and an index on a created_at column makes ordering by creation time fast. Most queries that filter or sort on a column benefit from a B-tree index on that column.

## Hash indexes

A hash index stores a hash of the key and is only useful for exact equality lookups, such as `WHERE email = 'a@b.com'`. Hash indexes cannot answer range queries or ordering, so they are rarely the right choice in a general-purpose database. PostgreSQL supports hash indexes, but B-tree indexes are the default and are usually preferred because they handle more query shapes.

## Composite and covering indexes

A composite index is an index on more than one column, for example (tenant_id, created_at). It is most effective when queries filter on the leftmost columns first; a query that filters only on created_at cannot use the (tenant_id, created_at) index efficiently, because the index is sorted by tenant_id first. A covering index is one that contains every column a query needs, so the database can answer the query from the index alone without touching the table at all. Covering indexes are powerful for read-heavy reporting queries.

## The cost of indexes

Every index costs storage space and write performance. On a write-heavy table, each additional index slows inserts and updates, because the database must maintain the index for every write. An index that is never used by any query is pure overhead. Good practice is to add indexes only for queries that actually run in production, to measure query performance with EXPLAIN ANALYZE, and to drop indexes that the query planner never uses. A table with ten indexes is usually a sign that the indexing strategy was not thought through.

## When indexes do not help

Indexes do not help when a query matches most of the table, because reading the index and then fetching the rows costs more than a simple sequential scan. They also do not help when the indexed column has very low cardinality, such as a boolean column with only two values, or when the query applies a function to the column, such as `WHERE LOWER(name) = 'alice'`, which prevents the planner from using a plain index on name. In those cases the fix is a different query, a partial index, or an expression index that matches the function used in the query.
