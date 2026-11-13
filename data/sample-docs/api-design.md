# API Design

An application programming interface (API) is the contract that lets one piece of software talk to another. Good API design makes a system easy to use, easy to change, and easy to debug. The most common style for web APIs is REST, which models resources as URLs and uses the standard HTTP verbs: GET to read, POST to create, PUT or PATCH to update, and DELETE to remove.

## REST and versioning

In a REST API, each resource has its own URL, such as /users/42/orders, and the response is usually JSON. APIs must be versioned, because clients depend on the exact shape of the response. The simplest approach is to put the version in the URL, for example /v1/users, so a new version can be shipped without breaking existing clients. Breaking changes, such as removing a field or changing its type, should only go into a new major version, and old versions should be supported for a deprecation period rather than removed overnight.

## Pagination and filtering

A list endpoint must never return an unbounded number of rows. Pagination splits results into pages, typically with limit and offset parameters, or better with cursor-based pagination, where each page returns a cursor token that fetches the next page. Cursor-based pagination is more stable than offset pagination, because it does not skip or repeat rows when new data is inserted between requests. Filtering and sorting should use explicit query parameters, such as ?status=active&sort=created_at, and the API should validate them rather than silently ignoring unknown parameters.

## Idempotency

An idempotency key is a unique token the client sends with a request, usually in a header, so that retrying the same request does not cause a duplicate effect. This matters most for POST requests that create something, such as charging a payment. If the network drops the response, the client retries with the same idempotency key and the server returns the original result instead of charging twice. Without idempotency, retries are unsafe, and clients must guess whether the first attempt succeeded.

## Rate limiting and retries

Rate limiting protects a service from being overwhelmed by too many requests. A common algorithm is the token bucket, which allows short bursts up to a limit and then enforces a steady rate. When a client is rate limited, the server should return HTTP status 429 with a Retry-After header. Clients should respect that header and use exponential backoff, waiting longer after each failed attempt, with jitter added so that many clients do not all retry at the same instant.

## Microservices and the API gateway

In a microservices architecture, the system is split into small services that each own one capability, such as billing or search, and communicate over the network. Services should own their own data and should not share a database directly. An API gateway sits in front of the services and gives clients a single entry point, handling authentication, rate limiting, and routing to the right service. A monolith is often the right starting point, and splitting into microservices only pays off when teams and scaling needs make the extra operational complexity worthwhile.
