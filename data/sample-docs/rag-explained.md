# Retrieval-Augmented Generation (RAG) Explained

## The Problem RAG Solves

Large language models are powerful, but they have two weaknesses: their knowledge stops at their training cutoff, and they sometimes hallucinate — confidently stating facts that are false. Retrieval-Augmented Generation, or RAG, addresses both problems by giving the model relevant documents at answer time. Instead of relying only on what it memorised during training, the model reads retrieved passages and grounds its answer in them. The result is more accurate, more up-to-date, and more verifiable: you can show the user exactly which sources the answer came from.

A basic RAG pipeline has three stages. First, ingestion: documents are split into chunks, each chunk is converted into an embedding, and the chunks and embeddings are stored in a vector database. Second, retrieval: a user question is embedded the same way, and the database returns the chunks whose embeddings are closest to the question embedding. Third, generation: the retrieved chunks are placed into the language model's prompt as context, and the model writes an answer based on that context.

## Embeddings and Vector Search

An embedding is a list of numbers — a vector — that represents the meaning of a piece of text. Embedding models are trained so that texts with similar meanings end up with vectors that are close together in vector space, while unrelated texts end up far apart. Similarity is usually measured with cosine similarity, which compares the angle between two vectors. A question about "Python async" will therefore be close to a chunk about `asyncio` and far from a chunk about database migrations, even if the two chunks share few exact words.

Vector search is the process of finding the nearest vectors to a query vector. Vector databases such as Qdrant, Pinecone, and pgvector are built for this: they index millions of vectors with approximate nearest neighbour algorithms such as HNSW, so a search that would take seconds with a brute-force scan returns in milliseconds. The quality of retrieval depends heavily on chunking. Chunks that are too large dilute the meaning and waste context; chunks that are too small lose the surrounding explanation. A common starting point is chunks of 500 to 1,000 characters with a small overlap of 100 to 200 characters, so a sentence split across a boundary is not lost.

## Hybrid Search and Reranking

Pure vector search understands meaning but can miss exact terms: a query for a specific error code or product name may retrieve semantically similar text that never mentions the term. Keyword search, such as BM25, has the opposite strength — it matches exact words but understands no meaning. Hybrid search combines both: it runs a vector search and a BM25 keyword search, then merges the two ranked lists, often with reciprocal rank fusion, so a chunk that scores well on either method rises to the top.

Retrieval usually fetches more candidates than the model needs — for example, the top 20 chunks — and then a reranking step reorders them. A reranker is a small cross-encoder model that scores each query-chunk pair directly, which is more accurate than comparing precomputed embeddings but too slow to run over the whole database. Reranking typically keeps the best 3 to 5 chunks for the final prompt. Together, hybrid search and reranking are the cheapest way to improve RAG answer quality: they fix retrieval errors before the language model ever sees the context, and a model given the right context rarely needs to be a bigger model.
