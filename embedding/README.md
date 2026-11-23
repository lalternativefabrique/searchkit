# embedding

Turns text into vectors through any OpenAI-compatible `/embeddings` endpoint,
and reorders candidates through any Cohere-compatible `/rerank` endpoint.
Standard library only.

```go
import "github.com/lalternativefabrique/searchkit/embedding"
```

One client for every product that indexes and searches text, in place of the
copies each one grew on its own.

## Embeddings

```go
c, err := embedding.New(embedding.Config{
    BaseURL:           "https://api.scaleway.ai/<project>/v1",
    APIKey:            key,
    Model:             "qwen3-embedding-8b",
    Dimensions:        1024,
    RequestDimensions: true,
})
vectors, err := c.Embed(ctx, texts)
```

- `BaseURL` takes the API root or the full `/embeddings` URL.
- `Dimensions` is enforced on every vector: a model swapped behind the same
  URL fails loudly instead of writing vectors of the wrong size into a
  `vector(N)` column. `RequestDimensions` also sends it, for models that
  truncate their output (Matryoshka); leave it off for models that reject the
  field.
- Texts go in batches of `BatchSize` (64), and come back in input order even
  when the provider reorders `data`.
- 429 and 5xx answers are retried `MaxRetries` times (2), honouring
  `Retry-After` up to 10 s; other errors return at once as `*StatusError`.

`FromEnv("EMBEDDING_")` reads `EMBEDDING_BASE_URL`, `_API_KEY`, `_MODEL`,
`_DIMENSIONS` and `_REQUEST_DIMENSIONS`, and returns `nil, nil` when the base
URL is unset, so a service can run without an endpoint and fall back to
lexical search.

## Usage

```go
c, _ := embedding.New(embedding.Config{..., Meter: meter})
vectors, _ := c.Embed(embedding.Attribute(ctx, userID, "search"), texts)
```

The `Meter` receives the model, the prompt tokens of each request, and the
subject and purpose `Attribute` put on the context.

## Rerank

```go
r, _ := embedding.NewReranker(embedding.RerankConfig{BaseURL: url, APIKey: key, Model: "bge-reranker-v2-m3"})
ranked, _ := r.Rerank(ctx, query, documents, 10) // []Ranked{Index, Score}, best first
```

Equal scores keep the caller's order. Rerank a first stage that over-fetched:
retrieval optimises recall, the reranker precision.
