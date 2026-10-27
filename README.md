# searchkit

Go libraries for finding and admitting content. Each directory is its own module, tagged `<dir>/vX.Y.Z`.

| Module | Purpose |
| --- | --- |
| `search` | Web search and page fetching behind one interface (SearXNG, Tornade). `go get github.com/lalternativefabrique/searchkit/search@search/v0.10.0` |
| `fileguard` | Admission checks on bytes and URLs a service did not choose: SSRF guard, magic-byte sniffing, size limits. Standard library only. `go get github.com/lalternativefabrique/searchkit/fileguard@fileguard/v0.6.0` |
| `retrieval` | Decides which candidates of a similarity search are worth keeping. `go get github.com/lalternativefabrique/searchkit/retrieval@retrieval/v0.2.0` |
| `language` | ISO 639-1 language of a text: `Normalize`, offline `Detect`, `Or`. `go get github.com/lalternativefabrique/searchkit/language@language/v0.2.0` |
