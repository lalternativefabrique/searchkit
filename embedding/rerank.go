package embedding

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// RerankConfig describes one Cohere-compatible /rerank endpoint (Cohere,
// Voyage, or a self-hosted cross-encoder behind the same contract).
type RerankConfig struct {
	// BaseURL is the API root (https://host/v1) or the full /rerank URL.
	BaseURL    string
	APIKey     string
	Model      string
	MaxRetries int
	Timeout    time.Duration
	HTTPClient *http.Client
}

// Reranker scores documents against a query.
type Reranker struct {
	cfg  RerankConfig
	url  string
	http *http.Client
}

// Ranked is one document of a rerank answer.
type Ranked struct {
	Index int
	Score float64
}

// NewReranker validates cfg and returns a Reranker.
func NewReranker(cfg RerankConfig) (*Reranker, error) {
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, errors.New("rerank: base URL is required")
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return nil, errors.New("rerank: model is required")
	}
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = 2
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{}
	}
	return &Reranker{cfg: cfg, url: endpoint(cfg.BaseURL, "/rerank"), http: hc}, nil
}

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// Rerank returns at most topK documents, best first. Equal scores keep the
// caller's order. topK <= 0 returns them all.
func (r *Reranker) Rerank(ctx context.Context, query string, documents []string, topK int) ([]Ranked, error) {
	if len(documents) == 0 {
		return nil, nil
	}
	if topK <= 0 || topK > len(documents) {
		topK = len(documents)
	}
	var resp rerankResponse
	req := rerankRequest{Model: r.cfg.Model, Query: query, Documents: documents, TopN: topK}
	if err := post(ctx, r.http, r.url, r.cfg.APIKey, r.cfg.Timeout, r.cfg.MaxRetries, req, &resp); err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}
	seen := make(map[int]bool, len(resp.Results))
	out := make([]Ranked, 0, len(resp.Results))
	for _, res := range resp.Results {
		if res.Index < 0 || res.Index >= len(documents) || seen[res.Index] {
			return nil, fmt.Errorf("rerank: invalid or repeated index %d for %d documents", res.Index, len(documents))
		}
		seen[res.Index] = true
		out = append(out, Ranked{Index: res.Index, Score: res.RelevanceScore})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Index < out[j].Index
	})
	if len(out) > topK {
		out = out[:topK]
	}
	return out, nil
}
