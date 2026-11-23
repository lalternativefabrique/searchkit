// Package embedding reaches OpenAI-compatible embedding endpoints and
// Cohere-compatible rerank endpoints.
package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Embedder turns texts into vectors, one per text, in input order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Model() string
}

// Config describes one embedding endpoint.
type Config struct {
	// BaseURL is the API root (https://host/v1) or the full /embeddings URL.
	BaseURL string
	APIKey  string
	Model   string
	// Dimensions is the vector size every response must have. Zero accepts any.
	Dimensions int
	// RequestDimensions sends Dimensions in the request, for models that can
	// truncate their output (Matryoshka models such as qwen3-embedding).
	RequestDimensions bool
	// BatchSize caps the texts sent per request. Zero means 64.
	BatchSize int
	// MaxRetries bounds the retries on 429 and 5xx answers. Negative disables them.
	MaxRetries int
	// Timeout bounds each request. Zero means 60s.
	Timeout    time.Duration
	HTTPClient *http.Client
	Meter      Meter
}

// Client is an Embedder over one OpenAI-compatible endpoint.
type Client struct {
	cfg  Config
	url  string
	http *http.Client
}

var _ Embedder = (*Client)(nil)

// New validates cfg and returns a Client.
func New(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, errors.New("embedding: base URL is required")
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return nil, errors.New("embedding: model is required")
	}
	if cfg.Dimensions < 0 {
		return nil, errors.New("embedding: dimensions cannot be negative")
	}
	if cfg.RequestDimensions && cfg.Dimensions == 0 {
		return nil, errors.New("embedding: RequestDimensions needs Dimensions")
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
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
	return &Client{cfg: cfg, url: endpoint(cfg.BaseURL, "/embeddings"), http: hc}, nil
}

func (c *Client) Model() string { return c.cfg.Model }

// Dimensions is the vector size the client enforces, zero when it accepts any.
func (c *Client) Dimensions() int { return c.cfg.Dimensions }

type embeddingRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type embeddingResponse struct {
	Data []struct {
		Index     *int      `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Usage *struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
}

// Embed sends texts in batches and returns their vectors in input order.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += c.cfg.BatchSize {
		end := min(start+c.cfg.BatchSize, len(texts))
		vectors, err := c.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vectors...)
	}
	return out, nil
}

func (c *Client) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	req := embeddingRequest{Model: c.cfg.Model, Input: texts}
	if c.cfg.RequestDimensions {
		req.Dimensions = c.cfg.Dimensions
	}
	var resp embeddingResponse
	if err := post(ctx, c.http, c.url, c.cfg.APIKey, c.cfg.Timeout, c.cfg.MaxRetries, req, &resp); err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("embedding: got %d vectors for %d inputs", len(resp.Data), len(texts))
	}
	vectors := make([][]float32, len(texts))
	for i, d := range resp.Data {
		at := i
		if d.Index != nil {
			at = *d.Index
		}
		if at < 0 || at >= len(texts) || vectors[at] != nil {
			return nil, fmt.Errorf("embedding: invalid or repeated index %d", at)
		}
		if c.cfg.Dimensions > 0 && len(d.Embedding) != c.cfg.Dimensions {
			return nil, fmt.Errorf("embedding: vector %d has %d dimensions, want %d", at, len(d.Embedding), c.cfg.Dimensions)
		}
		vectors[at] = d.Embedding
	}
	if c.cfg.Meter != nil && resp.Usage != nil && resp.Usage.PromptTokens > 0 {
		a := attributionFrom(ctx)
		c.cfg.Meter.Embedded(ctx, Usage{Model: c.cfg.Model, PromptTokens: resp.Usage.PromptTokens, Subject: a.subject, Purpose: a.purpose})
	}
	return vectors, nil
}

// StatusError is a non-2xx answer from the endpoint.
type StatusError struct {
	StatusCode int
	Body       string
}

func (e *StatusError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("status %d", e.StatusCode)
	}
	return fmt.Sprintf("status %d: %s", e.StatusCode, e.Body)
}

func retryable(code int) bool {
	return code == http.StatusTooManyRequests || code >= 500
}

const maxErrorBody = 512

func post(ctx context.Context, hc *http.Client, url, apiKey string, timeout time.Duration, maxRetries int, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return fmt.Errorf("encode request: %w", err)
	}
	for attempt := 0; ; attempt++ {
		wait, err := postOnce(ctx, hc, url, apiKey, timeout, body, out)
		if err == nil {
			return nil
		}
		var se *StatusError
		if !errors.As(err, &se) || !retryable(se.StatusCode) || attempt >= maxRetries {
			return err
		}
		if wait <= 0 {
			wait = backoff(attempt)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
	}
}

func postOnce(ctx context.Context, hc *http.Client, url, apiKey string, timeout time.Duration, body []byte, out any) (time.Duration, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := hc.Do(req)
	if err != nil {
		return 0, fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		return retryAfter(resp.Header.Get("Retry-After")), &StatusError{StatusCode: resp.StatusCode, Body: strings.TrimSpace(string(raw))}
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return 0, fmt.Errorf("decode response: %w", err)
	}
	return 0, nil
}

var backoffBase = 500 * time.Millisecond

const maxWait = 10 * time.Second

func backoff(attempt int) time.Duration {
	return min(backoffBase<<attempt, maxWait)
}

func retryAfter(v string) time.Duration {
	seconds, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil || seconds <= 0 {
		return 0
	}
	return min(time.Duration(seconds)*time.Second, maxWait)
}

func endpoint(base, path string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if strings.HasSuffix(base, path) {
		return base
	}
	return base + path
}
