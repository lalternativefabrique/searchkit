package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func init() { backoffBase = time.Millisecond }

type fakeEmbeddings struct {
	dims     int
	reversed bool
	usage    int
	mu       sync.Mutex
	requests []embeddingRequest
	auth     []string
	paths    []string
}

func (f *fakeEmbeddings) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var req embeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	f.requests = append(f.requests, req)
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	f.paths = append(f.paths, r.URL.Path)
	f.mu.Unlock()

	type item struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	items := make([]item, len(req.Input))
	for i, text := range req.Input {
		v := make([]float32, f.dims)
		if f.dims > 0 {
			v[0] = float32(len(text))
		}
		items[i] = item{Index: i, Embedding: v}
	}
	if f.reversed {
		for i, j := 0, len(items)-1; i < j; i, j = i+1, j-1 {
			items[i], items[j] = items[j], items[i]
		}
	}
	resp := map[string]any{"data": items}
	if f.usage > 0 {
		resp["usage"] = map[string]int{"prompt_tokens": f.usage}
	}
	_ = json.NewEncoder(w).Encode(resp)
}

type recordingMeter struct{ got []Usage }

func (m *recordingMeter) Embedded(_ context.Context, u Usage) { m.got = append(m.got, u) }

func TestNewRejectsIncompleteConfig(t *testing.T) {
	cases := []Config{
		{Model: "m"},
		{BaseURL: "http://x"},
		{BaseURL: "http://x", Model: "m", Dimensions: -1},
		{BaseURL: "http://x", Model: "m", RequestDimensions: true},
	}
	for _, cfg := range cases {
		if _, err := New(cfg); err == nil {
			t.Errorf("New(%+v) accepted an invalid config", cfg)
		}
	}
}

func TestEmbedAcceptsBaseOrFullURL(t *testing.T) {
	f := &fakeEmbeddings{dims: 3}
	srv := httptest.NewServer(f)
	defer srv.Close()

	for _, base := range []string{srv.URL + "/v1", srv.URL + "/v1/", srv.URL + "/v1/embeddings"} {
		c, err := New(Config{BaseURL: base, Model: "m", APIKey: "k"})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.Embed(context.Background(), []string{"a"}); err != nil {
			t.Fatalf("%s: %v", base, err)
		}
	}
	for i, p := range f.paths {
		if p != "/v1/embeddings" {
			t.Errorf("request %d went to %q", i, p)
		}
		if f.auth[i] != "Bearer k" {
			t.Errorf("request %d sent Authorization %q", i, f.auth[i])
		}
	}
}

func TestEmbedBatchesAndKeepsInputOrder(t *testing.T) {
	f := &fakeEmbeddings{dims: 2, reversed: true}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c, _ := New(Config{BaseURL: srv.URL, Model: "m", BatchSize: 2})

	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	vectors, err := c.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.requests) != 3 {
		t.Fatalf("sent %d requests, want 3", len(f.requests))
	}
	for i, v := range vectors {
		if int(v[0]) != len(texts[i]) {
			t.Errorf("vector %d belongs to a text of length %v, want %d", i, v[0], len(texts[i]))
		}
	}
}

func TestEmbedEmptyInputCallsNothing(t *testing.T) {
	f := &fakeEmbeddings{dims: 2}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c, _ := New(Config{BaseURL: srv.URL, Model: "m"})

	vectors, err := c.Embed(context.Background(), nil)
	if err != nil || vectors != nil || len(f.requests) != 0 {
		t.Fatalf("got %v, %v after %d requests", vectors, err, len(f.requests))
	}
}

func TestEmbedEnforcesAndRequestsDimensions(t *testing.T) {
	f := &fakeEmbeddings{dims: 4}
	srv := httptest.NewServer(f)
	defer srv.Close()

	enforced, _ := New(Config{BaseURL: srv.URL, Model: "m", Dimensions: 8})
	if _, err := enforced.Embed(context.Background(), []string{"a"}); err == nil || !strings.Contains(err.Error(), "want 8") {
		t.Fatalf("a 4-dimension vector passed an 8-dimension client: %v", err)
	}
	if f.requests[0].Dimensions != 0 {
		t.Errorf("dimensions were sent without RequestDimensions")
	}

	requested, _ := New(Config{BaseURL: srv.URL, Model: "m", Dimensions: 4, RequestDimensions: true})
	if _, err := requested.Embed(context.Background(), []string{"a"}); err != nil {
		t.Fatal(err)
	}
	if f.requests[1].Dimensions != 4 {
		t.Errorf("sent dimensions %d, want 4", f.requests[1].Dimensions)
	}
}

func TestEmbedReportsUsageWithAttribution(t *testing.T) {
	f := &fakeEmbeddings{dims: 1, usage: 7}
	srv := httptest.NewServer(f)
	defer srv.Close()
	meter := &recordingMeter{}
	c, _ := New(Config{BaseURL: srv.URL, Model: "m", Meter: meter, BatchSize: 1})

	ctx := Attribute(context.Background(), "user-1", "search")
	if _, err := c.Embed(ctx, []string{"a", "b"}); err != nil {
		t.Fatal(err)
	}
	want := Usage{Model: "m", PromptTokens: 7, Subject: "user-1", Purpose: "search"}
	if len(meter.got) != 2 || meter.got[0] != want || meter.got[1] != want {
		t.Fatalf("meter got %+v, want two of %+v", meter.got, want)
	}
}

func TestEmbedRetriesRateLimitsAndServerErrors(t *testing.T) {
	var calls atomic.Int32
	f := &fakeEmbeddings{dims: 1}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch calls.Add(1) {
		case 1:
			http.Error(w, "slow down", http.StatusTooManyRequests)
		case 2:
			http.Error(w, "oops", http.StatusBadGateway)
		default:
			f.ServeHTTP(w, r)
		}
	}))
	defer srv.Close()
	c, _ := New(Config{BaseURL: srv.URL, Model: "m"})

	if _, err := c.Embed(context.Background(), []string{"a"}); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 3 {
		t.Fatalf("made %d calls, want 3", calls.Load())
	}
}

func TestEmbedDoesNotRetryClientErrors(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, strings.Repeat("x", 4096), http.StatusUnauthorized)
	}))
	defer srv.Close()
	c, _ := New(Config{BaseURL: srv.URL, Model: "m"})

	_, err := c.Embed(context.Background(), []string{"a"})
	var se *StatusError
	if !errors.As(err, &se) || se.StatusCode != http.StatusUnauthorized {
		t.Fatalf("got %v, want a 401 StatusError", err)
	}
	if len(se.Body) > maxErrorBody {
		t.Errorf("error body kept %d bytes", len(se.Body))
	}
	if calls.Load() != 1 {
		t.Errorf("made %d calls, want 1", calls.Load())
	}
}

func TestEmbedGivesUpAfterMaxRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	c, _ := New(Config{BaseURL: srv.URL, Model: "m", MaxRetries: 1})
	if _, err := c.Embed(context.Background(), []string{"a"}); err == nil {
		t.Fatal("a failing endpoint returned no error")
	}
	if calls.Load() != 2 {
		t.Errorf("made %d calls, want 2", calls.Load())
	}

	calls.Store(0)
	noRetry, _ := New(Config{BaseURL: srv.URL, Model: "m", MaxRetries: -1})
	_, _ = noRetry.Embed(context.Background(), []string{"a"})
	if calls.Load() != 1 {
		t.Errorf("MaxRetries -1 made %d calls, want 1", calls.Load())
	}
}

func TestEmbedStopsWaitingWhenContextEnds(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "10")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	c, _ := New(Config{BaseURL: srv.URL, Model: "m"})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := c.Embed(ctx, []string{"a"}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want the context deadline", err)
	}
	if time.Since(start) > 2*time.Second {
		t.Errorf("waited %s past the context deadline", time.Since(start))
	}
}

func TestEmbedRejectsMismatchedResponses(t *testing.T) {
	for name, body := range map[string]string{
		"count":     `{"data":[{"embedding":[1]}]}`,
		"duplicate": `{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}`,
		"range":     `{"data":[{"index":0,"embedding":[1]},{"index":5,"embedding":[2]}]}`,
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(body))
		}))
		c, _ := New(Config{BaseURL: srv.URL, Model: "m"})
		if _, err := c.Embed(context.Background(), []string{"a", "b"}); err == nil {
			t.Errorf("%s: a mismatched response was accepted", name)
		}
		srv.Close()
	}
}

func TestFromEnv(t *testing.T) {
	t.Setenv("X_BASE_URL", "")
	c, err := FromEnv("X_")
	if c != nil || err != nil {
		t.Fatalf("unset base URL gave %v, %v; want nil, nil", c, err)
	}

	t.Setenv("X_BASE_URL", "http://h/v1")
	t.Setenv("X_MODEL", "qwen3-embedding-8b")
	t.Setenv("X_DIMENSIONS", "1024")
	c, err = FromEnv("X_")
	if err != nil {
		t.Fatal(err)
	}
	if c.Model() != "qwen3-embedding-8b" || c.Dimensions() != 1024 || !c.cfg.RequestDimensions {
		t.Fatalf("got %+v", c.cfg)
	}

	t.Setenv("X_REQUEST_DIMENSIONS", "false")
	c, _ = FromEnv("X_")
	if c.cfg.RequestDimensions || c.Dimensions() != 1024 {
		t.Fatalf("REQUEST_DIMENSIONS=false gave %+v", c.cfg)
	}

	t.Setenv("X_DIMENSIONS", "many")
	if _, err := FromEnv("X_"); err == nil {
		t.Fatal("a non-numeric DIMENSIONS was accepted")
	}
}
