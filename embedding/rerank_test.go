package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func rerankServer(t *testing.T, body string, got *rerankRequest) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" {
			http.NotFound(w, r)
			return
		}
		if got != nil {
			_ = json.NewDecoder(r.Body).Decode(got)
		}
		_, _ = w.Write([]byte(body))
	}))
}

func TestRerankSortsByScoreAndKeepsOrderOnTies(t *testing.T) {
	var req rerankRequest
	srv := rerankServer(t, `{"results":[
		{"index":2,"relevance_score":0.5},
		{"index":0,"relevance_score":0.9},
		{"index":3,"relevance_score":0.5},
		{"index":1,"relevance_score":0.1}]}`, &req)
	defer srv.Close()
	r, err := NewReranker(RerankConfig{BaseURL: srv.URL + "/v1", Model: "rr"})
	if err != nil {
		t.Fatal(err)
	}

	got, err := r.Rerank(context.Background(), "q", []string{"a", "b", "c", "d"}, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []Ranked{{0, 0.9}, {2, 0.5}, {3, 0.5}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	if req.TopN != 3 || req.Query != "q" || req.Model != "rr" {
		t.Errorf("sent %+v", req)
	}
}

func TestRerankRejectsBadIndices(t *testing.T) {
	for name, body := range map[string]string{
		"range":     `{"results":[{"index":7,"relevance_score":1}]}`,
		"duplicate": `{"results":[{"index":0,"relevance_score":1},{"index":0,"relevance_score":0.5}]}`,
	} {
		srv := rerankServer(t, body, nil)
		r, _ := NewReranker(RerankConfig{BaseURL: srv.URL + "/v1/rerank", Model: "rr"})
		if _, err := r.Rerank(context.Background(), "q", []string{"a", "b"}, 0); err == nil {
			t.Errorf("%s: accepted", name)
		}
		srv.Close()
	}
}

func TestRerankEmptyAndConfig(t *testing.T) {
	if _, err := NewReranker(RerankConfig{Model: "rr"}); err == nil {
		t.Error("missing base URL accepted")
	}
	r, _ := NewReranker(RerankConfig{BaseURL: "http://127.0.0.1:1", Model: "rr"})
	got, err := r.Rerank(context.Background(), "q", nil, 5)
	if got != nil || err != nil {
		t.Errorf("empty documents gave %v, %v", got, err)
	}

	t.Setenv("R_BASE_URL", "")
	if r, err := RerankerFromEnv("R_"); r != nil || err != nil {
		t.Errorf("unset base URL gave %v, %v", r, err)
	}
}
