package embedding

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// FromEnv builds a Client from <prefix>BASE_URL, <prefix>API_KEY,
// <prefix>MODEL and <prefix>DIMENSIONS. It returns (nil, nil) when BASE_URL
// is unset, so a service can run without an endpoint. A set DIMENSIONS is
// both enforced and requested; <prefix>REQUEST_DIMENSIONS=false only enforces it.
func FromEnv(prefix string) (*Client, error) {
	base := strings.TrimSpace(os.Getenv(prefix + "BASE_URL"))
	if base == "" {
		return nil, nil
	}
	cfg := Config{BaseURL: base, APIKey: os.Getenv(prefix + "API_KEY"), Model: strings.TrimSpace(os.Getenv(prefix + "MODEL"))}
	if v := strings.TrimSpace(os.Getenv(prefix + "DIMENSIONS")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("embedding: %sDIMENSIONS: %w", prefix, err)
		}
		cfg.Dimensions = n
		cfg.RequestDimensions = n > 0
	}
	if v := strings.TrimSpace(os.Getenv(prefix + "REQUEST_DIMENSIONS")); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("embedding: %sREQUEST_DIMENSIONS: %w", prefix, err)
		}
		cfg.RequestDimensions = b && cfg.Dimensions > 0
	}
	return New(cfg)
}

// RerankerFromEnv builds a Reranker from <prefix>BASE_URL, <prefix>API_KEY
// and <prefix>MODEL, or returns (nil, nil) when BASE_URL is unset.
func RerankerFromEnv(prefix string) (*Reranker, error) {
	base := strings.TrimSpace(os.Getenv(prefix + "BASE_URL"))
	if base == "" {
		return nil, nil
	}
	return NewReranker(RerankConfig{BaseURL: base, APIKey: os.Getenv(prefix + "API_KEY"), Model: strings.TrimSpace(os.Getenv(prefix + "MODEL"))})
}
