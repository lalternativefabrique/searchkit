package embedding

import "context"

// Usage is what one embedding request cost at the provider, in tokens.
type Usage struct {
	Model        string
	PromptTokens int
	// Subject and Purpose come from Attribute: whose text it was and what for.
	Subject string
	Purpose string
}

// Meter is told about every embedding request that reported usage.
type Meter interface {
	Embedded(ctx context.Context, u Usage)
}

type attribution struct{ subject, purpose string }

type attributionKey struct{}

// Attribute marks ctx with whose text is embedded and why, for the Meter.
func Attribute(ctx context.Context, subject, purpose string) context.Context {
	return context.WithValue(ctx, attributionKey{}, attribution{subject: subject, purpose: purpose})
}

func attributionFrom(ctx context.Context) attribution {
	a, _ := ctx.Value(attributionKey{}).(attribution)
	return a
}
