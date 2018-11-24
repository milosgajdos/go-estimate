package bootstrap

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

type mockPropagator struct{}

func (p *mockPropagator) Propage(*mat.Dense, *mat.Dense) (*mat.Dense, error) {
	return new(mat.Dense), nil
}

type mockObserver struct{}

func (o *mockObserver) Observe(*mat.Dense, *mat.Dense) (*mat.Dense, error) {
	return new(mat.Dense), nil
}

func TestNew(t *testing.T) {
	assert := assert.New(t)

	c := &Config{
		Propagator: &mockPropagator{},
		Observer:   &mockObserver{},
	}

	f, err := New(c)
	assert.NotNil(f)
	assert.NoError(err)
}
