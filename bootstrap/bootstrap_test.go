package bootstrap

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type mockPropagator struct{}

func (p *mockPropagator) Propagate(*mat.Dense, *mat.Dense) (*mat.Dense, error) {
	return new(mat.Dense), nil
}

type mockObserver struct{}

func (o *mockObserver) Observe(*mat.Dense, *mat.Dense) (*mat.Dense, error) {
	return new(mat.Dense), nil
}

func TestNew(t *testing.T) {
	assert := assert.New(t)

	pCount := 10
	alpha := 10.0
	errDist, _ := distmv.NewNormal([]float64{0, 0}, mat.NewSymDense(2, []float64{1, 0, 0, 1}), nil)

	stateVals := []float64{1.0, 1.0}
	state := mat.NewDense(2, 1, stateVals)
	covVals := []float64{1.0, 0.0, 0.0, 1.0}
	cov := mat.NewDense(2, 2, covVals)

	ic := &InitCond{
		State: state,
		Cov:   cov,
	}

	c := &Config{
		Propagator:    &mockPropagator{},
		Observer:      &mockObserver{},
		ParticleCount: pCount,
		Alpha:         alpha,
		ErrDist:       errDist,
	}

	f, err := New(c, ic)
	assert.NotNil(f)
	assert.NoError(err)

	// invalid count
	c.ParticleCount = -10
	f, err = New(c, ic)
	assert.Nil(f)
	assert.Error(err)
}
