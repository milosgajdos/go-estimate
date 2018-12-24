package estimate

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewBase(t *testing.T) {
	assert := assert.New(t)

	state := mat.NewVecDense(2, []float64{1.0, 1.0})
	output := mat.NewVecDense(1, []float64{1.0})
	cov := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	b, err := NewBase(state, output)
	assert.NotNil(b)
	assert.NoError(err)

	b, err = NewBaseWithCov(state, output, cov)
	assert.NotNil(b)
	assert.NoError(err)

	b, err = NewBaseWithCov(state, output, mat.NewSymDense(1, []float64{1.0}))
	assert.Nil(b)
	assert.Error(err)
}

func TestStateOutputCovariance(t *testing.T) {
	assert := assert.New(t)

	state := mat.NewVecDense(2, []float64{1.0, 2.0})
	output := mat.NewVecDense(1, []float64{1.0})
	cov := mat.NewSymDense(2, []float64{1.0, 2.0, 2.0, 4.0})

	b, err := NewBaseWithCov(state, output, cov)
	assert.NotNil(b)
	assert.NoError(err)

	s := b.State()
	for i := 0; i < state.Len(); i++ {
		assert.Equal(s.AtVec(i), b.State().AtVec(i))
	}

	o := b.Output()
	for i := 0; i < output.Len(); i++ {
		assert.Equal(o.AtVec(i), b.Output().AtVec(i))
	}

	r, c := b.Covariance().Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			assert.Equal(cov.At(i, j), b.cov.At(i, j))
		}
	}
}
