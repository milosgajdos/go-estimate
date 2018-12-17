package estimate

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestBase(t *testing.T) {
	assert := assert.New(t)

	state := mat.NewVecDense(2, []float64{1.0, 1.0})
	output := mat.NewVecDense(1, []float64{1.0})

	b := NewBase(state, output)
	assert.NotNil(b)

	for i := 0; i < state.Len(); i++ {
		assert.Equal(state.AtVec(i), b.State().AtVec(i))
	}

	for i := 0; i < output.Len(); i++ {
		assert.Equal(output.AtVec(i), b.Output().AtVec(i))
	}
}

func TestCovariance(t *testing.T) {
	assert := assert.New(t)

	state := mat.NewVecDense(2, []float64{1.0, 2.0})
	output := mat.NewVecDense(1, []float64{1.0})
	cov := mat.NewDense(2, 2, []float64{1.0, 2.0, 2.0, 4.0})

	b := NewBase(state, output)
	assert.NotNil(b)
	bCov := b.Covariance()

	rows, cols := cov.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			assert.Equal(cov.At(r, c), bCov.At(r, c))
		}
	}
}
