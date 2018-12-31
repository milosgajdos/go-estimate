package estimate

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewBase(t *testing.T) {
	assert := assert.New(t)

	val := mat.NewVecDense(2, []float64{1.0, 1.0})
	cov := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	b, err := NewBase(val)
	assert.NotNil(b)
	assert.NoError(err)

	b, err = NewBaseWithCov(val, cov)
	assert.NotNil(b)
	assert.NoError(err)

	b, err = NewBaseWithCov(val, mat.NewSymDense(1, []float64{1.0}))
	assert.Nil(b)
	assert.Error(err)
}

func TestValCov(t *testing.T) {
	assert := assert.New(t)

	val := mat.NewVecDense(2, []float64{1.0, 2.0})
	cov := mat.NewSymDense(2, []float64{1.0, 2.0, 2.0, 4.0})

	b, err := NewBaseWithCov(val, cov)
	assert.NotNil(b)
	assert.NoError(err)

	v := b.Val()
	for i := 0; i < val.Len(); i++ {
		assert.Equal(v.AtVec(i), b.Val().AtVec(i))
	}

	r, c := b.Cov().Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			assert.Equal(cov.At(i, j), b.cov.At(i, j))
		}
	}
}
