package rnd

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestWithCovN(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 0.0, 0.0, 1.0}
	covTest := mat.NewSymDense(2, data)
	covR, _ := covTest.Dims()

	// n must be bigger than 1
	nTest := -3
	res, err := WithCovN(covTest, nTest)
	assert.Error(err)
	assert.Nil(res)

	nTest = 1
	res, err = WithCovN(covTest, nTest)
	assert.NoError(err)
	assert.NotNil(res)

	// 2 samples
	nTest = 2
	res, err = WithCovN(covTest, nTest)
	assert.NoError(err)
	assert.NotNil(res)
	r, c := res.Dims()
	assert.Equal(r, covR)
	assert.Equal(c, nTest)
}

func TestRouletteDrawN(t *testing.T) {
	assert := assert.New(t)

	// p can't be nil or empty
	indices, err := RouletteDrawN(nil, 10)
	assert.Error(err)
	assert.Nil(indices)

	p := []float64{0.1, 0.7, 0.3, 0.4}
	n := 10
	indices, err = RouletteDrawN(p, n)
	assert.NoError(err)
	assert.NotNil(indices)
	assert.Equal(n, len(indices))
}
