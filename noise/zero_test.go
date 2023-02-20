package noise

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewZero(t *testing.T) {
	assert := assert.New(t)

	size := 2
	for _, test := range []struct {
		mean []float64
		cov  *mat.SymDense
		size int
	}{
		{
			mean: make([]float64, size),
			cov:  mat.NewSymDense(size, nil),
			size: size,
		},
	} {
		e, err := NewZero(test.size)
		assert.NotNil(e)
		assert.NoError(err)
	}

	e, err := NewZero(-10)
	assert.Nil(e)
	assert.Error(err)
}

func TestZeroMeanCov(t *testing.T) {
	assert := assert.New(t)

	size := 2
	mean := []float64{0, 0}
	cov := mat.NewSymDense(size, []float64{0, 0, 0, 0})

	for _, test := range []struct {
		mean []float64
		cov  *mat.SymDense
		size int
	}{
		{
			mean: mean,
			cov:  cov,
			size: size,
		},
	} {
		e, err := NewZero(size)
		assert.NotNil(e)
		assert.NoError(err)

		eCov := e.Cov()
		assert.Equal(test.cov.SymmetricDim(), eCov.SymmetricDim())

		rows, cols := eCov.Dims()
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				if eCov.At(r, c) != test.cov.At(r, c) {
					t.Errorf("Incorrect covariance matrix returned")
				}
			}
		}

		eMean := e.Mean()
		assert.EqualValues(test.mean, eMean)
	}
}

func TestSample(t *testing.T) {
	assert := assert.New(t)

	size := 2
	for _, test := range []struct {
		mean []float64
		cov  *mat.SymDense
		size int
	}{
		{
			mean: []float64{0, 0},
			cov:  mat.NewSymDense(size, nil),
			size: size,
		},
	} {
		e, err := NewZero(test.size)
		assert.NotNil(e)
		assert.NoError(err)

		sample := e.Sample()
		r, _ := sample.Dims()
		assert.Equal(r, len(test.mean))
	}
}

func TestReset(t *testing.T) {
	assert := assert.New(t)

	size := 2
	e, err := NewZero(size)
	assert.NotNil(e)
	assert.NoError(err)

	sample1 := e.Sample()

	e.Reset()
	assert.NoError(err)

	sample2 := e.Sample()
	assert.Equal(sample1, sample2)
}

func TestString(t *testing.T) {
	assert := assert.New(t)

	str := `Zero{
Mean=[0 0]
Cov=⎡0  0⎤
    ⎣0  0⎦
}`

	size := 2
	e, err := NewZero(size)
	assert.NotNil(e)
	assert.NoError(err)
	assert.Equal(str, e.String())
}
