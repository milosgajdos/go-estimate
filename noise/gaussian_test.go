package noise

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewGaussian(t *testing.T) {
	assert := assert.New(t)

	for _, test := range []struct {
		mean []float64
		cov  *mat.SymDense
	}{
		{
			mean: []float64{2, 3},
			cov:  mat.NewSymDense(2, []float64{1, 0.1, 0.1, 1}),
		},
	} {
		g, err := NewGaussian(test.mean, test.cov)
		assert.NotNil(g)
		assert.NoError(err)
	}

	// invalid mean
	mean := []float64{2}
	cov := mat.NewSymDense(2, []float64{1, 0.1, 0.1, 1})
	g, err := NewGaussian(mean, cov)
	assert.Nil(g)
	assert.Error(err)

	// invalid covariance: not positive definite
	mean = []float64{2, 3}
	cov = mat.NewSymDense(2, []float64{1, 2, 2, 1})
	g, err = NewGaussian(mean, cov)
	assert.Nil(g)
	assert.Error(err)
}

func TestGaussianMeanCov(t *testing.T) {
	assert := assert.New(t)

	mean := []float64{2, 3}
	cov := mat.NewSymDense(2, []float64{1, 0.1, 0.1, 1})

	for _, test := range []struct {
		mean []float64
		cov  *mat.SymDense
	}{
		{
			mean: mean,
			cov:  cov,
		},
	} {
		g, err := NewGaussian(test.mean, test.cov)
		assert.NotNil(g)
		assert.NoError(err)

		gCov := g.Cov()
		assert.Equal(cov.SymmetricDim(), gCov.SymmetricDim())

		rows, cols := gCov.Dims()
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				if gCov.At(r, c) != cov.At(r, c) {
					t.Errorf("Incorrect covariance matrix returned")
				}
			}
		}

		gMean := g.Mean()
		assert.EqualValues(mean, gMean)
	}
}

func TestGaussianSample(t *testing.T) {
	assert := assert.New(t)

	for _, test := range []struct {
		mean []float64
		cov  *mat.SymDense
	}{
		{
			mean: []float64{2, 3},
			cov:  mat.NewSymDense(2, []float64{1, 0.1, 0.1, 1}),
		},
	} {
		g, err := NewGaussian(test.mean, test.cov)
		assert.NotNil(g)
		assert.NoError(err)

		sample := g.Sample()
		r, _ := sample.Dims()
		assert.Equal(r, len(test.mean))
	}
}

func TestGaussianReset(t *testing.T) {
	assert := assert.New(t)

	mean := []float64{2, 3}
	cov := mat.NewSymDense(2, []float64{1, 0.1, 0.1, 1})

	g, err := NewGaussian(mean, cov)
	assert.NotNil(g)
	assert.NoError(err)

	sample1 := g.Sample()

	g.Reset()
	assert.NoError(err)

	sample2 := g.Sample()
	assert.NotEqual(sample1, sample2)
}

func TestGaussianString(t *testing.T) {
	assert := assert.New(t)

	str := `Gaussian{
Mean=[2 3]
Cov=⎡  1  0.1⎤
    ⎣0.1    1⎦
}`
	mean := []float64{2, 3}
	cov := mat.NewSymDense(2, []float64{1, 0.1, 0.1, 1})

	g, err := NewGaussian(mean, cov)
	assert.NotNil(g)
	assert.NoError(err)
	assert.Equal(str, g.String())
}
