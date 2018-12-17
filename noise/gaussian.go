package noise

import (
	"fmt"
	"time"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// Gaussian is gaussian noise
type Gaussian struct {
	// dist is a multivariate normal distribution
	dist *distmv.Normal
	// mean is Gaussian mean
	mean []float64
	// cov is Gaussian covariance
	cov mat.Symmetric
}

// NewGaussian creates new Gaussian noise with given mean and covariance.
// It returns error if it fails to create Gaussian.
func NewGaussian(mean []float64, cov mat.Symmetric) (*Gaussian, error) {
	dist, ok := newGaussianDist(mean, cov)
	if !ok {
		return nil, fmt.Errorf("Failed to create new Gaussian noise")
	}

	return &Gaussian{
		dist: dist,
		mean: mean,
		cov:  cov,
	}, nil
}

// Sample generates a sample from Gaussian noise and returns it.
func (g *Gaussian) Sample() mat.Vector {
	r := g.dist.Rand(nil)
	return mat.NewVecDense(len(r), r)
}

// Cov returns covariance matrix of Gaussian noise.
func (g *Gaussian) Cov() mat.Symmetric {
	return g.cov
}

// Mean returns Gaussian mean.
func (g *Gaussian) Mean() []float64 {
	return g.mean
}

// Reset resets Gaussian noise.
// It returns error if it fails to reset the noise.
func (g *Gaussian) Reset() error {
	dist, ok := newGaussianDist(g.mean, g.cov)
	if !ok {
		return fmt.Errorf("Failed to reset Gaussian noise")
	}
	g.dist = dist

	return nil
}

func newGaussianDist(mean []float64, cov mat.Symmetric) (*distmv.Normal, bool) {
	seed := rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
	// cov is square; rows and cols are the same size
	size, _ := cov.Dims()
	return distmv.NewNormal(make([]float64, size), cov, seed)
}

// String implements the Stringer interface.
func (g *Gaussian) String() string {
	return fmt.Sprintf("Gaussian{\nMean=%v\nCov=%v\n}", g.mean, mat.Formatted(g.cov, mat.Prefix("    "), mat.Squeeze()))
}
