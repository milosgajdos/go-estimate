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
}

// NewGaussian creates new Gaussian noise with given mean and covariance.
// It returns error if it fails to create Gaussian distribution handle.
func NewGaussian(mean []float64, cov mat.Symmetric) (*Gaussian, error) {
	dist, ok := newGaussianDist(mean, cov)
	if !ok {
		return nil, fmt.Errorf("Failed to create new Gaussian noise")
	}

	return &Gaussian{
		dist: dist,
	}, nil
}

// Sample generates a random sample from Gaussian distribution and returns it.
func (g *Gaussian) Sample() mat.Vector {
	r := g.dist.Rand(nil)
	return mat.NewVecDense(len(r), r)
}

// Cov returns covariance matrix of Gaussian noise.
func (g *Gaussian) Cov() mat.Symmetric {
	return g.dist.CovarianceMatrix(nil)
}

// Mean returns Gaussian mean.
func (g *Gaussian) Mean() []float64 {
	return g.dist.Mean(nil)
}

// Reset resets Gaussian noise: it resets the noise seed.
// It returns error if it fails to reset the noise.
func (g *Gaussian) Reset() {
	dist, ok := newGaussianDist(g.Mean(), g.Cov())
	if !ok {
		panic("Failed to reset Gaussian noise")
	}
	g.dist = dist
}

// String implements the Stringer interface.
func (g *Gaussian) String() string {
	return fmt.Sprintf("Gaussian{\nMean=%v\nCov=%v\n}", g.Mean(), mat.Formatted(g.Cov(), mat.Prefix("    "), mat.Squeeze()))
}

func newGaussianDist(mean []float64, cov mat.Symmetric) (*distmv.Normal, bool) {
	seed := rand.New(rand.NewSource(uint64(time.Now().UnixNano())))

	return distmv.NewNormal(mean, cov, seed)
}
