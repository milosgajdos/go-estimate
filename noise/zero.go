package noise

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Zero is zero noise i.e. no noise
type Zero struct {
	// mean stores zero mean values
	mean []float64
	// cov is zero covariance matrix
	cov *mat.SymDense
}

// NewZero creates new zero noise i.e. zero mean and zero covariance.
// It returns error if size is non-positive.
func NewZero(size int) (*Zero, error) {
	if size < 0 {
		return nil, fmt.Errorf("Invalid noise dimension: %d", size)
	}

	mean := make([]float64, size)
	cov := mat.NewSymDense(size, nil)

	return &Zero{
		mean: mean,
		cov:  cov,
	}, nil
}

// Sample generates empty sample and returns it: a vector with zero values.
func (e *Zero) Sample() mat.Vector {
	return mat.NewVecDense(len(e.mean), nil)
}

// Cov returns empty covariance matrix: symmetric matrix with zero values.
func (e *Zero) Cov() mat.Symmetric {
	cov := mat.NewSymDense(e.cov.Symmetric(), nil)
	cov.CopySym(e.cov)

	return cov
}

// Mean returns Zero mean.
func (e *Zero) Mean() []float64 {
	mean := make([]float64, len(e.mean))
	copy(mean, e.mean)

	return mean
}

// Reset resets Zero noise: it resets the noise seed.
// It returns error if it fails to reset the noise.
func (e *Zero) Reset() {}

// String implements the Stringer interface.
func (e *Zero) String() string {
	return fmt.Sprintf("Zero{\nMean=%v\nCov=%v\n}", e.Mean(), mat.Formatted(e.Cov(), mat.Prefix("    "), mat.Squeeze()))
}
