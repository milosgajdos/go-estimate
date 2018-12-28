package noise

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// None is noise with empty mean and zero covariance matrix.
// None is different from None: its mean vector length is 0 and its covariance matrix is zero size.
type None struct{}

// NewNone creates new None noise and returns it
func NewNone() (*None, error) {
	return &None{}, nil
}

// Sample returns zero size vector.
func (e *None) Sample() mat.Vector {
	sample := &mat.VecDense{}

	return sample
}

// Cov returns zero size covariance matrix.
func (e *None) Cov() mat.Symmetric {
	cov := &mat.SymDense{}

	return cov
}

// Mean returns None mean.
func (e *None) Mean() []float64 {
	var mean []float64

	return mean
}

// Reset does nothing: it's here to implement filter.Noise interface
func (e *None) Reset() {}

// String implements the Stringer interface.
func (e *None) String() string {
	return fmt.Sprintf("None{\nMean=%v\nCov=%v\n}", e.Mean(), mat.Formatted(e.Cov(), mat.Prefix("    "), mat.Squeeze()))
}
