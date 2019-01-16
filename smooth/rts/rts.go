package rts

import (
	filter "github.com/milosgajdos83/go-filter"
	"gonum.org/v1/gonum/mat"
)

// RTS is Rauch-Tung-Striebel smoother
type RTS struct {
	// c is a smoothing matrix
	c *mat.Dense
}

// New creates new RTS and returns it.
// It returns error if it fails to create RTS filter.
func New(init filter.InitCond) (*RTS, error) {
	dim := init.State().Len()

	c := mat.NewDense(dim, dim, nil)

	return &RTS{
		c: c,
	}, nil
}

// Smooth implements Rauch-Tung-Striebel smoothing algorithm.
// It uses updated estimates u, predicted estimates p and dynamics matrices m and returns smoothed estimates.
// It returns error if either the sizes of p and u differ or the smoothed estimates failed to be calculated.
func (s *RTS) Smooth(p []filter.Estimate, u []filter.Estimate, m []mat.Matrix) ([]filter.Estimate, error) {
	return nil, nil
}
