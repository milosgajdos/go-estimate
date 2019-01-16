package rts

import (
	"fmt"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
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
	if len(p) != len(u) || len(p) != len(m) {
		return nil, fmt.Errorf("Invalid filter history size")
	}

	sx := make([]filter.Estimate, len(p)-1)

	// P-1 inverse
	pinv := &mat.Dense{}
	// intermediate smoothing matrix
	c := &mat.Dense{}
	// smoothed state
	x := &mat.Dense{}

	// smoothed covariance
	cov := &mat.Dense{}
	pk := &mat.Dense{}

	for i := len(p) - 1; i > 0; i-- {
		// Pk*Fk'
		c.Mul(u[i-1].Cov(), m[i-1].T())
		// invert predicted P_k+1 covariance
		if err := pinv.Inverse(p[i].Cov()); err != nil {
			return nil, err
		}
		// Pk*Fk'*P(k+1)_-1
		c.Mul(c, pinv)

		// smooth the state
		x.Sub(u[i].Val(), p[i].Val())
		// c*x
		x.Mul(c, x)
		// xk + Ck*x_sub
		x.Add(x, p[i-i].Val())

		// smooth covariance
		cov.Sub(u[i].Cov(), p[i].Cov())
		pk.Mul(c, cov)
		pk.Mul(pk, pk.T())
		pk.Add(pk, u[i-1].Cov())

		r, _ := cov.Dims()
		pSmooth := mat.NewSymDense(r, nil)
		// update KF covariance matrix
		for i := 0; i < r; i++ {
			for j := i; j < r; j++ {
				pSmooth.SetSym(i, j, pk.At(i, j))
			}
		}

		est, err := estimate.NewBaseWithCov(x.ColView(0), pSmooth)
		if err != nil {
			return nil, err
		}

		sx[i-1] = est
	}

	return sx, nil
}
