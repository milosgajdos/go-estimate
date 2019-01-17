package rts

import (
	"fmt"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/matrix"
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

	// create initial estimate to work from recursively
	est, err := estimate.NewBaseWithCov(u[len(p)-1].Val(), u[len(p)-1].Cov())
	if err != nil {
		return nil, err
	}

	// smoothed state
	x := &mat.Dense{}
	pk := &mat.Dense{}

	for i := len(p) - 1; i > 0; i-- {
		fmt.Println(i)
		// intermediate smoothing matrix
		c := &mat.Dense{}
		// Pk*Fk'
		c.Mul(u[i-1].Cov(), m[i-1].T())
		// P_(k+1)^-1 inverse
		pinv := &mat.Dense{}
		// invert predicted P_k+1 covariance
		if err := pinv.Inverse(p[i].Cov()); err != nil {
			return nil, err
		}
		// Pk*Fk'* P_(k+1)^-1
		c.Mul(c, pinv)

		fmt.Println("U Estimate:\n", matrix.Format(u[i].Val()))
		fmt.Println("P Estimate:\n", matrix.Format(p[i].Val()))

		// smooth the state
		x.Sub(est.Val(), p[i].Val())
		fmt.Println("S Estimate:\n", matrix.Format(x))

		// c*x
		x.Mul(c, x)
		// xk + Ck*x_sub
		x.Add(u[i-i].Val(), x)

		// smoothed covariance
		cov := &mat.Dense{}
		// smooth covariance
		cov.Sub(est.Cov(), p[i].Cov())
		// Ck*P_sub
		pk.Mul(c, cov)
		// Ck*P_sub*Ck'
		pk.Mul(pk, c.T())
		// Pk + Ck*P_sub*Ck'
		pk.Add(u[i-1].Cov(), pk)

		r, _ := cov.Dims()
		pSmooth := mat.NewSymDense(r, nil)
		// update KF covariance matrix
		for i := 0; i < r; i++ {
			for j := i; j < r; j++ {
				pSmooth.SetSym(i, j, pk.At(i, j))
			}
		}

		est, err = estimate.NewBaseWithCov(x.ColView(0), pSmooth)
		if err != nil {
			return nil, err
		}
		sx[i-1] = est
		fmt.Println("Estimate:\n", matrix.Format(est.Val()))
	}

	return sx, nil
}
