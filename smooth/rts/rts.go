package rts

import (
	"fmt"

	filter "github.com/milosgajdos/go-estimate"
	"github.com/milosgajdos/go-estimate/estimate"
	"github.com/milosgajdos/go-estimate/noise"
	"gonum.org/v1/gonum/mat"
)

// RTS is Rauch-Tung-Striebel smoother
type RTS struct {
	// q is state noise a.k.a. process noise
	q filter.Noise
	// m is system model
	m filter.DiscreteControlSystem
	// start is initial condition
	start filter.InitCond
}

// New creates new RTS and returns it.
// It returns error if it fails to create RTS smoother.
func New(m filter.DiscreteControlSystem, init filter.InitCond, q filter.Noise) (*RTS, error) {
	in, _, out, _ := m.SystemDims()
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("Invalid model dimensions: [%d x %d]", in, out)
	}

	if q != nil {
		if q.Cov().Symmetric() != in {
			return nil, fmt.Errorf("Invalid state noise dimension: %d", q.Cov().Symmetric())
		}
	} else {
		q, _ = noise.NewNone()
	}

	return &RTS{
		q:     q,
		m:     m,
		start: init,
	}, nil
}

// Smooth implements Rauch-Tung-Striebel smoothing algorithm.
// It uses estimates est to compute smoothed estimates and returns them.
// It returns error if either est is nil or smoothing could not be computed.
func (s *RTS) Smooth(est []filter.Estimate, u []mat.Vector) ([]filter.Estimate, error) {
	if est == nil {
		return nil, fmt.Errorf("Invalid estimates size")
	}

	if u != nil && len(u) != len(est) {
		return nil, fmt.Errorf("Invalid input vector size")
	}

	sx := make([]filter.Estimate, len(est))

	// create initial estimate to work from recursively
	e, err := estimate.NewBaseWithCov(s.start.State(), s.start.Cov())
	if err != nil {
		return nil, err
	}

	// smoothed state
	x := &mat.Dense{}
	pk := &mat.Dense{}

	var uEst mat.Vector = nil
	for i := len(est) - 1; i >= 0; i-- {
		// propagate input state to the next step
		if u != nil {
			uEst = u[i]
		}
		xk1, err := s.m.Propagate(est[i].Val(), uEst, s.q.Sample())
		if err != nil {
			return nil, fmt.Errorf("Model state propagation failed: %v", err)
		}

		// propagate covariance matrix to the next step
		pk1 := &mat.Dense{}
		pk1.Mul(s.m.SystemMatrix(), est[i].Cov())
		pk1.Mul(pk1, s.m.SystemMatrix().T())

		if _, ok := s.q.(*noise.None); !ok {
			pk1.Add(pk1, s.q.Cov())
		}

		// calculat smoothing matrix
		c := &mat.Dense{}
		// Pk*Ak'
		c.Mul(est[i].Cov(), s.m.SystemMatrix().T())
		// P_(k+1)^-1 inverse
		pinv := &mat.Dense{}
		// invert predicted P_k+1 covariance
		if err := pinv.Inverse(pk1); err != nil {
			return nil, err
		}
		// Pk*Fk'* P_(k+1)^-1
		c.Mul(c, pinv)

		// smooth the state
		x.Sub(e.Val(), xk1)
		// c*x
		x.Mul(c, x)
		// xk + Ck*x_sub
		x.Add(est[i].Val(), x)

		// smoothed covariance
		cov := &mat.Dense{}
		// smooth covariance
		cov.Sub(e.Cov(), pk1)
		// Ck*P_sub
		pk.Mul(c, cov)
		// Ck*P_sub*Ck'
		pk.Mul(pk, c.T())
		// Pk + Ck*P_sub*Ck'
		pk.Add(est[i].Cov(), pk)

		r, _ := cov.Dims()
		pSmooth := mat.NewSymDense(r, nil)
		// update KF covariance matrix
		for i := 0; i < r; i++ {
			for j := i; j < r; j++ {
				pSmooth.SetSym(i, j, pk.At(i, j))
			}
		}

		e, err = estimate.NewBaseWithCov(x.ColView(0), pSmooth)
		if err != nil {
			return nil, err
		}
		sx[i] = e
	}

	return sx, nil
}
