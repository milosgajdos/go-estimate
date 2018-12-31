package kf

import (
	"fmt"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/go-filter/noise"
	"gonum.org/v1/gonum/mat"
)

// KF is Kalman Filter
type KF struct {
	// m is KF system model
	m filter.Model
	// q is state noise a.k.a. process noise
	q filter.Noise
	// r is output noise a.k.a. measurement noise
	r filter.Noise
	// f is KF propagation Jacobian
	f *mat.Dense
	// h is KF observation Jacobian
	h *mat.Dense
	// p is the UKF covariance matrix
	p *mat.SymDense
	// pNext is the UKF predicted covariance matrix
	pNext *mat.SymDense
	// inn is innovation vector
	inn *mat.VecDense
	// k is Kalman gain
	k *mat.Dense
}

// New creates new KF and returns it.
// It accepts the following parameters:
// - m:      dynamical system model
// - init:   initial condition of the filter
// - q:      state a.k.a. process noise
// - r:      output a.k.a. measurement noise
// It returns error if either of the following conditions is met:
// - invalid model is given: model dimensions must be positive integers
// - invalid state or output noise is given: noise covariance must either be nil or match the model dimensions
func New(m filter.Model, init filter.InitCond, q, r filter.Noise) (*KF, error) {
	// size of the input and output vectors
	in, out := m.Dims()
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

	if r != nil {
		if r.Cov().Symmetric() != out {
			return nil, fmt.Errorf("Invalid output noise dimension: %d", r.Cov().Symmetric())
		}
	} else {
		r, _ = noise.NewNone()
	}

	// propagation Jacobian
	f := mat.NewDense(in, in, nil)

	// observation Jacobian
	h := mat.NewDense(out, in, nil)

	// initialize covariance matrix to initial condition covariance
	p := mat.NewSymDense(init.Cov().Symmetric(), nil)
	p.CopySym(init.Cov())

	// predicted state covariance
	pNext := mat.NewSymDense(init.Cov().Symmetric(), nil)

	// innovation vector
	inn := mat.NewVecDense(out, nil)

	// kalman gain
	k := mat.NewDense(in, out, nil)

	return &KF{
		m:     m,
		q:     q,
		r:     r,
		f:     f,
		h:     h,
		p:     p,
		pNext: pNext,
		inn:   inn,
		k:     k,
	}, nil
}

// Predict calculates the next system state given the state x and input u and returns its estimate.
// It first generates new sigma points around x and then attempts to propagate them to the next step.
// It returns error if it either fails to generate or propagate the sigma points (and x) to the next step.
func (k *KF) Predict(x, u mat.Vector) (filter.Estimate, error) {
	// propagate input state to the next step
	xNext, err := k.m.Propagate(x, u, k.q.Sample())
	if err != nil {
		return nil, fmt.Errorf("System state propagation failed: %v", err)
	}

	// TODO: implement

	return estimate.NewBaseWithCov(xNext, k.pNext)
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (k *KF) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	_, out := k.m.Dims()

	if z.Len() != out {
		return nil, fmt.Errorf("Invalid measurement supplied: %v", z)
	}

	// TODO: implement

	return estimate.NewBaseWithCov(x, k.p)
}

// Run runs one step of KF for given state x, input u and measurement z.
// It corrects system state x using measurement z and returns new system estimate.
// It returns error if it either fails to propagate or correct state x.
func (k *KF) Run(x, u, z mat.Vector) (filter.Estimate, error) {
	pred, err := k.Predict(x, u)
	if err != nil {
		return nil, err
	}

	est, err := k.Update(pred.Val(), u, z)
	if err != nil {
		return nil, err
	}

	return est, nil
}

// Covariance returns KF covariance
func (k *KF) Covariance() mat.Symmetric {
	cov := mat.NewSymDense(k.p.Symmetric(), nil)
	cov.CopySym(k.p)

	return cov
}

// Gain returns Kalman gain
func (k *KF) Gain() mat.Matrix {
	gain := &mat.Dense{}
	gain.Clone(k.k)

	return gain
}
