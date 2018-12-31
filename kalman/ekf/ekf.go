package ekf

import (
	"fmt"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/go-filter/noise"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
)

// EKF is Extended Kalman Filter
type EKF struct {
	// m is EKF system model
	m filter.Model
	// q is state noise a.k.a. process noise
	q filter.Noise
	// r is output noise a.k.a. measurement noise
	r filter.Noise
	// fFunc is propagation Jacobian function
	fFunc func(u mat.Vector) func(y, x []float64)
	// f is EKF propagation Jacobian
	f *mat.Dense
	// hFunc is observation Jacobian function
	hFunc func(u mat.Vector) func(y, x []float64)
	// h is EKF observation Jacobian
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

// New creates new EKF and returns it.
// It accepts the following parameters:
// - m:      dynamical system model
// - init:   initial condition of the filter
// - q:      state a.k.a. process noise
// - r:      output a.k.a. measurement noise
// It returns error if either of the following conditions is met:
// - invalid model is given: model dimensions must be positive integers
// - invalid state or output noise is given: noise covariance must either be nil or match the model dimensions
func New(m filter.Model, init filter.InitCond, q, r filter.Noise) (*EKF, error) {
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
	fFunc := func(u mat.Vector) func([]float64, []float64) {
		q, _ := noise.NewZero(in)

		return func(xOut, xNow []float64) {
			x := mat.NewVecDense(len(xNow), xNow)
			xNext, err := m.Propagate(x, u, q.Sample())
			if err != nil {
				panic(err)
			}

			for i := 0; i < len(xOut); i++ {
				xOut[i] = xNext.At(i, 0)
			}
		}
	}
	f := mat.NewDense(in, in, nil)

	// observation Jacobian
	hFunc := func(u mat.Vector) func([]float64, []float64) {
		r, _ := noise.NewZero(out)

		return func(y, xNow []float64) {
			x := mat.NewVecDense(len(xNow), xNow)
			// observe system output in the next step
			yNext, err := m.Observe(x, u, r.Sample())
			if err != nil {
				panic(err)
			}

			for i := 0; i < len(y); i++ {
				y[i] = yNext.At(i, 0)
			}
		}
	}
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

	return &EKF{
		m:     m,
		q:     q,
		r:     r,
		fFunc: fFunc,
		f:     f,
		hFunc: hFunc,
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
func (k *EKF) Predict(x, u mat.Vector) (filter.Estimate, error) {
	// propagate input state to the next step
	xNext, err := k.m.Propagate(x, u, k.q.Sample())
	if err != nil {
		return nil, fmt.Errorf("System state propagation failed: %v", err)
	}

	// calculate Jacobian matrix
	fd.Jacobian(k.f, k.fFunc(u), mat.Col(nil, 0, x), &fd.JacobianSettings{
		Formula:    fd.Central,
		Concurrent: true,
	})

	cov := &mat.Dense{}
	cov.Mul(k.p, k.f.T())
	cov.Mul(cov, k.f)

	if _, ok := k.q.(*noise.None); !ok {
		cov.Add(cov, k.q.Cov())
	}

	// update EKF covariance matrix
	n := k.pNext.Symmetric()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			k.pNext.SetSym(i, j, cov.At(i, j))
		}
	}

	return estimate.NewBaseWithCov(xNext, k.pNext)
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (k *EKF) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	in, out := k.m.Dims()

	if z.Len() != out {
		return nil, fmt.Errorf("Invalid measurement supplied: %v", z)
	}

	// observe system output in the next step
	yNext, err := k.m.Observe(x, u, k.r.Sample())
	if err != nil {
		return nil, fmt.Errorf("Failed to observe system output: %v", err)
	}

	// innovation vector
	inn := &mat.VecDense{}
	inn.SubVec(z, yNext)

	// calculate Jacobian matrix
	fd.Jacobian(k.h, k.hFunc(u), mat.Col(nil, 0, x), &fd.JacobianSettings{
		Formula:    fd.Central,
		Concurrent: true,
	})

	pxy := mat.NewDense(in, out, nil)
	pyy := mat.NewDense(out, out, nil)

	// P*H'
	pxy.Mul(k.pNext, k.h.T())

	// Note: pxy = P * H' so we don't need to do the same Mul() again
	// H*P*H'
	pyy.Mul(k.h, pxy)
	// no measurement noise
	if _, ok := k.r.(*noise.None); !ok {
		pyy.Add(pyy, k.r.Cov())
	}

	// calculate Kalman gain
	pyyInv := &mat.Dense{}
	pyyInv.Inverse(pyy)

	gain := &mat.Dense{}
	gain.Mul(pxy, pyyInv)

	// update state x
	corr := &mat.Dense{}
	corr.Mul(gain, inn)
	x.(*mat.VecDense).AddVec(x, corr.ColView(0))

	// Joseph form update
	eye := mat.NewDiagonal(x.Len(), nil)
	for i := 0; i < x.Len(); i++ {
		eye.SetDiag(i, 1.0)
	}
	a := &mat.Dense{}
	// K*H
	a.Mul(gain, k.h)
	// eye - K*H
	a.Sub(eye, a)

	// K*R*K'
	pkrk := &mat.Dense{}
	if _, ok := k.r.(*noise.None); !ok {
		kr := &mat.Dense{}
		kr.Mul(k.r.Cov(), gain.T())
		pkrk.Mul(gain, kr)
	}

	pa := &mat.Dense{}
	pa.Mul(k.pNext, a.T())
	apa := &mat.Dense{}
	apa.Mul(a, pa)

	pCorr := &mat.Dense{}
	if !pkrk.IsZero() {
		pCorr.Add(apa, pkrk)
	}

	// update UKF innovation vector
	k.inn.CopyVec(inn)
	k.k.Copy(gain)
	// update UKF covariance matrix
	for i := 0; i < in; i++ {
		for j := i; j < in; j++ {
			k.p.SetSym(i, j, pCorr.At(i, j))
		}
	}

	return estimate.NewBaseWithCov(x, k.p)
}

// Run runs one step of EKF for given state x, input u and measurement z.
// It corrects system state x using measurement z and returns new system estimate.
// It returns error if it either fails to propagate or correct state x.
func (k *EKF) Run(x, u, z mat.Vector) (filter.Estimate, error) {
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

// Covariance returns EKF covariance
func (k *EKF) Covariance() mat.Symmetric {
	cov := mat.NewSymDense(k.p.Symmetric(), nil)
	cov.CopySym(k.p)

	return cov
}

// Gain returns Kalman gain
func (k *EKF) Gain() mat.Matrix {
	gain := &mat.Dense{}
	gain.Clone(k.k)

	return gain
}
