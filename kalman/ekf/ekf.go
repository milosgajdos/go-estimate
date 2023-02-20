package ekf

import (
	"fmt"

	filter "github.com/milosgajdos/go-estimate"
	"github.com/milosgajdos/go-estimate/estimate"
	"github.com/milosgajdos/go-estimate/noise"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
)

// JacFunc defines jacobian function to calculate Jacobian matrix
type JacFunc func(u mat.Vector) func(y, x []float64)

// EKF is Extended Kalman Filter
type EKF struct {
	// m is EKF system model
	m filter.Model
	// q is state noise a.k.a. process noise
	q filter.Noise
	// r is output noise a.k.a. measurement noise
	r filter.Noise
	// FJacFn is propagation Jacobian function
	FJacFn JacFunc
	// f is EKF propagation matrix
	f *mat.Dense
	// HJacFn is observation Jacobian function
	HJacFn JacFunc
	// h is EKF observation matrix
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
	nx, _, ny, _ := m.SystemDims()
	if nx <= 0 || ny <= 0 {
		return nil, fmt.Errorf("invalid model dimensions: [%d x %d]", nx, ny)
	}

	if q != nil {
		if q.Cov().SymmetricDim() != nx {
			return nil, fmt.Errorf("invalid state noise dimension: %d", q.Cov().SymmetricDim())
		}
	} else {
		q, _ = noise.NewNone()
	}

	if r != nil {
		if r.Cov().SymmetricDim() != ny {
			return nil, fmt.Errorf("invalid output noise dimension: %d", r.Cov().SymmetricDim())
		}
	} else {
		r, _ = noise.NewNone()
	}

	// propagation Jacobian
	fJacFn := func(u mat.Vector) func([]float64, []float64) {
		q, _ := noise.NewZero(nx)

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
	f := mat.NewDense(nx, nx, nil)

	// observation Jacobian
	hJacFn := func(u mat.Vector) func([]float64, []float64) {
		r, _ := noise.NewZero(ny)

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
	h := mat.NewDense(ny, nx, nil)

	// initialize covariance matrix to initial condition covariance
	p := mat.NewSymDense(init.Cov().SymmetricDim(), nil)
	p.CopySym(init.Cov())

	// predicted state covariance
	pNext := mat.NewSymDense(init.Cov().SymmetricDim(), nil)

	// innovation vector
	inn := mat.NewVecDense(ny, nil)

	// kalman gain
	k := mat.NewDense(nx, ny, nil)

	return &EKF{
		m:      m,
		q:      q,
		r:      r,
		FJacFn: fJacFn,
		f:      f,
		HJacFn: hJacFn,
		h:      h,
		p:      p,
		pNext:  pNext,
		inn:    inn,
		k:      k,
	}, nil
}

// Predict calculates the next system state given the state x and input u and returns its estimate.
// It first generates new sigma points around x and then attempts to propagate them to the next step.
// It returns error if it either fails to generate or propagate the sigma points (and x) to the next step.
func (k *EKF) Predict(x, u mat.Vector) (filter.Estimate, error) {
	// propagate input state to the next step
	xNext, err := k.m.Propagate(x, u, k.q.Sample())
	if err != nil {
		return nil, fmt.Errorf("system state propagation failed: %v", err)
	}

	// calculate propagation Jacobian matrix
	fd.Jacobian(k.f, k.FJacFn(u), mat.Col(nil, 0, x), &fd.JacobianSettings{
		Formula:    fd.Central,
		Concurrent: true,
	})

	cov := &mat.Dense{}
	cov.Mul(k.f, k.p)
	cov.Mul(cov, k.f.T())

	if _, ok := k.q.(*noise.None); !ok {
		cov.Add(cov, k.q.Cov())
	}

	// update EKF covariance matrix
	n := k.pNext.SymmetricDim()
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
	nx, _, ny, _ := k.m.SystemDims()

	if z.Len() != ny {
		return nil, fmt.Errorf("invalid measurement supplied: %v", z)
	}

	// observe system output in the next step
	y, err := k.m.Observe(x, u, k.r.Sample())
	if err != nil {
		return nil, fmt.Errorf("failed to observe system output: %v", err)
	}

	// calculate observation Jacobian matrix
	fd.Jacobian(k.h, k.HJacFn(u), mat.Col(nil, 0, x), &fd.JacobianSettings{
		Formula:    fd.Central,
		Concurrent: true,
	})

	pxy := mat.NewDense(nx, ny, nil)
	pyy := mat.NewDense(ny, ny, nil)

	// P*H'
	pxy.Mul(k.pNext, k.h.T())

	// Note: pxy = P * H' so we reuse the result here
	// H*P*H'
	pyy.Mul(k.h, pxy)
	// no measurement noise
	if _, ok := k.r.(*noise.None); !ok {
		pyy.Add(pyy, k.r.Cov())
	}

	// calculate Kalman gain
	pyyInv := &mat.Dense{}
	if err := pyyInv.Inverse(pyy); err != nil {
		return nil, fmt.Errorf("failed to calculat Pyy inverse: %v", err)
	}
	gain := &mat.Dense{}
	gain.Mul(pxy, pyyInv)

	// innovation vector
	inn := &mat.VecDense{}
	inn.SubVec(z, y)

	// update state x
	corr := &mat.Dense{}
	corr.Mul(gain, inn)
	x.(*mat.VecDense).AddVec(x, corr.ColView(0))

	// Joseph form update
	eye := mat.NewDiagDense(x.Len(), nil)
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
	// if there is some output noise
	if _, ok := k.r.(*noise.None); !ok {
		kr := &mat.Dense{}
		kr.Mul(gain, k.r.Cov())
		pkrk.Mul(kr, gain.T())
	}

	ap := &mat.Dense{}
	ap.Mul(a, k.pNext)
	apa := &mat.Dense{}
	apa.Mul(ap, a.T())

	pCorr := &mat.Dense{}
	if !pkrk.IsEmpty() {
		pCorr.Add(apa, pkrk)
	}

	// update EKF innovation vector
	k.inn.CopyVec(inn)
	k.k.Copy(gain)
	// update EKF covariance matrix
	for i := 0; i < nx; i++ {
		for j := i; j < nx; j++ {
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

// Model returns EKF model
func (k *EKF) Model() filter.Model {
	return k.m
}

// StateNoise retruns state noise
func (k *EKF) StateNoise() filter.Noise {
	return k.q
}

// OutputNoise retruns output noise
func (k *EKF) OutputNoise() filter.Noise {
	return k.r
}

// Cov returns EKF covariance
func (k *EKF) Cov() mat.Symmetric {
	cov := mat.NewSymDense(k.p.SymmetricDim(), nil)
	cov.CopySym(k.p)

	return cov
}

// SetCov sets EKF covariance matrix to cov.
// It returns error if either cov is nil or its dimensions are not the same as EKF covariance dimensions.
func (k *EKF) SetCov(cov mat.Symmetric) error {
	if cov == nil {
		return fmt.Errorf("invalid covariance matrix: %v", cov)
	}

	if cov.SymmetricDim() != k.p.SymmetricDim() {
		return fmt.Errorf("invalid covariance matrix dims: [%d x %d]", cov.SymmetricDim(), cov.SymmetricDim())
	}

	k.p.CopySym(cov)

	return nil
}

// Gain returns Kalman gain
func (k *EKF) Gain() mat.Matrix {
	gain := &mat.Dense{}
	gain.CloneFrom(k.k)

	return gain
}
