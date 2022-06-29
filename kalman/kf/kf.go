package kf

import (
	"fmt"

	filter "github.com/milosgajdos/go-estimate"
	"github.com/milosgajdos/go-estimate/estimate"
	"github.com/milosgajdos/go-estimate/noise"
	"gonum.org/v1/gonum/mat"
)

// KF is Kalman Filter
type KF struct {
	// m is KF system model
	m filter.DiscreteControlSystem
	// q is state noise a.k.a. process noise
	q filter.Noise
	// r is output noise a.k.a. measurement noise
	r filter.Noise
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
//  - m:      dynamical system model
//  - init:   initial condition of the filter
//  - z:      disturbance input a.k.a. process noise corresponding to E disturbance matrix
//  - wn:     output noise a.k.a. measurement noise
//  - c:      KF configuration (contains propagation and observation matrices)
// It returns error if either of the following conditions is met:
//  - invalid model is given: model dimensions must be positive integers
//  - invalid state or output noise is given: noise covariance must either be nil or match the model dimensions
func New(m filter.DiscreteControlSystem, init filter.InitCond, z, wn filter.Noise) (*KF, error) {
	// size of the input and output vectors
	nx, _, ny, _ := m.SystemDims()
	if nx <= 0 || ny <= 0 {
		return nil, fmt.Errorf("invalid model dimensions: [%d x %d]", nx, ny)
	}

	if z != nil {
		if z.Cov().Symmetric() != nx {
			return nil, fmt.Errorf("invalid state noise dimension: %d != %d", z.Cov().Symmetric(), ny)
		}
	} else {
		z, _ = noise.NewNone()
	}

	if wn != nil {
		if wn.Cov().Symmetric() != ny {
			return nil, fmt.Errorf("invalid output noise dimension: %d", wn.Cov().Symmetric())
		}
	} else {
		wn, _ = noise.NewNone()
	}

	rows, cols := m.SystemMatrix().Dims()
	if rows != nx || cols != nx {
		return nil, fmt.Errorf("invalid propagation matrix dimensions: [%d x %d]", rows, cols)
	}

	if m.ControlMatrix() != nil && !m.ControlMatrix().(*mat.Dense).IsEmpty() {
		rows, cols := m.ControlMatrix().Dims()
		if rows != nx {
			return nil, fmt.Errorf("invalid ctl propagation matrix dimensions: [%d x %d]", rows, cols)
		}
	}

	rows, cols = m.OutputMatrix().Dims()
	if rows != ny || cols != nx {
		return nil, fmt.Errorf("invalid observation matrix dimensions: [%d x %d]", rows, cols)
	}

	if m.FeedForwardMatrix() != nil && !m.FeedForwardMatrix().(*mat.Dense).IsEmpty() {
		rows, cols = m.FeedForwardMatrix().Dims()
		if rows != ny {
			return nil, fmt.Errorf("invalid ctl observation matrix dimensions: [%d x %d]", rows, cols)
		}
	}

	// initialize covariance matrix to initial condition covariance
	p := mat.NewSymDense(init.Cov().Symmetric(), nil)
	p.CopySym(init.Cov())

	// predicted state covariance
	pNext := mat.NewSymDense(init.Cov().Symmetric(), nil)

	// innovation vector
	inn := mat.NewVecDense(ny, nil)

	// kalman gain
	k := mat.NewDense(nx, ny, nil)

	return &KF{
		m:     m,
		q:     z,
		r:     wn,
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
		return nil, fmt.Errorf("system state propagation failed: %v", err)
	}

	cov := &mat.Dense{}
	cov.Mul(k.m.SystemMatrix(), k.p)
	cov.Mul(cov, k.m.SystemMatrix().T())

	if _, ok := k.q.(*noise.None); !ok {
		cov.Add(cov, k.q.Cov())
	}

	// update KF predicted covariance matrix
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
func (k *KF) Update(x, u, ym mat.Vector) (filter.Estimate, error) {
	nx, _, ny, _ := k.m.SystemDims()

	if ym.Len() != ny {
		return nil, fmt.Errorf("invalid measurement supplied: %v", ym)
	}

	// observe system output in the next step
	yNext, err := k.m.Observe(x, u, k.r.Sample())
	if err != nil {
		return nil, fmt.Errorf("failed to observe system output: %v", err)
	}

	pxy := mat.NewDense(nx, ny, nil)
	pyy := mat.NewDense(ny, ny, nil)

	// P*H'
	pxy.Mul(k.pNext, k.m.OutputMatrix().T())

	// Note: pxy = P * H' so we reuse the result here
	// H*P*H'
	pyy.Mul(k.m.OutputMatrix(), pxy)
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
	inn.SubVec(ym, yNext)

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
	a.Mul(gain, k.m.OutputMatrix())
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

	// update KF innovation vector
	k.inn.CopyVec(inn)
	k.k.Copy(gain)
	// update KF covariance matrix
	for i := 0; i < nx; i++ {
		for j := i; j < nx; j++ {
			k.p.SetSym(i, j, pCorr.At(i, j))
		}
	}
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

// Model returns KF models
func (k *KF) Model() filter.DiscreteModel {
	return k.m
}

// StateNoise retruns state noise
func (k *KF) StateNoise() filter.Noise {
	return k.q
}

// OutputNoise retruns output noise
func (k *KF) OutputNoise() filter.Noise {
	return k.r
}

// Cov returns KF covariance
func (k *KF) Cov() mat.Symmetric {
	cov := mat.NewSymDense(k.p.Symmetric(), nil)
	cov.CopySym(k.p)

	return cov
}

// SetCov sets KF covariance matrix to cov.
// It returns error if either cov is nil or its dimensions are not the same as KF covariance dimensions.
func (k *KF) SetCov(cov mat.Symmetric) error {
	if cov == nil {
		return fmt.Errorf("invalid covariance matrix: %v", cov)
	}

	if cov.Symmetric() != k.p.Symmetric() {
		return fmt.Errorf("invalid covariance matrix dims: [%d x %d]", cov.Symmetric(), cov.Symmetric())
	}

	k.p.CopySym(cov)

	return nil
}

// Gain returns Kalman gain
func (k *KF) Gain() mat.Matrix {
	gain := &mat.Dense{}
	gain.CloneFrom(k.k)

	return gain
}
