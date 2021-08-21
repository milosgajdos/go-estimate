package ekf

import (
	"fmt"

	filter "github.com/milosgajdos/go-estimate"
	"github.com/milosgajdos/go-estimate/estimate"
	"github.com/milosgajdos/go-estimate/noise"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
)

// IEKF is Iterated Extended Kalman Filter
type IEKF struct {
	// ekf.EKF is extended Kalman filter
	*EKF
	// n is number of update iterations
	n int
}

// NewIter creates new Iterated EKF and returns it.
// It accepts the following parameters:
// - m:  dynamical system model
// - ic: initial condition of the filter
// - q:  state a.k.a. process noise
// - r:  output a.k.a. measurement noise
// - n:  number of update iterations
// It returns error if either of the following conditions is met:
// - invalid model is given: model dimensions must be positive integers
// - invalid state or output noise is given: noise covariance must either be nil or match the model dimensions
// - invalid number of update iterations is given: n must be non-negative
func NewIter(m filter.DiscreteModel, ic filter.InitCond, q, r filter.Noise, n int) (*IEKF, error) {
	if n <= 0 {
		return nil, fmt.Errorf("invalid number of update iterations: %d", n)
	}

	// IEKF is EKF which uses iterating updates
	f, err := New(m, ic, q, r)
	if err != nil {
		return nil, err
	}

	return &IEKF{
		EKF: f,
		n:   n,
	}, nil
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate of x.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (k *IEKF) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	nx, _, ny, _ := k.m.SystemDims()

	if z.Len() != ny {
		return nil, fmt.Errorf("invalid measurement supplied: %v", z)
	}

	// observe system output in the next step
	y, err := k.m.Observe(x, u, k.OutputNoise().Sample())
	if err != nil {
		return nil, fmt.Errorf("failed to observe system output: %v", err)
	}

	pxy := mat.NewDense(nx, ny, nil)
	pyy := mat.NewDense(ny, ny, nil)

	// innovation vector
	inn := &mat.VecDense{}
	inn.SubVec(z, y)

	// kalman gain
	gain := &mat.Dense{}

	// corrected covariance
	corr := &mat.Dense{}

	// iterate k.n number of iterations and keep updating x
	for i := 0; i < k.n; i++ {
		// calculate Jacobian matrix
		fd.Jacobian(k.h, k.HJacFn(u), mat.Col(nil, 0, x), &fd.JacobianSettings{
			Formula:    fd.Central,
			Concurrent: true,
		})

		// P*H'
		pxy.Mul(k.pNext, k.h.T())

		// Note: pxy = P * H' so we reuse the result here
		// H*P*H'
		pyy.Mul(k.h, pxy)
		// if there is any measurement noise
		if _, ok := k.r.(*noise.None); !ok {
			pyy.Add(pyy, k.r.Cov())
		}

		// calculate Kalman gain
		pyyInv := &mat.Dense{}
		if err := pyyInv.Inverse(pyy); err != nil {
			return nil, fmt.Errorf("failed to calculat Pyy inverse: %v", err)
		}
		gain.Mul(pxy, pyyInv)

		// update state x
		corr.Mul(gain, inn)
		x.(*mat.VecDense).AddVec(x, corr.ColView(0))
	}

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

	return estimate.NewBaseWithCov(x, k.Cov())
}

// Run runs one step of IEKF for given state x, input u and measurement z.
// It corrects system state x using measurement z and returns new system estimate.
// It returns error if it either fails to propagate or correct state x.
func (k *IEKF) Run(x, u, z mat.Vector) (filter.Estimate, error) {
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
