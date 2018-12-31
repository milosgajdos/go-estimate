package ukf

import (
	"fmt"
	"math"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/go-filter/matrix"
	"github.com/milosgajdos83/go-filter/noise"
	"gonum.org/v1/gonum/mat"
)

// SigmaPoints represents UKF sigma points and their covariance
type SigmaPoints struct {
	// X stores sigma points in its columns
	X *mat.Dense
	// Cov is sigma points covariance
	Cov *mat.SymDense
}

// sigmaPointsNext stores sigma points predicted values
type sigmaPointsNext struct {
	x     *mat.Dense
	xMean *mat.VecDense
}

// Config contains UKF [unitless] configuration parameters
type Config struct {
	// Alpha is alpha parameter (0,1]
	Alpha float64
	// Beta is beta parameter (2 is optimal choice for Gaussian)
	Beta float64
	// Kappa is kappa parameter (must be non-negative)
	Kappa float64
}

// UKF is Unscented (a.k.a. Sigma Point) Kalman Filter
type UKF struct {
	// m is UKF system model
	m filter.Model
	// q is state noise a.k.a. process noise
	q filter.Noise
	// r is output noise a.k.a. measurement noise
	r filter.Noise
	// gamma is a unitless UKF parameter
	gamma float64
	// Wm0 is mean sigma point weight
	Wm0 float64
	// Wc0 is mean sigma point covariance weight
	Wc0 float64
	// Wsp is weight for regular sigma points and covariances
	W float64
	// sp stores UKF sigma points
	sp *SigmaPoints
	// spNext stores sigma points predictions
	spNext *sigmaPointsNext
	// p is the UKF covariance matrix
	p *mat.SymDense
	// pNext is the UKF predicted covariance matrix
	pNext *mat.SymDense
	// inn is innovation vector
	inn *mat.VecDense
	// k is Kalman gain
	k *mat.Dense
}

// New creates new UKF and returns it.
// It accepts the following parameters:
// - m:      dynamical system model
// - init:   initial condition of the filter
// - q:      state a.k.a. process noise
// - r:      output a.k.a. measurement noise
// - c:      filter configuration
// It returns error if either of the following conditions is met:
// - invalid model is given: model dimensions must be positive integers
// - invalid state or output noise is given: noise covariance must either be nil or match the model dimensions
// - invalid sigma points parameters (alpha, beta, kappa) are supplied
// - sigma points fail to be generated: due to covariance SVD factorizations failure
func New(m filter.Model, init filter.InitCond, q, r filter.Noise, c *Config) (*UKF, error) {
	// size of the input and output vectors
	in, out := m.Dims()
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("Invalid model dimensions: [%d x %d]", in, out)
	}

	// sigma point dimension (length of sigma point vector)
	spDim := init.State().Len()

	if q != nil {
		if q.Cov().Symmetric() != in {
			return nil, fmt.Errorf("Invalid state noise dimension: %d", q.Cov().Symmetric())
		}
		spDim += q.Cov().Symmetric()
	} else {
		q, _ = noise.NewNone()
	}

	if r != nil {
		if r.Cov().Symmetric() != out {
			return nil, fmt.Errorf("Invalid output noise dimension: %d", r.Cov().Symmetric())
		}
		spDim += r.Cov().Symmetric()
	} else {
		r, _ = noise.NewNone()
	}

	// lambda is one of the unitless UKF parameters calculated using the config ones
	lambda := c.Alpha*c.Alpha*(float64(spDim)+c.Kappa) - float64(spDim)

	// gamma is the square root Sigma Point covariance scaling factor
	gamma := math.Sqrt(float64(spDim) + lambda)

	// weight of the mean sigma point
	Wm0 := lambda / (float64(spDim) + lambda)

	// weight of the mean sigma point covariance
	Wc0 := Wm0 + (1 - c.Alpha*c.Alpha + c.Beta)

	// weight of the rest of sigma points and covariance
	W := 1 / (2 * (float64(spDim) + lambda))

	// sigma points matrix: stores sigma points in its columns
	x := mat.NewDense(spDim, 2*spDim+1, nil)

	// sigma points covariance matrix: this is a block diagonal matrix
	cov := matrix.BlockSymDiag([]mat.Symmetric{init.Cov(), q.Cov(), r.Cov()})

	sp := &SigmaPoints{
		X:   x,
		Cov: cov,
	}

	// sigma points predicted states
	xPred := mat.NewDense(in, 2*spDim+1, nil)
	// expected predicted sigma point state a.k.a. mean sigma point predicted state
	xMean := mat.NewVecDense(in, nil)

	spNext := &sigmaPointsNext{
		x:     xPred,
		xMean: xMean,
	}

	// predicted covariance; this covariance is corrected using new measurement
	pNext := mat.NewSymDense(init.Cov().Symmetric(), nil)
	pNext.CopySym(init.Cov())

	// initialize covariance matrix to initial condition covariance
	p := mat.NewSymDense(init.Cov().Symmetric(), nil)
	p.CopySym(init.Cov())

	// innovation vector
	inn := mat.NewVecDense(out, nil)

	// kalman gain
	k := mat.NewDense(in, out, nil)

	return &UKF{
		m:      m,
		q:      q,
		r:      r,
		gamma:  gamma,
		Wm0:    Wm0,
		Wc0:    Wc0,
		W:      W,
		sp:     sp,
		spNext: spNext,
		p:      p,
		pNext:  pNext,
		inn:    inn,
		k:      k,
	}, nil
}

// GenSigmaPoints generates UKF sigma points around x and returns them.
// It returns error if it fails to generate new sigma points due to covariance SVD facrtorization failure.
func (k *UKF) GenSigmaPoints(x mat.Vector) (*SigmaPoints, error) {
	rows, cols := k.sp.X.Dims()
	sp := mat.NewDense(rows, cols, nil)
	cov := matrix.BlockSymDiag([]mat.Symmetric{k.p, k.q.Cov(), k.r.Cov()})

	var svd mat.SVD
	ok := svd.Factorize(cov, mat.SVDFull)
	if !ok {
		return nil, fmt.Errorf("SVD factorization failed")
	}
	SqrtCov := new(mat.Dense)
	svd.UTo(SqrtCov)
	vals := svd.Values(nil)
	for i := range vals {
		vals[i] = math.Sqrt(vals[i])
	}
	diagSqrt := mat.NewDiagonal(len(vals), vals)

	r, c := SqrtCov.Dims()
	// copy SqrtCov values into sigma points covariance matrix
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			//cov.SetSym(i, j, SqrtCov.At(i, j))
			cov.SetSym(i, j, diagSqrt.At(i, j))
		}
	}
	cov.ScaleSym(k.gamma, cov)

	// we center the sigma points around the mean sigma point
	for j := 0; j < cols; j++ {
		sp.Slice(0, rows, j, j+1).(*mat.Dense).Copy(x)
	}
	// positive sigma points
	sx := sp.Slice(0, rows, 1, 1+((cols-1)/2)).(*mat.Dense)
	sx.Add(sx, cov)
	// negative sigma points
	sx = sp.Slice(0, rows, 1+((cols-1)/2), cols).(*mat.Dense)
	sx.Sub(sx, cov)

	return &SigmaPoints{
		X:   sp,
		Cov: cov,
	}, nil
}

// propagateSigmaPoints propagates sigma points to the next step and observes their output.
// It calculates mean predicted sigma point state and returns it with predicted sigma points states.
// It returns error if it fails to propagate the sigma points or observe their outputs.
func (k *UKF) propagateSigmaPoints(sp *SigmaPoints, u mat.Vector) (*sigmaPointsNext, error) {
	in, _ := k.m.Dims()
	_, cols := sp.X.Dims()

	// x stores predicted sigma point states
	x := mat.NewDense(in, cols, nil)

	// xMean stores predicted mean sigma point state
	xMean := mat.NewVecDense(in, nil)

	var spNext mat.Vector
	var err error
	qLen := k.q.Cov().Symmetric()

	// propagate all sigma points and observe their output
	for c := 0; c < cols; c++ {
		if qLen == 0 {
			spNext, err = k.m.Propagate(sp.X.ColView(c).(*mat.VecDense).SliceVec(0, in), u, nil)
		} else {
			spNext, err = k.m.Propagate(sp.X.ColView(c).(*mat.VecDense).SliceVec(0, in), u,
				sp.X.ColView(c).(*mat.VecDense).SliceVec(in, in+qLen))
		}
		if err != nil {
			return nil, fmt.Errorf("Failed to propagate sigma point: %v", err)
		}
		x.Slice(0, spNext.Len(), c, c+1).(*mat.Dense).Copy(spNext)

		if c == 0 {
			xMean.AddScaledVec(xMean, k.Wm0, spNext)
		} else {
			xMean.AddScaledVec(xMean, k.W, spNext)
		}
	}

	return &sigmaPointsNext{
		x:     x,
		xMean: xMean,
	}, nil
}

// predictCovariance estimates new UKF covariance based on sigma points x and their mean xMean and returns it.
// It returns error if it fails to calculate new covariance from predicted sigma points.
func (k *UKF) predictCovariance(x *mat.Dense, xMean *mat.VecDense) (*mat.SymDense, error) {
	rows, cols := x.Dims()

	predCov := mat.NewSymDense(rows, nil)

	// cov is an accumulator matrix that stores added covariances
	cov := mat.NewDense(rows, rows, nil)

	// helper vector used when calculating predicted covariance
	sigmaPoint := mat.NewVecDense(rows, nil)

	for c := 0; c < cols; c++ {
		sigmaPoint.SubVec(x.ColView(c), xMean)
		cov.Mul(sigmaPoint, sigmaPoint.T())

		if c == 0 {
			cov.Scale(k.Wc0, cov)
		} else {
			cov.Scale(k.W, cov)
		}

		for i := 0; i < rows; i++ {
			for j := i; j < rows; j++ {
				predCov.SetSym(i, j, predCov.At(i, j)+cov.At(i, j))
			}
		}
	}

	return predCov, nil
}

// Predict calculates the next system state given the state x and input u and returns its estimate.
// It first generates new sigma points around x and then attempts to propagate them to the next step.
// It returns error if it either fails to generate or propagate the sigma points (and x) to the next step.
func (k *UKF) Predict(x, u mat.Vector) (filter.Estimate, error) {
	// generate new sigma points around x
	sigmaPoints, err := k.GenSigmaPoints(x)
	if err != nil {
		return nil, fmt.Errorf("Failed to generate sigma points: %v", err)
	}

	// propagate x to the next step
	xNext, err := k.m.Propagate(x, u, k.q.Sample())
	if err != nil {
		return nil, fmt.Errorf("Failed to propagate system state: %v", err)
	}

	sigmaPointsNext, err := k.propagateSigmaPoints(sigmaPoints, u)
	if err != nil {
		return nil, fmt.Errorf("Failed to propagate sigma points: %v", err)
	}

	cov, err := k.predictCovariance(sigmaPointsNext.x, sigmaPointsNext.xMean)
	if err != nil {
		return nil, fmt.Errorf("Failed to predict covariance: %v", err)
	}

	// it's now safe to update the internal state of the filter
	k.spNext.x.Copy(sigmaPointsNext.x)
	k.spNext.xMean.CopyVec(sigmaPointsNext.xMean)
	k.pNext.CopySym(cov)

	return estimate.NewBaseWithCov(xNext, cov)
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (k *UKF) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	in, out := k.m.Dims()
	_, cols := k.spNext.x.Dims()

	if z.Len() != out {
		return nil, fmt.Errorf("Invalid measurement supplied: %v", z)
	}

	// y stores predicted sigma point outputs
	y := mat.NewDense(out, cols, nil)

	// yMean stores predicted mean sigma point output
	yMean := mat.NewVecDense(out, nil)

	var spOut mat.Vector
	var err error
	rLen := k.r.Cov().Symmetric()

	// observe sigma points outputs
	for c := 0; c < cols; c++ {
		if rLen == 0 {
			spOut, err = k.m.Observe(k.spNext.x.ColView(c), u, nil)
		} else {
			spOut, err = k.m.Observe(k.spNext.x.ColView(c), u, k.r.Sample())
		}
		if err != nil {
			return nil, fmt.Errorf("Failed to observe sigma point output: %v", err)
		}
		y.Slice(0, spOut.Len(), c, c+1).(*mat.Dense).Copy(spOut)

		if c == 0 {
			yMean.AddScaledVec(yMean, k.Wm0, spOut)
		} else {
			yMean.AddScaledVec(yMean, k.W, spOut)
		}
	}

	// covariance of x and y; y is predicted sigma point output
	pxy := mat.NewDense(in, out, nil)

	// predicted sigma points output covariance
	pyy := mat.NewDense(out, out, nil)

	// helper vectors used in calculating covariances
	sigmaPoint := mat.NewVecDense(in, nil)
	sigmaPointOut := mat.NewVecDense(out, nil)

	// helper matrices which hold intermediary covariances
	covxy := mat.NewDense(in, out, nil)
	covyy := mat.NewDense(out, out, nil)

	for c := 0; c < cols; c++ {
		sigmaPoint.SubVec(k.spNext.x.ColView(c), k.spNext.xMean)
		sigmaPointOut.SubVec(y.ColView(c), yMean)

		covxy.Mul(sigmaPoint, sigmaPointOut.T())
		covyy.Mul(sigmaPointOut, sigmaPointOut.T())

		if c == 0 {
			covxy.Scale(k.Wc0, covxy)
			covyy.Scale(k.Wc0, covyy)
		} else {
			covxy.Scale(k.W, covxy)
			covyy.Scale(k.W, covyy)
		}

		pxy.Add(pxy, covxy)
		pyy.Add(pyy, covyy)
	}

	// calculate Kalman gain
	pyyInv := &mat.Dense{}
	pyyInv.Inverse(pyy)

	gain := &mat.Dense{}
	gain.Mul(pxy, pyyInv)

	// innovation vector
	inn := &mat.VecDense{}
	inn.SubVec(z, yMean)

	// update state x
	corr := &mat.Dense{}
	corr.Mul(gain, inn)
	x.(*mat.VecDense).AddVec(k.spNext.xMean, corr.ColView(0))

	// correct UKF covariance
	kr := &mat.Dense{}
	kr.Mul(pyy, gain.T())
	pCorr := &mat.Dense{}
	pCorr.Mul(gain, kr)
	pCorr.Sub(k.pNext, pCorr)

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

// Run runs one step of UKF for given state x, input u and measurement z.
// It corrects system state x using measurement z and returns new system estimate.
// It returns error if it either fails to propagate or correct state x or UKF sigma points.
func (k *UKF) Run(x, u, z mat.Vector) (filter.Estimate, error) {
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

// Covariance returns UKF covariance
func (k *UKF) Covariance() mat.Symmetric {
	cov := mat.NewSymDense(k.p.Symmetric(), nil)
	cov.CopySym(k.p)

	return cov
}

// Gain returns Kalman gain
func (k *UKF) Gain() mat.Matrix {
	gain := &mat.Dense{}
	gain.Clone(k.k)

	return gain
}
