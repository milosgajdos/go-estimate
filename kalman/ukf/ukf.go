package ukf

import (
	"fmt"
	"math"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"gonum.org/v1/gonum/mat"
)

// Config contains UKF [unitless] configuration parameters
type Config struct {
	// Alpha is alpha parameter (0,1]
	Alpha float64
	// Beta is beta parameter (2 is optimal choice for Gaussian)
	Beta float64
	// Kappa is kappa parameter (must be non-negative)
	Kappa float64
}

// UKF is Unscented (aka Sigma Point) Kalman Filter
type UKF struct {
	// model us UKF model
	model filter.Model
	// gamma is a unitless UKF parameter
	gamma float64
	// Wm0 is mean sigma point weight
	Wm0 float64
	// Wc0 is mean sigma point covariance weight
	Wc0 float64
	// Wsp is weight for regular sigma points and covariances
	W float64
	// xmpred is predicted mean sigma point
	xmpred *mat.VecDense
	// ympred is predicted mean sigma point output
	ympred *mat.VecDense
	// xpred is a matrix which stores regular sigma points predicted states
	xpred *mat.Dense
	// x is a matrix which stores regular sigma points states
	x *mat.Dense
	// ypred is a matrix which stores regular sigma points predicted outputs
	ypred *mat.Dense
	// y is a matrix which stores regular sigma points outputs
	y *mat.Dense
	// ps is the covariance matrix of sigma points
	ps *mat.SymDense
	// ppred is predicted UKF covariance matrix
	ppred *mat.SymDense
	// p is the UKF covariance matrix
	p *mat.SymDense
	// inn is innovation vector
	inn *mat.VecDense
	// k is Kalman gain
	k *mat.Dense
}

// New creates new UKF and returns it.
// It accepts following arguments:
// - model:  dynamical system model
// - init:   initial condition of the filter
// - c:      filter configuration
// It returns error if non-positive number of sigma points is given or if the sigma points fail to be generated.
func New(model filter.Model, init filter.InitCond, c *Config) (*UKF, error) {
	// size of the input and output vectors
	in, out := model.Dims()
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("Invalid model dimensions: [%d x %d]", in, out)
	}

	// config parameters can't be negative numbers
	if c.Alpha < 0 || c.Beta < 0 || c.Kappa < 0 {
		return nil, fmt.Errorf("Invalid config supplied: %v", c)
	}

	// sigmaDim stores dimension of sigma point
	sigmaDim := init.State().Len()

	rows, _ := model.StateNoise().Cov().Dims()
	sigmaDim += rows

	rows, _ = model.OutputNoise().Cov().Dims()
	sigmaDim += rows

	// lambda is another unitless UKF parameter - calculates using the config ones
	lambda := c.Alpha*c.Alpha*(float64(sigmaDim)+c.Kappa) - float64(sigmaDim)

	// gamma is the square root Sigma Point covariance scaling factor
	gamma := math.Sqrt(float64(sigmaDim) + lambda)

	// weight of the mean sigma point
	Wm0 := lambda / (float64(sigmaDim) + lambda)
	// weight of the mean sigma point covariance
	Wc0 := Wm0 + (1 - c.Alpha*c.Alpha + c.Beta)
	// weight of the rest of sigma points and covariance
	W := 1 / (2 * (float64(sigmaDim) + lambda))

	// expected predicted state and output aka. mean sigma points
	xmpred := mat.NewVecDense(in, nil)
	ympred := mat.NewVecDense(out, nil)

	// sigma points prediction matrix; stores only system sigma points vectors
	xpred := mat.NewDense(in, 2*sigmaDim+1, nil)
	// sigma points matrix: stores sigma points in its columns
	x := mat.NewDense(sigmaDim, 2*sigmaDim+1, nil)

	// sigma point outputs; stores inly system sigma points outputs
	ypred := mat.NewDense(out, 2*sigmaDim+1, nil)
	// sigma point outputs
	y := mat.NewDense(2*sigmaDim+1, out, nil)

	// sigma points covariance matrix; this is a block diagonal matrix
	ps := mat.NewSymDense(sigmaDim, nil)
	// copy the covariance matrices into sigma point covariance
	r, _ := init.Cov().Dims()
	s, _ := model.StateNoise().Cov().Dims()
	o, _ := model.OutputNoise().Cov().Dims()

	ps.SliceSquare(0, r).(*mat.SymDense).CopySym(init.Cov())
	ps.SliceSquare(r, r+s).(*mat.SymDense).CopySym(model.StateNoise().Cov())
	ps.SliceSquare(r+s, r+s+o).(*mat.SymDense).CopySym(model.OutputNoise().Cov())

	// predicted covariance; this covariance will be corrected in Update
	ppred := mat.NewSymDense(init.Cov().Symmetric(), nil)
	ppred.CopySym(init.Cov())
	// corrected matrix starts the same as p_
	p := mat.NewSymDense(init.Cov().Symmetric(), nil)
	p.CopySym(init.Cov())

	// innovation vector
	inn := mat.NewVecDense(out, nil)

	// kalman gain
	k := mat.NewDense(in, out, nil)

	return &UKF{
		model:  model,
		gamma:  gamma,
		Wm0:    Wm0,
		Wc0:    Wc0,
		W:      W,
		xmpred: xmpred,
		ympred: ympred,
		xpred:  xpred,
		x:      x,
		ypred:  ypred,
		y:      y,
		ps:     ps,
		ppred:  ppred,
		p:      p,
		inn:    inn,
		k:      k,
	}, nil
}

// genSigmaPoints generates new UKF sigma points around state x
func (k *UKF) genSigmaPoints(x mat.Vector) error {
	rows, cols := k.x.Dims()
	// first column contains mean sigma point
	k.x.Slice(0, x.Len(), 0, 1).(*mat.Dense).Copy(x)
	for i := x.Len(); i < rows; i++ {
		k.x.Set(i, 0, 0.0)
	}

	// sigma point covariance computation
	// k.p holds current UKF covariance
	r, _ := k.p.Dims()
	s, _ := k.model.StateNoise().Cov().Dims()
	o, _ := k.model.OutputNoise().Cov().Dims()

	k.ps.SliceSquare(0, r).(*mat.SymDense).CopySym(k.p)
	k.ps.SliceSquare(r, r+s).(*mat.SymDense).CopySym(k.model.StateNoise().Cov())
	k.ps.SliceSquare(r+s, r+s+o).(*mat.SymDense).CopySym(k.model.OutputNoise().Cov())

	var svd mat.SVD
	ok := svd.Factorize(k.ps, mat.SVDFull)
	if !ok {
		return fmt.Errorf("SVD factorization failed")
	}
	SqrtCov := new(mat.Dense)
	svd.UTo(SqrtCov)
	vals := svd.Values(nil)
	for i := range vals {
		vals[i] = math.Sqrt(vals[i])
	}
	diag := mat.NewDiagonal(len(vals), vals)
	SqrtCov.Mul(SqrtCov, diag)
	SqrtCov.Scale(k.gamma, SqrtCov)

	r, c := SqrtCov.Dims()
	// copy SqrtCov values into sigma points covariance matrix
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			k.ps.SetSym(i, j, SqrtCov.At(i, j))
		}
	}

	// we center the sigmapoints around the mean sigma point stored in 1st column
	for j := 1; j < cols; j++ {
		k.x.Slice(0, rows, j, j+1).(*mat.Dense).Copy(k.x.ColView(0))
	}
	// positive sigma points
	sigmas := k.x.Slice(0, r, 1, 1+(c/2)).(*mat.Dense)
	sigmas.Add(sigmas, k.ps)
	// negative sigma points
	sigmas = k.x.Slice(0, r, 1+(c/2), c+1).(*mat.Dense)
	sigmas.Sub(sigmas, k.ps)

	return nil
}

// Predict predicts the next output of the system given the state x and input u and returns it.
// It returns error if it either fails to generate or propagate sigma points (and x) to the next state.
func (k *UKF) Predict(x, u mat.Vector) (filter.Estimate, error) {
	// generate new sigma points around the new state
	if err := k.genSigmaPoints(x); err != nil {
		return nil, fmt.Errorf("Failed to generate sigma points: %v", err)
	}

	// propagate x to the next step
	xNext, err := k.model.Propagate(x, u)
	if err != nil {
		return nil, fmt.Errorf("Failed to propagate system state: %v", err)
	}

	// observe system output in the next step
	yNext, err := k.model.Observe(xNext, u)
	if err != nil {
		return nil, fmt.Errorf("System state observation failed: %v", err)
	}

	_, cols := k.x.Dims()
	xrows, _ := k.xpred.Dims()
	yrows, _ := k.ypred.Dims()

	// sigmaPred and sigmaOutPred store predicted sigma point states and outputs
	sigmaPred := mat.NewDense(xrows, cols, nil)
	sigmaOutPred := mat.NewDense(yrows, cols, nil)

	xmPred := mat.NewVecDense(xrows, nil)
	ymPred := mat.NewVecDense(yrows, nil)

	// propagate all sigma points and observe their output
	for c := 0; c < cols; c++ {
		sigmaNext, err := k.model.Propagate(k.x.ColView(c).(*mat.VecDense).SliceVec(0, x.Len()), u)
		if err != nil {
			return nil, fmt.Errorf("Failed to propagage sigma point: %v", err)
		}
		sigmaPred.Slice(0, sigmaNext.Len(), c, c+1).(*mat.Dense).Copy(sigmaNext)

		sigmaOutNext, err := k.model.Observe(sigmaNext, u)
		if err != nil {
			return nil, fmt.Errorf("Failed to observe sigma point output: %v", err)
		}
		sigmaOutPred.Slice(0, sigmaOutNext.Len(), c, c+1).(*mat.Dense).Copy(sigmaOutNext)

		if c == 0 {
			xmPred.AddScaledVec(xmPred, k.Wm0, sigmaNext)
			ymPred.AddScaledVec(ymPred, k.Wm0, sigmaNext)
		} else {
			xmPred.AddScaledVec(xmPred, k.W, sigmaNext)
			ymPred.AddScaledVec(ymPred, k.W, sigmaOutNext)
		}
	}

	// coy all predicted values to UKF
	k.xmpred.CopyVec(xmPred)
	k.ympred.CopyVec(ymPred)
	k.xpred.Copy(sigmaPred)
	k.ypred.Copy(sigmaOutPred)

	// predict covariance
	ppred := mat.NewSymDense(xrows, nil)
	cov := mat.NewDense(xrows, xrows, nil)
	sigmaVec := mat.NewVecDense(xrows, nil)

	for c := 0; c < cols; c++ {
		sigmaVec = sigmaPred.ColView(c).(*mat.VecDense)
		sigmaVec.SubVec(sigmaVec, xmPred)
		cov.Mul(sigmaVec, sigmaVec.T())

		if c == 0 {
			cov.Scale(k.Wc0, cov)
		} else {
			cov.Scale(k.W, cov)
		}

		for i := 0; i < xrows; i++ {
			for j := i; j < xrows; j++ {
				ppred.SetSym(i, j, ppred.At(i, j)+cov.At(i, j))
			}
		}
	}

	k.ppred.CopySym(ppred)

	return estimate.NewBaseWithCov(xNext, yNext, ppred)
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (k *UKF) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	return nil, nil
}

// Run runs one step of UKF for given state x, input u and measurement z.
// It corrects system state x using measurement z and returns new system estimate.
// It returns error if it either fails to propagate or correct state x or UKF sigma points.
func (k *UKF) Run(x, u, z mat.Vector) (filter.Estimate, error) {
	return nil, nil
}

// Covariance returns current UKF covariance
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
