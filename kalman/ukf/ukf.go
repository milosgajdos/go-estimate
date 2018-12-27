package ukf

import (
	"fmt"
	"math"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"gonum.org/v1/gonum/mat"
)

// SigmaPoints stores sigma points and covariance
type SigmaPoints struct {
	// X stores sigma point vectors in columns
	X *mat.Dense
	// Cov contains sigma points covariance
	Cov *mat.SymDense
}

// sigmaPointsPred stores sigma points predicted values
type sigmaPointsPred struct {
	x     *mat.Dense
	xMean *mat.VecDense
	y     *mat.Dense
	yMean *mat.VecDense
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
	// sp stores UKF sigma points
	sp *SigmaPoints
	// spred stores sigma points predictions
	spred *sigmaPointsPred
	// p is the UKF covariance matrix
	p *mat.SymDense
	// ppred is the UKF predicted covariance matrix
	ppred *mat.SymDense
	// inn is innovation vector
	inn *mat.VecDense
	// k is Kalman gain
	k *mat.Dense
}

// New creates new UKF and returns it.
// It accepts the following arguments:
// - model:  dynamical system filter model
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

	// sigma point dimension (length of sigma point vector)
	spDim := init.State().Len()

	if !model.StateNoise().Cov().(*mat.SymDense).IsZero() {
		spDim += model.StateNoise().Cov().Symmetric()
	}

	if !model.OutputNoise().Cov().(*mat.SymDense).IsZero() {
		spDim += model.OutputNoise().Cov().Symmetric()
	}
	//fmt.Println("spDim", spDim)

	// lambda is another unitless UKF parameter - calculates using the config ones
	lambda := c.Alpha*c.Alpha*(float64(spDim)+c.Kappa) - float64(spDim)
	//fmt.Println("lambda:", lambda)

	// gamma is the square root Sigma Point covariance scaling factor
	gamma := math.Sqrt(float64(spDim) + lambda)
	//fmt.Println("gamma:", gamma)

	// weight of the mean sigma point
	Wm0 := lambda / (float64(spDim) + lambda)
	//fmt.Println("Wm0:", Wm0)
	// weight of the mean sigma point covariance
	Wc0 := Wm0 + (1 - c.Alpha*c.Alpha + c.Beta)
	//fmt.Println("Wc0:", Wc0)
	// weight of the rest of sigma points and covariance
	W := 1 / (2 * (float64(spDim) + lambda))
	//fmt.Println("W:", W)

	// sigma points matrix: stores sigma points in its columns
	x := mat.NewDense(spDim, 2*spDim+1, nil)

	// sigma points covariance matrix: this is a block diagonal matrix
	cov := mat.NewSymDense(spDim, nil)

	r := init.Cov().Symmetric()
	s := model.StateNoise().Cov().Symmetric()
	o := model.OutputNoise().Cov().Symmetric()

	// copy the covariance matrices into sigma point covariance
	cov.SliceSquare(0, r).(*mat.SymDense).CopySym(init.Cov())

	if !model.StateNoise().Cov().(*mat.SymDense).IsZero() {
		cov.SliceSquare(r, r+s).(*mat.SymDense).CopySym(model.StateNoise().Cov())
	}

	if !model.OutputNoise().Cov().(*mat.SymDense).IsZero() {
		cov.SliceSquare(r+s, r+s+o).(*mat.SymDense).CopySym(model.OutputNoise().Cov())
	}

	sp := &SigmaPoints{
		X:   x,
		Cov: cov,
	}

	// sigma points predicted states
	xPred := mat.NewDense(in, 2*spDim+1, nil)
	// expected predicted sigma point state a.k.a. mean sigma point predicted state
	xMean := mat.NewVecDense(in, nil)

	// sigma points predicted outputs
	yPred := mat.NewDense(out, 2*spDim+1, nil)
	// expected predicted sigma point output a.k.a. mean sigma point predicted output
	yMean := mat.NewVecDense(out, nil)

	spred := &sigmaPointsPred{
		x:     xPred,
		xMean: xMean,
		y:     yPred,
		yMean: yMean,
	}

	// predicted covariance; this covariance is corrected using new measurement
	ppred := mat.NewSymDense(init.Cov().Symmetric(), nil)
	ppred.CopySym(init.Cov())

	// initialize covariance matrix to initial condition covariance
	p := mat.NewSymDense(init.Cov().Symmetric(), nil)
	p.CopySym(init.Cov())

	// innovation vector
	inn := mat.NewVecDense(out, nil)

	// kalman gain
	k := mat.NewDense(in, out, nil)

	return &UKF{
		model: model,
		gamma: gamma,
		Wm0:   Wm0,
		Wc0:   Wc0,
		W:     W,
		sp:    sp,
		spred: spred,
		p:     p,
		ppred: ppred,
		inn:   inn,
		k:     k,
	}, nil
}

// GenSigmaPoints generates new UKF sigma points and their covariance and returns them.
// It returns error if it fails to generate new sigma points.
func (k *UKF) GenSigmaPoints(x mat.Vector) (*SigmaPoints, error) {
	rows, cols := k.sp.X.Dims()
	sp := mat.NewDense(rows, cols, nil)

	n := k.sp.Cov.Symmetric()
	spCov := mat.NewSymDense(n, nil)

	//fmt.Println("N:", n)

	//fmt.Println(matrix.Format(k.p))

	p := k.p.Symmetric()
	s := k.model.StateNoise().Cov().Symmetric()
	o := k.model.OutputNoise().Cov().Symmetric()

	//fmt.Println("p+s+o:", p+s+o)

	spCov.SliceSquare(0, p).(*mat.SymDense).CopySym(k.p)
	if !k.model.StateNoise().Cov().(*mat.SymDense).IsZero() {
		spCov.SliceSquare(p, p+s).(*mat.SymDense).CopySym(k.model.StateNoise().Cov())
	}
	if !k.model.OutputNoise().Cov().(*mat.SymDense).IsZero() {
		spCov.SliceSquare(p+s, p+s+o).(*mat.SymDense).CopySym(k.model.OutputNoise().Cov())
	}

	//fmt.Println(matrix.Format(spCov))

	var svd mat.SVD
	ok := svd.Factorize(spCov, mat.SVDFull)
	if !ok {
		return nil, fmt.Errorf("SVD factorization failed")
	}
	SqrtCov := new(mat.Dense)
	svd.UTo(SqrtCov)
	vals := svd.Values(nil)
	for i := range vals {
		vals[i] = math.Sqrt(vals[i])
	}
	diag := mat.NewDiagonal(len(vals), vals)
	//fmt.Println(matrix.Format(diag))

	SqrtCov.Mul(SqrtCov, diag)
	SqrtCov.Scale(k.gamma, SqrtCov)

	//fmt.Println(matrix.Format(SqrtCov))

	r, c := SqrtCov.Dims()
	// copy SqrtCov values into sigma points covariance matrix
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			spCov.SetSym(i, j, SqrtCov.At(i, j))
		}
	}

	//fmt.Println(matrix.Format(spCov))

	//fmt.Println(matrix.Format(x))
	// we center the sigmapoints around the mean sigma point stored in 1st column
	for j := 0; j < cols; j++ {
		sp.Slice(0, rows, j, j+1).(*mat.Dense).Copy(x)
	}
	//fmt.Println(matrix.Format(sp))
	// positive sigma points
	sx := sp.Slice(0, rows, 1, 1+((cols-1)/2)).(*mat.Dense)
	sx.Add(sx, spCov)
	// negative sigma points
	sx = sp.Slice(0, rows, 1+((cols-1)/2), cols).(*mat.Dense)
	sx.Sub(sx, spCov)

	//fmt.Println(matrix.Format(sp))

	return &SigmaPoints{
		X:   sp,
		Cov: spCov,
	}, nil
}

// propagateSigmaPoints propagates sigma points to the next step and observes their output.
// It calculates mean predicted sigma point state and returns it with predicted sigma points states.
// It returns error if it fails to propagate the sigma points or observe their outputs.
func (k *UKF) propagateSigmaPoints(sp *SigmaPoints, u mat.Vector) (*sigmaPointsPred, error) {
	in, out := k.model.Dims()
	_, cols := sp.X.Dims()

	// sigmaPred and sigmaOutPred store predicted sigma point states and outputs
	x := mat.NewDense(in, cols, nil)
	y := mat.NewDense(out, cols, nil)

	// xmPred and ymPred store predicted mean sigma point and its output
	xMean := mat.NewVecDense(in, nil)
	yMean := mat.NewVecDense(out, nil)

	// propagate all sigma points and observe their output
	for c := 0; c < cols; c++ {
		sigmaNext, err := k.model.Propagate(sp.X.ColView(c).(*mat.VecDense).SliceVec(0, in), u)
		if err != nil {
			return nil, fmt.Errorf("Failed to propagate sigma point: %v", err)
		}
		x.Slice(0, sigmaNext.Len(), c, c+1).(*mat.Dense).Copy(sigmaNext)

		sigmaOut, err := k.model.Observe(sigmaNext, u)
		if err != nil {
			return nil, fmt.Errorf("Failed to observe sigma point output: %v", err)
		}
		y.Slice(0, sigmaOut.Len(), c, c+1).(*mat.Dense).Copy(sigmaOut)

		if c == 0 {
			xMean.AddScaledVec(xMean, k.Wm0, sigmaNext)
			yMean.AddScaledVec(yMean, k.Wm0, sigmaOut)
		} else {
			xMean.AddScaledVec(xMean, k.W, sigmaNext)
			yMean.AddScaledVec(yMean, k.W, sigmaOut)
		}
	}

	return &sigmaPointsPred{
		x:     x,
		xMean: xMean,
		y:     y,
		yMean: yMean,
	}, nil
}

// predictCovariance predicts UKF covariance and returns it.
// It returns error if it fails to calculate predicted covariance from predicted sigma points.
func (k *UKF) predictCovariance(x *mat.Dense, xMean *mat.VecDense) (*mat.SymDense, error) {
	rows, cols := x.Dims()

	// ppred is predicted covariance
	ppred := mat.NewSymDense(rows, nil)
	cov := mat.NewDense(rows, rows, nil)
	sigmaVec := mat.NewVecDense(rows, nil)

	for c := 0; c < cols; c++ {
		sigmaVec.CopyVec(x.ColView(c))
		sigmaVec.SubVec(sigmaVec, xMean)
		cov.Mul(sigmaVec, sigmaVec.T())

		if c == 0 {
			cov.Scale(k.Wc0, cov)
		} else {
			cov.Scale(k.W, cov)
		}

		for i := 0; i < rows; i++ {
			for j := i; j < rows; j++ {
				ppred.SetSym(i, j, ppred.At(i, j)+cov.At(i, j))
			}
		}
	}

	return ppred, nil
}

// Predict predicts the next output of the system given the state x and input u and returns its estimate.
// It returns error if it either fails to generate or propagate sigma points (and x) to the next state.
func (k *UKF) Predict(x, u mat.Vector) (filter.Estimate, error) {
	// generate new sigma points around the new state
	sigmaPoints, err := k.GenSigmaPoints(x)
	if err != nil {
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
		return nil, fmt.Errorf("Failed to observe system output: %v", err)
	}

	sigmaPointsNext, err := k.propagateSigmaPoints(sigmaPoints, u)
	if err != nil {
		return nil, fmt.Errorf("Failed to propagate sigma points: %v", err)
	}

	cov, err := k.predictCovariance(sigmaPointsNext.x, sigmaPointsNext.xMean)
	if err != nil {
		return nil, fmt.Errorf("Failed to predict covariance: %v", err)
	}

	est, err := estimate.NewBaseWithCov(xNext, yNext, cov)
	if err != nil {
		return nil, fmt.Errorf("Failed to estimate next state: %v", err)
	}

	// it's safe to update the filter state
	k.spred.x.Copy(sigmaPointsNext.x)
	k.spred.xMean.CopyVec(sigmaPointsNext.xMean)
	k.spred.y.Copy(sigmaPointsNext.y)
	k.spred.yMean.CopyVec(sigmaPointsNext.yMean)
	k.ppred.CopySym(cov)

	return est, nil
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (k *UKF) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	in, out := k.model.Dims()
	// covariance of x and y, where y is predicted sigma point output
	pxy := mat.NewDense(in, out, nil)
	// predicted sigma points output covariance
	pyy := mat.NewDense(out, out, nil)

	// these are helper vectors used in calculating covariances
	sigmaVec := mat.NewVecDense(in, nil)
	sigmaOutVec := mat.NewVecDense(out, nil)

	// these are helper matrices which hold intermediary covariances
	covxy := mat.NewDense(in, out, nil)
	covyy := mat.NewDense(out, out, nil)

	_, cols := k.sp.X.Dims()
	for c := 0; c < cols; c++ {
		sigmaVec.CopyVec(k.spred.x.ColView(c))
		sigmaVec.SubVec(sigmaVec, k.spred.xMean)

		sigmaOutVec.CopyVec(k.spred.y.ColView(c))
		sigmaOutVec.SubVec(sigmaOutVec, k.spred.yMean)

		covxy.Mul(sigmaVec, sigmaOutVec.T())
		covyy.Mul(sigmaOutVec, sigmaOutVec.T())

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
	inn.SubVec(z, k.spred.yMean)

	// correct state x
	corr := &mat.Dense{}
	corr.Mul(gain, inn)
	x.(*mat.VecDense).AddVec(x, corr.ColView(0))

	// correct UKF covariance
	kr := &mat.Dense{}
	kr.Mul(pyy, gain.T())
	pCorr := &mat.Dense{}
	pCorr.Mul(gain, kr)
	pCorr.Sub(k.ppred, pCorr)

	output, err := k.model.Observe(x, u)
	if err != nil {
		return nil, fmt.Errorf("Unable to calculate output estimate: %v", err)
	}

	est, err := estimate.NewBaseWithCov(x, output, k.p)
	if err != nil {
		return nil, fmt.Errorf("Failed to update estimate: %v", err)
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

	return est, nil
}

// Run runs one step of UKF for given state x, input u and measurement z.
// It corrects system state x using measurement z and returns new system estimate.
// It returns error if it either fails to propagate or correct state x or UKF sigma points.
func (k *UKF) Run(x, u, z mat.Vector) (filter.Estimate, error) {
	return nil, nil
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
