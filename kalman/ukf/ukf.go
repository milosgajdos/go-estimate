package ukf

import (
	"fmt"
	"math"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/matrix"
	"gonum.org/v1/gonum/mat"
)

// Config is UKF config
type Config struct {
	// Alpha is alpha parameter
	Alpha float64
	// Beta is beta parameter
	Beta float64
	// Kappa is kappa parameter
	Kappa float64
}

// UKF is Unscented (aka Sigma Point) Kalman Filter
type UKF struct {
	// Wm0 is mean sigma point weight
	Wm0 float64
	// Wc0 is mean sigma point covariance weight
	Wc0 float64
	// Wsp is weight for regular sigma points and covariances
	Wsp float64
	// xm is mean sigma point state
	xm *mat.VecDense
	// ym is mean sigma point output
	ym *mat.VecDense
	// x is a matrix which stores regular sigma points states
	x *mat.Dense
	// y is a matrix which stores regular sigma points outputs
	y *mat.Dense
	// p is UKF covariance matrix
	p *mat.Dense
	// innov is innovation vector
	innov *mat.VecDense
	// k is Kalman gain
	k *mat.Dense
}

// New creates new UKF and returns it.
// It accepts following arguments:
// - model:  system model
// - init:   initial condition of the filter
// - c:      filter configuration
// It returns error if non-positive number of sigma points is given or if the sigma points fail to be generated.
func New(model filter.Model, init filter.InitCond, c *Config) (*UKF, error) {
	// size of input and output vectors
	in, out := model.Dims()
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("Invalid model dimensions: [%d x %d]", in, out)
	}

	// sigmaDim stores dimension of sigma point
	sigmaDim := init.State().Len()

	stateCov := model.StateNoise().Cov()
	stateRows, _ := stateCov.Dims()
	sigmaDim += stateRows

	outCov := model.OutputNoise().Cov()
	outRows, _ := outCov.Dims()
	sigmaDim += outRows

	// lambda is one of unitless UKF parameters
	lambda := c.Alpha*c.Alpha*(float64(sigmaDim)+c.Kappa) - float64(sigmaDim)
	gamma := math.Sqrt(float64(sigmaDim) + lambda)

	// weight for mean sigma point
	Wm0 := lambda / (float64(sigmaDim) + lambda)
	// weight for mean sigma point covariance
	Wc0 := Wm0 + (1 - c.Alpha*c.Alpha + c.Beta)
	// weight for the rest of sigma points and covariance
	Wsp := 1 / (2 * (float64(sigmaDim) + lambda))

	// mean sigma point: sigmaDim x 1
	xm := mat.NewVecDense(sigmaDim, nil)
	xm.SliceVec(0, init.State().Len()).(*mat.VecDense).CopyVec(init.State())

	// sigma point covariance
	mx := []mat.Matrix{init.Cov(), stateCov, outCov}
	cov := matrix.BlockDiag(mx)

	// factorize covariance matrix
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
	diag := mat.NewDiagonal(len(vals), vals)
	SqrtCov.Mul(SqrtCov, diag)
	SqrtCov.Scale(gamma, SqrtCov)

	// regular sigma points
	x := mat.NewDense(in, 2*sigmaDim, nil)

	for j := 0; j < 2*sigmaDim; j++ {
		x.Slice(0, sigmaDim, j, j+1).(*mat.Dense).Copy(xm)
	}
	// positive sigmas
	sigmas := x.Slice(0, sigmaDim, 0, sigmaDim).(*mat.Dense)
	sigmas.Add(sigmas, SqrtCov)
	// negative sigmas
	sigmas = x.Slice(0, sigmaDim, sigmaDim, 2*sigmaDim).(*mat.Dense)
	sigmas.Sub(sigmas, SqrtCov)

	// innovation vector
	innov := mat.NewVecDense(out, nil)

	return &UKF{
		Wm0:   Wm0,
		Wc0:   Wc0,
		Wsp:   Wsp,
		xm:    xm,
		x:     x,
		innov: innov,
	}, nil
}

// Predict predicts the next output of the system given the state x and input u and returns it.
// it returns error if it fails to propagate either the filter particles or x to the next state.
func (k *UKF) Predict(x, u mat.Vector) (filter.Estimate, error) {
	return nil, nil
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (k *UKF) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	return nil, nil
}

// Covariance returns current state filter covariance
func (k *UKF) Covariance() mat.Symmetric {
	return nil
}

// Gain returns Kalman gain
func (k *UKF) Gain() mat.Matrix {
	return nil
}
