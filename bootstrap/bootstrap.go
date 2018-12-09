package bootstrap

import (
	"fmt"
	"math"
	"math/rand"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/matrix"
	"github.com/milosgajdos83/go-filter/rnd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// Config is Bootstrap Filter configuration
type Config struct {
	// Model is a system model
	Model filter.Model
	// ParticleCount specifies number of filter particles
	ParticleCount int
	// Err probability density function (PDF) of output error
	Err distmv.LogProber
}

// InitCond is Bootstrap Filter initial state condition
type InitCond struct {
	// State is initial state
	State *mat.Dense
	// Cov is initial covariance
	Cov *mat.SymDense
}

// Bootstrap is a Bootstrap Filter (BF) aka Particle Filter
// For more information about Bootstrap (Particle) Filter see:
// https://en.wikipedia.org/wiki/Particle_filter
type Bootstrap struct {
	// Model is bootstrap filter system model
	Model filter.Model
	// w stores particle weights
	w []float64
	// x stores filter particles
	x *mat.Dense
	// y stores particle outputs
	y *mat.Dense
	// diff is a buffer that stores a diff
	// between measurement and particle output
	// we pre-allocate it in Init() so it doesnt
	// have to be reallocated on every call to Correct()
	diff []float64
	// ErrPDF is PDF (Probab. Density Function) of filter output error
	ErrPDF distmv.LogProber
}

// NewFilter creates new Bootstrap Filter with config c and returns it.
// It returns error if non-negative number or filter particles is given.
func NewFilter(c *Config) (*Bootstrap, error) {
	// must have at least one particle; can't be negative
	if c.ParticleCount <= 0 {
		return nil, fmt.Errorf("Invalid particle count: %d", c.ParticleCount)
	}

	// Initialize particle weights to equal probabilities:
	// particle weights must sum up to 1 to represent probability
	w := make([]float64, c.ParticleCount)
	for i := 0; i < c.ParticleCount; i++ {
		w[i] = 1 / float64(c.ParticleCount)
	}

	return &Bootstrap{
		Model:  c.Model,
		w:      w,
		ErrPDF: c.Err,
	}, nil
}

// Init initializes internal state of Bootstrap Filter to initial condition init.
// It returns error if it fails to generate filter particles.
func (b *Bootstrap) Init(init *InitCond) error {
	// draw particles from distribution with covariance start.Cov
	x, err := rnd.WithCovN(init.Cov, len(b.w))
	if err != nil {
		return fmt.Errorf("Failed to initialize filter: %v", err)
	}

	rows, cols := x.Dims()
	// center particles around initial condition start.State
	for c := 0; c < cols; c++ {
		for r := 0; r < rows; r++ {
			val := x.At(r, c)
			x.Set(r, c, val+init.State.At(r, 0))
		}
	}
	b.x = x

	// initialise particle filter output
	_, out := b.Model.Dims()
	b.y = mat.NewDense(out, cols, nil)

	b.diff = make([]float64, out)

	return nil
}

// Predict predicts the next output of the system given the state x and input u and returns it.
// Both input parameters are single column matrices:
// - x: system inputs (Nx1; N: dimension of input state)
// - u: system control inputs stored as columns (Mx1; M: dimension of input)
// It returns error if it fails to propagate either the system or the particles to the next state.
func (b *Bootstrap) Predict(x, u *mat.Dense) (*mat.Dense, error) {
	// propagate input state to the next step
	xNext, err := b.Model.Propagate(x, u)
	if err != nil {
		return nil, fmt.Errorf("Input state propagation failed: %v", err)
	}

	// propagate particle filters to the next step
	for i := range b.w {
		xPartNext, err := b.Model.Propagate(b.x.ColView(i), u)
		if err != nil {
			return nil, fmt.Errorf("Particle state propagation failed: %v", err)
		}
		vec, ok := xPartNext.ColView(0).(*mat.VecDense)
		if !ok {
			return nil, fmt.Errorf("Invalid state returned: %v", xPartNext)
		}
		b.x.SetCol(i, vec.RawVector().Data)
	}

	// observe system output in the next step
	yNext, err := b.Model.Observe(xNext, u)
	if err != nil {
		return nil, fmt.Errorf("Output state observation failed: %v", err)
	}

	// observe particle output in the next step
	for i := range b.w {
		yPartNext, err := b.Model.Observe(b.x.ColView(i), u)
		if err != nil {
			return nil, fmt.Errorf("Particle state observation failed: %v", err)
		}
		vec, ok := yPartNext.ColView(0).(*mat.VecDense)
		if !ok {
			return nil, fmt.Errorf("Invalid output returned: %v", yPartNext)
		}
		b.y.SetCol(i, vec.RawVector().Data)
	}

	return yNext, nil
}

// Correct corrects the system state x and output y using the measurement z and returns it.
// Both function parameters are single column matrices:
// - x: system state to correct (Nx1; N: dimension of system state)
// - z: external measurements (Rx1; R: dimension of measurement)
func (b *Bootstrap) Correct(x, z *mat.Dense) (*mat.Dense, error) {
	// get output matrix dimensions
	rows, _ := z.Dims()

	// Update particle weights:
	// - calculate observation error for each particle output
	// - multiply the resulting error with particle weight
	for c := range b.w {
		for r := 0; r < rows; r++ {
			b.diff[r] = z.At(r, 0) - b.y.ColView(c).AtVec(r)
		}
		b.w[c] = b.w[c] * math.Exp(b.ErrPDF.LogProb(b.diff))
	}

	// normalize the particle weights so they express probability
	floats.Scale(1/floats.Sum(b.w), b.w)

	wavg := 0.0
	// update/correct particles estimates to weighted average
	for r := 0; r < rows; r++ {
		for c := range b.w {
			wavg += b.w[c] * b.x.At(r, c)
		}
		x.Set(r, 0, wavg)
		wavg = 0.0
	}

	return x, nil
}

// Run runs Bootstrap Filter for given system parameters.
// It accepts following parameters, all being single column matrices:
// - x: internal system state (Nx1; N: dimension of state)
// - u: system inputs stores as columns (Mx1; M: dimension of input)
// - z: external measurements (Rx1; R: dimension of output)
// It corrects system state x in place using measurement z and returns it.
// It returns error if it fails to correct x state.
func (b *Bootstrap) Run(x, u, z *mat.Dense) (*mat.Dense, error) {
	// predict the next state
	y, err := b.Predict(x, u)
	if err != nil {
		return nil, err
	}

	// correct the output and return it
	xCor, err := b.Correct(y, z)
	if err != nil {
		return nil, err
	}

	return xCor, nil
}

// Resample allows to resample filter particles.
// It runs the filter and replaces the existing particles with new ones
// It allows to specify a regularization parameter alpha.
// If invalid (non-positive) alpha is provided we use optimal alpha for gaussian kernel.
func (b *Bootstrap) Resample(alpha float64) error {
	// randomly pick new particles based on their weights
	// rnd.RouletteDrawN returns a slice of column indices to b.x
	indices, err := rnd.RouletteDrawN(b.w, len(b.w))
	if err != nil {
		return fmt.Errorf("Failed to initialize filter particles: %v", err)
	}

	// we need to clone b.x to avoid overriding the filter particles
	x := new(mat.Dense)
	x.Clone(b.x)
	rows, cols := x.Dims()
	for c := range indices {
		for r := 0; r < rows; r++ {
			b.x.Set(r, c, x.At(r, indices[c]))
		}
	}

	// we have resampled particles, therefore we must reinitialize the weights, too
	// weights will have the same probability: 1/len(b.w): they must sum up to 1
	for i := 0; i < len(b.w); i++ {
		b.w[i] = 1 / float64(len(b.w))
	}

	// we need to create covariance matrix of particles
	// 1. we will calculate zero mean of the particle statues
	// 2. X * X^T will give us particle state covariance
	rowAvgs := matrix.RowSums(b.x)
	floats.Scale(float64(len(b.w)), rowAvgs)
	for c := range b.w {
		for r := 0; r < rows; r++ {
			x.Set(r, c, x.At(r, c)-rowAvgs[r])
		}
	}
	sigma := new(mat.Dense)
	sigma.Mul(x, x.T())
	sigma.Scale(1/(float64(len(b.w))-1.0), sigma)

	// factorize the matrix
	var svd mat.SVD
	ok := svd.Factorize(sigma, mat.SVDFull)
	if !ok {
		return fmt.Errorf("SVD factorization failed")
	}
	U := new(mat.Dense)
	svd.UTo(U)
	vals := svd.Values(nil)
	for i := range vals {
		vals[i] = math.Sqrt(vals[i])
	}
	diag := mat.NewDiagonal(len(vals), vals)
	U.Mul(U, diag)

	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	m := mat.NewDense(rows, cols, data)
	m.Mul(U, m)

	// if invalid alpha is given, use the optimal param for Gaussian
	if alpha <= 0 {
		alpha = AlphaGauss(rows, cols)
	}
	m.Scale(alpha, m)
	// draw random perturbations
	b.x.Add(b.x, m)

	return nil
}

// AlphaGauss computes optimal regulariation parameter for Gaussian kernel and returns it.
func AlphaGauss(r, c int) float64 {
	return math.Pow(4.0/(float64(c)*(float64(r)+2.0)), 1/(float64(r)+4.0))
}
