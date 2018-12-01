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
	// Propagator is particle filter propagator
	Propagator filter.Propagator
	// Observer is particle filter Observer
	Observer filter.Observer
	// ParticleCount specifies number of filter particles
	ParticleCount int
	// Err is output error probability density function (PDF)
	Err distmv.RandLogProber
}

// InitCond is Bootstrap Filter initial state condition
type InitCond struct {
	// State is initial state
	State *mat.Dense
	// Cov is initial covariance
	Cov *mat.Dense
}

// Bootstrap is a Bootstrap Filter aka Particle Filter
// For more information about Bootstrap (Particle) Filter see:
// https://en.wikipedia.org/wiki/Particle_filter
type Bootstrap struct {
	// p propagates internal system state
	p filter.Propagator
	// o observes external system state
	o filter.Observer
	// w stores  particle weights
	w []float64
	// x stores filter particles
	x *mat.Dense
	// errDist isutput error distribution PDF
	err distmv.RandLogProber
}

// New creates new Bootstrap Filter with config c and returns it.
// It returns error if the filter fails to be created.
func New(c *Config) (*Bootstrap, error) {
	// must have at least one particle; can't be negative
	if c.ParticleCount <= 0 {
		return nil, fmt.Errorf("Invalid particle count: %d", c.ParticleCount)
	}

	// initialize particle weights to equal probabilities
	// particle weights must sum up to 1 to represent probability
	w := make([]float64, c.ParticleCount)
	for i := 0; i < c.ParticleCount; i++ {
		w[i] = 1 / float64(c.ParticleCount)
	}

	return &Bootstrap{
		p:   c.Propagator,
		o:   c.Observer,
		w:   w,
		err: c.Err,
	}, nil
}

// Init initializes Bootstrap Filter state to start initial condition and returns error if it fails.
func (b *Bootstrap) Init(start *InitCond) error {
	// initialize filter particles by drawing randomly from distribution with covariance stat.Cov
	x, err := rnd.WithCovN(start.Cov, len(b.w))
	if err != nil {
		return fmt.Errorf("Failed to initialize filter: %v", err)
	}
	rows, cols := x.Dims()

	for c := 0; c < cols; c++ {
		for r := 0; r < rows; r++ {
			x.Set(r, c, x.At(r, c)+start.State.At(r, 0))
		}
	}

	b.x = x

	return nil
}

// Run runs Bootstrap Filter for given system parameters.
// It accepts following parameters, all being single column matrices:
// - x: internal system state (Nx1; N: dimension of state)
// - u: system inputs stores as columns (Mx1; M: dimension of input)
// - z: external measurements (Rx1; R: dimension of output)
// It corrects system state x in place using measurement z and returns it.
// It returns error if it fails to correct x state.
func (b *Bootstrap) Run(x, u, z *mat.Dense) (*mat.Dense, error) {
	// propagate particles to the next step
	xNext, err := b.p.Propagate(b.x, u)
	if err != nil {
		return nil, fmt.Errorf("State propagation failed: %v", err)
	}
	// observe particle system output in the next step
	zNext, err := b.o.Observe(xNext, u)
	if err != nil {
		return nil, fmt.Errorf("State observation failed: %v", err)
	}

	// get state and measurement matrix dimensions
	xRows, _ := x.Dims()
	zRows, _ := z.Dims()

	// update particle weights:
	// - calculate observation error for each particle output
	// - multiply the resulting error with particle weight
	diff := make([]float64, zRows)
	for c := range b.w {
		for r := 0; r < zRows; r++ {
			diff[r] = z.At(r, 0) - zNext.ColView(c).AtVec(r)
		}
		b.w[c] = b.w[c] * math.Exp(b.err.LogProb(diff))
	}
	// normalize the weights so they express probability
	floats.Scale(floats.Sum(b.w), b.w)

	// update/correct particles estimates to weighted average
	for c := range b.w {
		for r := 0; r < xRows; r++ {
			b.x.Set(r, c, b.x.At(r, c)*b.w[c])
		}
	}
	// set new x estimate and return it
	x.SetCol(0, matrix.RowSums(b.x))

	return x, nil
}

// Resample allows to resample filter particles.
// It runs the filter and replaces the existing particles with new ones
// It allows to specify a regularization parameter alpha.
// If invalid (non-positive) alpha is provided we compute optimal alpha for  gaussian kernel.
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

	// zero mean (particle) state values
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
	b.x.Add(b.x, m)

	return nil
}

// AlphaGauss computes optimal regulariation parameter for Gaussian kernel and returns it.
func AlphaGauss(r, c int) float64 {
	return math.Pow(4.0/(float64(c)*(float64(r)+2.0)), 1/(float64(r)+4.0))
}
