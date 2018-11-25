package bootstrap

import (
	"fmt"
	"math"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/matrix"
	"github.com/milosgajdos83/go-filter/rnd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// InitCond is Bootstrap Filter initial condition
type InitCond struct {
	// State is Bootstrap Filter initial state
	State *mat.Dense
	// Cov is initial state covariance matrix
	// It allows to specify confidence do in State
	Cov *mat.Dense
}

// Config is Bootstrap Filter configuration
type Config struct {
	// Prop is system Propagator
	Propagator filter.Propagator
	// Obsrv is system Observer
	Observer filter.Observer
	// ParticleCount is number of filter particles
	ParticleCount int
	// Alpha is particle filter regulariser
	Alpha float64
	// ErrDist is expected output error distribution
	ErrDist distmv.RandLogProber
}

// Bootstrap is a Bootstrap Filter (BF) aka Particle Filter (PF)
// For more information about BF/PF see:
// https://en.wikipedia.org/wiki/Particle_filter
type Bootstrap struct {
	// p is Propagator of the systems internal state
	p filter.Propagator
	// o is Observer of the systems external state
	o filter.Observer
	// alpha is particle filter regularise
	alpha float64
	// w stores particle weights
	w []float64
	// x stores filter particles
	x *mat.Dense
	// errDist is expected output error distribution
	errDist distmv.RandLogProber
}

// New creates new Bootstrap Filter (BF) and returns it.
// It accepts two parameters:
// - c: Filter configuration
// - init: Initial Filer state stored in columns
// It returns error if the filter fails to be created.
func New(c *Config, init *InitCond) (*Bootstrap, error) {
	// must have at least one particle; can't be negative
	if c.ParticleCount <= 0 {
		return nil, fmt.Errorf("Invalid particle count supplied: %d", c.ParticleCount)
	}

	// initialize particle weights to equal probabilities
	// particle weights must sum up to 1 to represent probability
	w := make([]float64, c.ParticleCount)
	for i := 0; i < c.ParticleCount; i++ {
		w[i] = 1 / float64(c.ParticleCount)
	}

	// initialize filter particles
	x, err := rnd.WithCovN(init.Cov, c.ParticleCount)
	if err != nil {
		return nil, fmt.Errorf("Failed to initialize BF particles: %v", err)
	}
	xRows, xCols := x.Dims()
	// center particles around initial state
	for c := 0; c < xCols; c++ {
		for r := 0; r < xRows; r++ {
			x.Set(r, c, x.At(r, c)+init.State.At(r, 0))
		}
	}

	alpha := c.Alpha
	// if invalid alpha is given, use the optimal param for Gaussian
	if alpha <= 0 {
		alpha = math.Pow(4.0/(float64(xCols)*(float64(xRows)+2.0)), 1/(float64(xRows)+4.0))
	}

	return &Bootstrap{
		p:       c.Propagator,
		o:       c.Observer,
		alpha:   alpha,
		w:       w,
		x:       x,
		errDist: c.ErrDist,
	}, nil
}

// Run runs Bootstrap Filter for given system parameters.
// It accepts following parameters:
// - x: internal system state (this is a Nx1 matrix)
// - u: system inputs stores as columns (this is a Mx1 matrix)
// - z: external measurements (this a Rx1 matrix)
// It corrects internal system state x in place and returns it.
// It returns error if it fails to correct internal state.
func (b *Bootstrap) Run(x, u, z *mat.Dense) (*mat.Dense, error) {
	// get state matrix dimensions
	xRows, _ := x.Dims()
	zRows, _ := z.Dims()

	// propagate particles to the next step
	xNext, err := b.p.Propagate(b.x, u)
	if err != nil {
		return nil, err
	}
	// observe particle system output in the next step
	zNext, err := b.o.Observe(xNext, u)
	if err != nil {
		return nil, err
	}

	// update particle weights:
	// - calculate observation error for each particle output
	// - multiply the resulting error with particle weight
	diff := make([]float64, zRows)
	for c := range b.w {
		for r := 0; r < zRows; r++ {
			diff[r] = z.At(r, 0) - zNext.ColView(c).AtVec(r)
		}
		b.w[c] = b.w[c] * math.Exp(b.errDist.LogProb(diff))
	}
	// normalize the weights so they express probability
	floats.Scale(floats.Sum(b.w), b.w)

	// update particles estimates
	for c := range b.w {
		for r := 0; r < xRows; r++ {
			b.x.Set(r, c, b.x.At(r, c)*b.w[c])
		}
	}
	// set new x estimate and return it
	x.SetCol(0, matrix.RowSums(b.x))

	return x, nil
}
