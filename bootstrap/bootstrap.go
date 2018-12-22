package bootstrap

import (
	"fmt"
	"math"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/go-filter/matrix"
	"github.com/milosgajdos83/go-filter/rnd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// Bootstrap is a Bootstrap Filter (BF) aka Particle Filter
// For more information about Bootstrap (Particle) Filter see:
// https://en.wikipedia.org/wiki/Particle_filter
type Bootstrap struct {
	// model is bootstrap filter model
	model filter.Model
	// w stores particle weights
	w []float64
	// x stores filter particles
	x *mat.Dense
	// y stores particle outputs
	y *mat.Dense
	// diff is a buffer which stores a diff // between measurement and each particle output.
	// The size of diff is fixed -- it's equal to the size of output,
	// so we preallocate it to avoid reallocating it on every call to Update()
	diff []float64
	// errPDF is PDF (Probability Density Function) of filter output error
	errPDF distmv.LogProber
}

// New creates new Bootstrap Filter with following parameters:
// - model: system model
// - init:  initial condition of the filter
// - particles:  number of filter particles
// - pdf:   Probability Density Function (PDF) of filter output error
// It returns error if non-positive number of particles is given or if the particles fail to be generated.
func New(model filter.Model, init filter.InitCond, particles int, pdf distmv.LogProber) (*Bootstrap, error) {
	// must have at least one particle; can't be negative
	if particles <= 0 {
		return nil, fmt.Errorf("Invalid particle count: %d", particles)
	}

	// size of input and output vectors
	in, out := model.Dims()
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("Invalid model dimensions: [%d x %d]", in, out)
	}

	// Initialize particle weights to equal probabilities:
	// particle weights must sum up to 1 to represent probability
	w := make([]float64, particles)
	for i := range w {
		w[i] = 1 / float64(particles)
	}

	// draw particles from distribution with covariance init.Cov()
	x, err := rnd.WithCovN(init.Cov(), particles)
	if err != nil {
		return nil, fmt.Errorf("Failed to generate filter particles: %v", err)
	}

	rows, cols := x.Dims()
	// center particles around initial condition init.State()
	for c := 0; c < cols; c++ {
		for r := 0; r < rows; r++ {
			x.Set(r, c, x.At(r, c)+init.State().AtVec(r))
		}
	}

	y := mat.NewDense(out, particles, nil)
	diff := make([]float64, out)

	return &Bootstrap{
		model:  model,
		w:      w,
		x:      x,
		y:      y,
		diff:   diff,
		errPDF: pdf,
	}, nil
}

// Predict predicts the next output of the system given the state x and input u and returns it.
// It returns error if it fails to propagate either the filter particles or x to the next state.
func (b *Bootstrap) Predict(x, u mat.Vector) (filter.Estimate, error) {
	// propagate input state to the next step
	xNext, err := b.model.Propagate(x, u)
	if err != nil {
		return nil, fmt.Errorf("System state propagation failed: %v", err)
	}

	// propagate filter particles to the next step
	for c := range b.w {
		xPartNext, err := b.model.Propagate(b.x.ColView(c), u)
		if err != nil {
			return nil, fmt.Errorf("Particle state propagation failed: %v", err)
		}
		b.x.Slice(0, xPartNext.Len(), c, c+1).(*mat.Dense).Copy(xPartNext)
	}

	// observe system output in the next step
	yNext, err := b.model.Observe(xNext, u)
	if err != nil {
		return nil, fmt.Errorf("System state observation failed: %v", err)
	}

	// observe particle output in the next step
	for c := range b.w {
		yPartNext, err := b.model.Observe(b.x.ColView(c), u)
		if err != nil {
			return nil, fmt.Errorf("Particle state observation failed: %v", err)
		}
		b.y.Slice(0, yPartNext.Len(), c, c+1).(*mat.Dense).Copy(yPartNext)
	}

	return estimate.NewBase(xNext, yNext), nil
}

// Update corrects state x using the measurement z, given control intput u and returns corrected estimate.
// It returns error if either invalid state was supplied or if it fails to calculate system output estimate.
func (b *Bootstrap) Update(x, u, z mat.Vector) (filter.Estimate, error) {
	// get measurement dimensions
	zRows := z.Len()
	if zRows != len(b.diff) {
		return nil, fmt.Errorf("Invalid measurement size: %d", zRows)
	}

	// Update particle weights:
	// - calculate observation error for each particle output
	// - multiply the resulting error with particle weight
	for c := range b.w {
		for r := 0; r < zRows; r++ {
			b.diff[r] = z.At(r, 0) - b.y.ColView(c).AtVec(r)
		}
		b.w[c] = b.w[c] * math.Exp(b.errPDF.LogProb(b.diff))
	}

	// normalize the particle weights so they express probability
	floats.Scale(1/floats.Sum(b.w), b.w)

	// attempt to convert x to *mat.VecDense
	state, ok := x.(*mat.VecDense)
	if !ok {
		return nil, fmt.Errorf("Invalid state supplied: %v", x)
	}

	pRows, _ := b.x.Dims()
	wavg := 0.0
	// update/correct particles estimates to weighted average
	for r := 0; r < pRows; r++ {
		for c := range b.w {
			wavg += b.w[c] * b.x.At(r, c)
		}
		state.SetVec(r, wavg)
		wavg = 0.0
	}

	// calculate corrected output estimate
	output, err := b.model.Observe(x, u)
	if err != nil {
		return nil, fmt.Errorf("Unable to calculate output estimate: %v", err)
	}

	return estimate.NewBase(state, output), nil
}

// Run runs one step of Bootstrap Filter for given state x, input u and measurement z.
// It corrects system state x using measurement z and returns new system estimate.
// It returns error if it fails to either propagate or correct state x.
func (b *Bootstrap) Run(x, u, z mat.Vector) (filter.Estimate, error) {
	// predict the next output state
	pred, err := b.Predict(x, u)
	if err != nil {
		return nil, err
	}

	// correct the output and return it
	est, err := b.Update(pred.State(), u, z)
	if err != nil {
		return nil, err
	}

	return est, nil
}

// Resample allows to resample filter particles with regularization parameter alpha.
// It generates new filter particles and replaces the existing ones with them.
// If invalid (non-positive) alpha is provided we use optimal alpha for gaussian kernel.
// It returns error if it fails to generate new filter particles.
func (b *Bootstrap) Resample(alpha float64) error {
	// randomly pick new particles based on their weights
	// rnd.RouletteDrawN returns a slice of column indices to b.x
	indices, err := rnd.RouletteDrawN(b.w, len(b.w))
	if err != nil {
		return fmt.Errorf("Failed to sample filter particles: %v", err)
	}

	// we need to clone b.x to avoid overriding the existing filter particles
	x := new(mat.Dense)
	x.Clone(b.x)
	rows, cols := x.Dims()
	// length of inidices slice is the same as number of columns: number of particles
	for c := range indices {
		b.x.Slice(0, rows, c, c+1).(*mat.Dense).Copy(x.ColView(indices[c]))
	}

	// we have resampled particles, therefore we must reinitialize their weights, too:
	// weights will have the same probability: 1/len(b.w): they must sum up to 1
	for i := 0; i < len(b.w); i++ {
		b.w[i] = 1 / float64(len(b.w))
	}

	// We need to calculate covariance matrix of particles
	cov, err := matrix.Cov(b.x, "cols")
	if err != nil {
		return fmt.Errorf("Failed to calculate covariance matrix: %v", err)
	}

	// randomly draw values with given particle covariance
	m, err := rnd.WithCovN(cov, cols)
	if err != nil {
		return fmt.Errorf("Failed to draw random particle pertrubations: %v", err)
	}

	// if invalid alpha is given, use the optimal value for Gaussian
	if alpha <= 0 {
		alpha = AlphaGauss(rows, cols)
	}
	m.Scale(alpha, m)

	// add random perturbations to the new particles
	b.x.Add(b.x, m)

	return nil
}

// AlphaGauss computes optimal regulariation parameter for Gaussian kernel and returns it.
func AlphaGauss(r, c int) float64 {
	return math.Pow(4.0/(float64(c)*(float64(r)+2.0)), 1/(float64(r)+4.0))
}
