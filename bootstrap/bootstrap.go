package bootstrap

import (
	filter "github.com/milosgajdos83/go-filter"
	"gonum.org/v1/gonum/mat"
)

// Config stores Bootstrap Filter configuration
type Config struct {
	// Prop is system Propagator
	Propagator filter.Propagator
	// Obsrv is system Observer
	Observer filter.Observer
}

// Bootstrap is a Bootstrap filter (BF) aka Particle Filter (PF)
// For more information about BF/PF see:
// https://en.wikipedia.org/wiki/Particle_filter
type Bootstrap struct {
	// p is Propagator of the systems internal state
	p filter.Propagator
	// o is Observer of the systems external state
	o filter.Observer
}

// New creates new Bootstrap Filter (BF) using the provided configuration and returns it.
// It returns error if the filter couldnt be created.
func New(c *Config) (*Bootstrap, error) {
	return &Bootstrap{
		p: c.Propagator,
		o: c.Observer,
	}, nil
}

// Run runs one step of Bootstrap filter algorithm.
// It accepts following parameters:
// - internal system states X
// - system inputs U
// - external measurements Z
// - resample requests particle filter weights resampling
// - alpha is particle filter regulariser parameter
// It returns the corrected system state estimates or error.
func (b *Bootstrap) Run(X, U, Z *mat.Dense, resample bool, alpha float64) (*mat.Dense, error) {
	return nil, nil
}
