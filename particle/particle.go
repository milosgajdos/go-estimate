package particle

import (
	filter "github.com/milosgajdos83/go-filter"
	"gonum.org/v1/gonum/mat"
)

// Particle is Particle Filter
type Particle interface {
	// filter.Filter is dynamical system filter
	filter.Filter
	// Weights returns particle weights
	Weights() mat.Vector
}
