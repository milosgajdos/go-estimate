package filter

import "gonum.org/v1/gonum/mat"

// Filter is dynamical system filter.
// For more information about dynamical systems see:
// https://en.wikipedia.org/wiki/Dynamical_system#Linear_dynamical_systems
type Filter interface {
	// Runs one iteration of filtering algorithm.
	// It corrects internal system state based on provided measurement and returns it.
	Run(*mat.Dense, *mat.Dense, *mat.Dense) (*mat.Dense, error)
}

// Propagator propagates internal state
type Propagator interface {
	// Propagate propagates internal state of the system to the next step and returns it
	Propage(*mat.Dense, *mat.Dense) (*mat.Dense, error)
}

// Observer observes external state
type Observer interface {
	// Observe observer external state of the system and returns it
	Observe(*mat.Dense, *mat.Dense) (*mat.Dense, error)
}
