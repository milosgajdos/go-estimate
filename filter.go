package filter

import "gonum.org/v1/gonum/mat"

// Filter is a dynamical system filter.
// For more information about dynamical systems see:
// https://en.wikipedia.org/wiki/Dynamical_system
type Filter interface {
	// Predict predicts the next system output
	Predict(*mat.Dense, *mat.Dense, *mat.Dense) (*mat.Dense, error)
	// Correct corrects the system output based on external measurement
	Correct(*mat.Dense, *mat.Dense, *mat.Dense) (*mat.Dense, error)
}

// Propagator propagates internal state of the system
type Propagator interface {
	// Propagate propagates internal state of the system to the next step
	Propagate(mat.Matrix, mat.Matrix) (*mat.Dense, error)
}

// Observer observes external state of the system
type Observer interface {
	// Observe observes external state of the system
	Observe(mat.Matrix, mat.Matrix) (*mat.Dense, error)
}

// Model is a model of the system
type Model interface {
	// Propagator is system propagator
	Propagator
	// Observer is system observer
	Observer
	// Dims returns input and output dimensions of the model
	Dims() (in int, out int)
}
