package filter

import "gonum.org/v1/gonum/mat"

// Filter is a dynamical system filter.
type Filter interface {
	// Predict estimates the next system output
	Predict(mat.Vector, mat.Vector) (Estimate, error)
	// Update updates the system state based on external measurement
	Update(mat.Vector, mat.Vector, mat.Vector) (Estimate, error)
}

// Propagator propagates internal state of the system
type Propagator interface {
	// Propagate propagates internal state of the system to the next step
	Propagate(mat.Vector, mat.Vector, mat.Vector) (mat.Vector, error)
}

// Observer observes external state (output) of the system
type Observer interface {
	// Observe observes external state of the system
	Observe(mat.Vector, mat.Vector, mat.Vector) (mat.Vector, error)
}

// Model is a model of a dynamical system
type Model interface {
	// Propagator is system propagator
	Propagator
	// Observer is system observer
	Observer
	// Dims returns input and output dimensions of the model
	Dims() (in int, out int)
}

// InitCond is initial state condition of the filter
type InitCond interface {
	// State returns initial filter state
	State() mat.Vector
	// Cov returns initial state covariance
	Cov() mat.Symmetric
}

// Estimate is dynamical system filter estimate
type Estimate interface {
	// State returns state estimate
	State() mat.Vector
	// Output returns output estimate
	Output() mat.Vector
	// Cov returns state covariance
	Cov() mat.Symmetric
}

// Noise is dynamical system noise
type Noise interface {
	// Mean returns noise mean
	Mean() []float64
	// Cov returns noise covariance matrix
	Cov() mat.Symmetric
	// Sample returns a sample of the noise
	Sample() mat.Vector
	// Reset resets noise
	Reset()
}
