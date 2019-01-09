package filter

import "gonum.org/v1/gonum/mat"

// Filter is a dynamical system filter.
type Filter interface {
	// Predict estimates the next internal state of the system
	Predict(mat.Vector, mat.Vector) (Estimate, error)
	// Update updates the system state based on external measurement
	Update(mat.Vector, mat.Vector, mat.Vector) (Estimate, error)
}

// Propagator propagates internal state of the system to the next step
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

// DiscreteModel is a dynamical system whose state is driven by
// static propagation and observation dynamics matrices
type DiscreteModel interface {
	// Model is a model of a dynamical system
	Model
	// StateMatrix returns state propagation matrix
	StateMatrix() mat.Matrix
	// StateCtlMatrix returns state propagation control matrix
	StateCtlMatrix() mat.Matrix
	// OutputMatrix returns observation matrix
	OutputMatrix() mat.Matrix
	// OutputCtlMatrix returns observation control matrix
	OutputCtlMatrix() mat.Matrix
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
	// Val returns estimate value
	Val() mat.Vector
	// Cov returns estimate covariance
	Cov() mat.Symmetric
}

// Noise is dynamical system noise
type Noise interface {
	// Mean returns noise mean
	Mean() []float64
	// Cov returns covariance matrix of the noise
	Cov() mat.Symmetric
	// Sample returns a sample of the noise
	Sample() mat.Vector
	// Reset resets the noise
	Reset()
}
