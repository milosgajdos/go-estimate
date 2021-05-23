package filter

import (
	"gonum.org/v1/gonum/mat"
)

// Filter is a dynamical system filter.
type Filter interface {
	// Predict returns the expected change in internal state
	Predict(x, u mat.Vector) (Estimate, error)
	// Update returns estimated system state based on external measurement ym.
	Update(x, u, ym mat.Vector) (Estimate, error)
}

// Propagator propagates internal state of the system to the next step
type Propagator interface {
	// Propagate propagates internal state of the system to the next step.
	// x is starting state, u is input vector, and z is disturbance input
	Propagate(x, u, z mat.Vector) (mat.Vector, error)
}

// Observer observes external state (output) of the system
type Observer interface {
	// Observe observes external state of the system.
	// Result for a linear system would be y=C*x+D*u+wn (last term is measurement noise)
	Observe(x, u, wn mat.Vector) (y mat.Vector, err error)
}

// Model is a model of a dynamical system
type Model interface {
	// Propagator is system propagator
	Propagator
	// Observer is system observer
	Observer
	// Dims returns the dimension of state vector, input vector,
	// output (measurements, written as y) vector and disturbance vector (only dynamical systems).
	// Below are dimension of matrices as returned by Dims() (row,column)
	//  nx, nx = A.Dims()
	//  nx, nu = B.Dims()
	//  ny, nx = C.Dims()
	//  ny, nu = D.Dims()
	//  nx, nz = E.Dims()
	Dims() (nx, nu, ny, nz int)
}

// Smoother is a filter smoother
type Smoother interface {
	// Smooth implements filter smoothing and returns new estimates
	Smooth([]Estimate, []mat.Vector) ([]Estimate, error)
}

// DiscreteModel is a dynamical system whose state is driven by
// static propagation and observation dynamics matrices
type DiscreteModel interface {
	// Model is a model of a dynamical system
	Model
	// SystemMatrix returns state propagation matrix
	SystemMatrix() (A mat.Matrix) // TODO rename to SystemMatrix
	// ControlMatrix returns state propagation control matrix
	ControlMatrix() (B mat.Matrix) // TODO rename to ControlMatrix
	// OutputMatrix returns observation matrix
	OutputMatrix() (C mat.Matrix)
	// FeedForwardMatrix returns observation control matrix
	FeedForwardMatrix() (D mat.Matrix) // TODO Rename to FeedMatrix/FeedForwardMatrix
	// TODO DisturbanceMatrix
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
