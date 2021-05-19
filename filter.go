package filter

import (
	"gonum.org/v1/gonum/mat"
)

// Filter is a dynamical system filter.
type Filter interface {
	// Predict returns a prediction of which will be
	// next internal state
	Predict(x, u mat.Vector) (Estimate, error)
	// Update returns estimated system state based on external measurement ym.
	Update(x, u, ym mat.Vector) (Estimate, error)
}

// DiscretePropagator propagates internal state of a discrete-time system to the next step
type DiscretePropagator interface {
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

// DiscreteModel is a discrete-time model of a dynamical system which contains all the
// logic to propagate internal state and observe external state.
type DiscreteModel interface {
	// Propagator is system propagator
	DiscretePropagator
	// Observer is system observer
	Observer
	// SystemDims returns the dimension of state vector, input vector,
	// output (measurements, written as y) vector and disturbance vector (only dynamical systems).
	// Below are dimension of matrices as returned by SystemDims() (row,column) (if DModel uses matrix representation)
	//  nx, nx = A.SystemDims()
	//  nx, nu = B.SystemDims()
	//  ny, nx = C.SystemDims()
	//  ny, nu = D.SystemDims()
	//  nx, nz = E.SystemDims()
	SystemDims() (nx, nu, ny, nz int)
}

// Smoother is a filter smoother
type Smoother interface {
	// Smooth implements filter smoothing and returns new estimates
	Smooth([]Estimate, []mat.Vector) ([]Estimate, error)
}

// DiscreteControlSystem is a dynamical system whose state is driven by
// static propagation and observation dynamics matrices
type DiscreteControlSystem interface {
	// DModel is a model of a dynamical system
	DiscreteModel
	// SystemMatrix returns state propagation matrix
	SystemMatrix() (A mat.Matrix)
	// ControlMatrix returns state propagation control matrix
	ControlMatrix() (B mat.Matrix)
	// OutputMatrix returns observation matrix
	OutputMatrix() (C mat.Matrix)
	// FeedForwardMatrix returns observation control matrix
	FeedForwardMatrix() (D mat.Matrix)
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
