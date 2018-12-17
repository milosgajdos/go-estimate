package estimate

import "gonum.org/v1/gonum/mat"

// Base is base estimate
type Base struct {
	// state is system state
	state mat.Vector
	// output is system output
	output mat.Vector
}

// NewBase returns base information estimate
func NewBase(state, output mat.Vector) *Base {
	return &Base{
		state:  state,
		output: output,
	}
}

// State returns state estimate
func (b *Base) State() mat.Vector {
	return b.state
}

// Output returns output estimate
func (b *Base) Output() mat.Vector {
	return b.output
}

// Covariance returns covariance estimate
func (b *Base) Covariance() mat.Symmetric {
	cov := mat.NewSymDense(b.state.Len(), nil)
	dim := cov.Symmetric()

	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			cov.SetSym(r, c, b.state.AtVec(r)*b.state.T().At(0, c))
		}
	}
	cov.ScaleSym(1/float64(b.state.Len()-1), cov)

	return cov
}
