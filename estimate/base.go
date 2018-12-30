package estimate

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Base is base estimate
type Base struct {
	// state is system state
	state *mat.VecDense
	// output is system output
	output *mat.VecDense
	// cov is state covariance
	cov *mat.SymDense
}

// NewBase returns base information estimate given state and output
func NewBase(state, output mat.Vector) (*Base, error) {
	s := &mat.VecDense{}
	if state != nil {
		s.CloneVec(state)
	}

	o := &mat.VecDense{}
	if output != nil {
		o.CloneVec(output)
	}

	c := mat.NewSymDense(s.Len(), nil)

	return &Base{
		state:  s,
		output: o,
		cov:    c,
	}, nil
}

// NewBaseWithCov returns base information estimate given state, output and covariance
func NewBaseWithCov(state, output mat.Vector, cov mat.Symmetric) (*Base, error) {
	rs, _ := state.Dims()
	rc := cov.Symmetric()

	if rs != rc {
		return nil, fmt.Errorf("Invalid dimensions. State: %d, Cov: %d x %d", rs, rc, rc)
	}

	s := &mat.VecDense{}
	s.CloneVec(state)

	o := &mat.VecDense{}
	o.CloneVec(output)

	c := mat.NewSymDense(cov.Symmetric(), nil)
	c.CopySym(cov)

	return &Base{
		state:  s,
		output: o,
		cov:    c,
	}, nil
}

// State returns state estimate
func (b *Base) State() mat.Vector {
	s := &mat.VecDense{}
	s.CloneVec(b.state)

	return s
}

// Output returns output estimate
func (b *Base) Output() mat.Vector {
	o := &mat.VecDense{}
	o.CloneVec(b.output)

	return o
}

// Cov returns covariance estimate
func (b *Base) Cov() mat.Symmetric {
	cov := mat.NewSymDense(b.cov.Symmetric(), nil)
	cov.CopySym(b.cov)

	return cov
}
