package sim

import "gonum.org/v1/gonum/mat"

// InitCond implements filter.InitCond
type InitCond struct {
	state *mat.VecDense
	cov   *mat.SymDense
}

// NewInitCond creates new InitCond and returns it
func NewInitCond(state mat.Vector, cov mat.Symmetric) *InitCond {
	s := &mat.VecDense{}
	s.CloneFromVec(state)

	c := mat.NewSymDense(cov.Symmetric(), nil)
	c.CopySym(cov)

	return &InitCond{
		state: s,
		cov:   c,
	}
}

// State returns initial state
func (c *InitCond) State() mat.Vector {
	state := mat.NewVecDense(c.state.Len(), nil)
	state.CloneFromVec(c.state)

	return state
}

// Cov returns initial covariance
func (c *InitCond) Cov() mat.Symmetric {
	cov := mat.NewSymDense(c.cov.Symmetric(), nil)
	cov.CopySym(c.cov)

	return cov
}
