package sim

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

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

// BaseModel is a basic model of a dynamical system
type BaseModel struct {
	// A is internal state matrix
	A *mat.Dense
	// B is control matrix
	B *mat.Dense
	// C is output state matrix
	C *mat.Dense
	// D is output control matrix
	D *mat.Dense
	// E is Disturbance matrix
	E *mat.Dense
}

// NewBaseModel creates a model of falling ball and returns it
func NewBaseModel(A, B, C, D, E *mat.Dense) (*BaseModel, error) {
	return &BaseModel{A: A, B: B, C: C, D: D, E: E}, nil
}

// Propagate propagates internal state x of a falling ball to the next step
// given an input vector u and a disturbance input z. (wd is process noise, z not implemented yet)
func (b *BaseModel) Propagate(x, u, wd mat.Vector) (mat.Vector, error) {
	_nx, _nu, _, _ := b.Dims()
	if u != nil && u.Len() != _nu {
		return nil, fmt.Errorf("invalid input vector")
	}

	if x.Len() != _nx {
		return nil, fmt.Errorf("invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(b.A, x)

	if u != nil && b.B != nil {
		outU := new(mat.Dense)
		outU.Mul(b.B, u)

		out.Add(out, outU)
	}

	if wd != nil && wd.Len() == _nx { // TODO change _nx to _nz when switching to z
		// outZ := new(mat.Dense) // TODO add E disturbance matrix
		// outZ.Mul(b.E, z)
		// out.Add(out, outZ)
		out.Add(out, wd)
	}

	return out.ColView(0), nil
}

// Observe observes external state of falling ball given internal state x and input u.
// wn is added to the output as a noise vector.
func (b *BaseModel) Observe(x, u, wn mat.Vector) (mat.Vector, error) {
	_nx, _nu, _ny, _ := b.Dims()
	if u != nil && u.Len() != _nu {
		return nil, fmt.Errorf("invalid input vector")
	}

	if x.Len() != _nx {
		return nil, fmt.Errorf("invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(b.C, x)

	if u != nil && b.D != nil {
		outU := new(mat.Dense)
		outU.Mul(b.D, u)

		out.Add(out, outU)
	}

	if wn != nil && wn.Len() == _ny {
		out.Add(out, wn)
	}

	return out.ColView(0), nil
}

// Dims returns input and output model dimensions.
// n is state vector length, p is input vector length, q is
// measured state length (output vector) and r is distrubance input length.
func (b *BaseModel) Dims() (nx, nu, ny, nz int) {
	nx, _ = b.A.Dims()
	if b.B != nil {
		_, nu = b.B.Dims()
	}
	ny, _ = b.C.Dims()
	if b.E != nil {
		_, nz = b.E.Dims()
	}
	return nx, nu, ny, nz
}

// SystemMatrix returns state propagation matrix
func (b *BaseModel) SystemMatrix() mat.Matrix {
	m := &mat.Dense{}
	m.CloneFrom(b.A)

	return m
}

// ControlMatrix returns state propagation control matrix
func (b *BaseModel) ControlMatrix() mat.Matrix {
	m := &mat.Dense{}
	if b.B != nil {
		m.CloneFrom(b.B)
	}

	return m
}

// OutputMatrix returns observation matrix
func (b *BaseModel) OutputMatrix() mat.Matrix {
	m := &mat.Dense{}
	m.CloneFrom(b.C)

	return m
}

// FeedForwardMatrix returns observation control matrix
func (b *BaseModel) FeedForwardMatrix() mat.Matrix {
	m := &mat.Dense{}
	if b.D != nil {
		m.CloneFrom(b.D)
	}
	return m
}
