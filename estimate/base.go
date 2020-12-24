package estimate

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Base is base estimate
type Base struct {
	// val is estimated value
	val *mat.VecDense
	// cov is estimated covariance
	cov *mat.SymDense
}

// NewBase returns base estimate given val
func NewBase(val mat.Vector) (*Base, error) {
	v := &mat.VecDense{}
	if val != nil {
		v.CloneFromVec(val)
	}

	c := mat.NewSymDense(v.Len(), nil)

	return &Base{
		val: v,
		cov: c,
	}, nil
}

// NewBaseWithCov returns base information estimate given state, output and covariance
func NewBaseWithCov(val mat.Vector, cov mat.Symmetric) (*Base, error) {
	rv, _ := val.Dims()
	rc := cov.Symmetric()

	if rv != rc {
		return nil, fmt.Errorf("Invalid dimensions. Val: %d, Cov: %d x %d", rv, rc, rc)
	}

	v := &mat.VecDense{}
	v.CloneFromVec(val)

	c := mat.NewSymDense(cov.Symmetric(), nil)
	c.CopySym(cov)

	return &Base{
		val: v,
		cov: c,
	}, nil
}

// Val returns estimated value
func (b *Base) Val() mat.Vector {
	v := &mat.VecDense{}
	v.CloneFromVec(b.val)

	return v
}

// Cov returns covariance estimate
func (b *Base) Cov() mat.Symmetric {
	cov := mat.NewSymDense(b.cov.Symmetric(), nil)
	cov.CopySym(b.cov)

	return cov
}
