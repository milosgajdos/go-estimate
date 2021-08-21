package sim

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// System defines a linear model of a plant using
// traditional matrices of modern control theory.
//
// It contains the System (A), input (B), Observation/Output (C)
// Feedthrough (D) and disturbance (E) matrices.
type System struct {
	// System/State matrix A
	A *mat.Dense
	// Control/Input Matrix B
	B *mat.Dense
	// Observation/Output Matrix C
	C *mat.Dense
	// Feedthrough matrix D
	D *mat.Dense
	// Perturbation matrix (related to process noise wd) E
	E *mat.Dense
}

func newSystem(A, B, C, D, E mat.Matrix) System {
	sys := System{A: mat.DenseCopyOf(A)}
	if B != nil && B.(*mat.Dense) != nil {
		sys.B = mat.DenseCopyOf(B)
	}
	if C != nil && C.(*mat.Dense) != nil {
		sys.C = mat.DenseCopyOf(C)
	}
	if D != nil && D.(*mat.Dense) != nil {
		sys.D = mat.DenseCopyOf(D)
	}
	if E != nil && E.(*mat.Dense) != nil {
		sys.E = mat.DenseCopyOf(E)
	}
	return sys
}

// SystemDims returns internal state length (nx), input vector length (nu),
// external/observable/output state length (ny) and disturbance vector length (nz).
func (s System) SystemDims() (nx, nu, ny, nz int) {
	nx, _ = s.A.Dims()
	if s.B != nil {
		_, nu = s.B.Dims()
	}
	if s.C != nil {
		ny, _ = s.C.Dims()
	}
	if s.E != nil {
		_, nz = s.E.Dims()
	}
	return nx, nu, ny, nz
}

// SystemMatrix returns state propagation matrix `A`.
func (s System) SystemMatrix() (A mat.Matrix) { return s.A }

// ControlMatrix returns state propagation control matrix `B`
func (s System) ControlMatrix() (B mat.Matrix) {
	if s.B == nil {
		return nil
	}
	return s.B
}

// OutputMatrix returns observation matrix `C`
func (s System) OutputMatrix() (C mat.Matrix) {
	if s.C == nil {
		return nil
	}
	return s.C
}

// FeedForwardMatrix returns observation control matrix `D`
func (s System) FeedForwardMatrix() (D mat.Matrix) {
	if s.D == nil {
		return nil
	}
	return s.D
}

// Observe returns external/observable state given internal state x and input u.
// wn is added to the output as a noise vector.
func (s System) Observe(x, u, wn mat.Vector) (y mat.Vector, err error) {
	nx, nu, ny, _ := s.SystemDims()
	if u != nil && u.Len() != nu {
		return nil, fmt.Errorf("invalid input vector")
	}

	if x.Len() != nx {
		return nil, fmt.Errorf("invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(s.C, x)

	if u != nil && s.D != nil {
		outU := new(mat.Dense)
		outU.Mul(s.D, u)

		out.Add(out, outU)
	}

	if wn != nil && wn.Len() == ny {
		out.Add(out, wn)
	}

	return out.ColView(0), nil
}
