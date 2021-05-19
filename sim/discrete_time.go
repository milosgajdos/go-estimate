package sim

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Discrete is a basic model of a linear, discrete-time, dynamical system
type Discrete struct {
	System
}

// NewDiscrete creates a linear discrete-time model based on the control theory equations.
//
//  x[n+1] = A*x[n] + B*u[n] + E*z[n] (disturbances E not implemented yet)
//  y[n] = C*x[n] + D*u[n]
func NewDiscrete(A, B, C, D, E *mat.Dense) (*Discrete, error) {
	if A == nil {
		return nil, fmt.Errorf("system matrix must be defined for a model")
	}
	return &Discrete{System: System{A: A, B: B, C: C, D: D, E: E}}, nil
}

// Propagate propagates returns the next internal state x
// of a linear, continuous-time system given an input vector u and a
// disturbance input z. (wd is process noise, z not implemented yet)
func (ct *Discrete) Propagate(x, u, wd mat.Vector) (mat.Vector, error) {
	nx, nu, _, _ := ct.SystemDims()
	if u != nil && u.Len() != nu {
		return nil, fmt.Errorf("invalid input vector")
	}

	if x.Len() != nx {
		return nil, fmt.Errorf("invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(ct.A, x)
	if u != nil && ct.B != nil {
		outU := new(mat.Dense)
		outU.Mul(ct.B, u)

		out.Add(out, outU)
	}

	if wd != nil && wd.Len() == nx {
		out.Add(out, wd)
	}
	return out.ColView(0), nil
}
