package sim

import (
	"fmt"

	"github.com/milosgajdos/matrix"
	"gonum.org/v1/gonum/mat"
)

// Continuous is a basic model of a linear, continuous-time, dynamical system
type Continuous struct {
	System
}

// NewContinuous creates a linear continuous-time model based on the control theory equations
// which is advanced by timestep dt.
//
//  dx/dt = A*x + B*u + E*z (disturbances E not implemented yet)
//  y = C*x + D*u
func NewContinuous(A, B, C, D, E *mat.Dense) (*Continuous, error) {
	if A == nil {
		return nil, fmt.Errorf("system matrix must be defined for a model")
	}
	sys := newSystem(A, B, C, D, E)
	return &Continuous{System: sys}, nil
}

// ToDiscrete creates a discrete-time model from a continuous time model
// using Ts as the sampling time.
//
// It is calculated using Euler's method, an approximation valid for small timesteps.
func (ct *Continuous) ToDiscrete(Ts float64) (*Discrete, error) {
	nx, _, _, _ := ct.SystemDims()
	dsys := newSystem(ct.A, ct.B, ct.C, ct.D, ct.E)
	// continuous -> discrete time conversion
	// See Discrete-Time Control Systems by Katsuhiko Ogata
	// Eq. (5-73) p. 315  Second Edition (Spanish)
	dsys.A.Scale(Ts, dsys.A)
	dsys.A.Exp(dsys.A)

	// shorthand name for discrete B matrix
	Bd := dsys.B
	Aaux := mat.NewDense(nx, nx, nil)
	// Given A is not singular, the following is valid
	// Bd(Ts) = (exp(A*Ts) - I)*inv(A)*B  Eq. (5-74 bis) Ogata
	eye, _ := matrix.NewDenseValIdentity(nx, 1.0)

	Aaux.Sub(dsys.A, eye)
	Ainv := mat.NewDense(nx, nx, nil)
	err := Ainv.Inverse(ct.A)
	if err == nil {
		Aaux.Mul(Aaux, Ainv)
		// Store subtraction result in Bd
		Bd.Mul(Aaux, ct.B)
		return &Discrete{dsys}, nil
	}

	Asum := Ainv        // change identifier to not confuse
	Asum.Scale(0, Asum) // reset data
	// if A matrix is singular we integrate with closed form
	// from 0 to Ts
	// Bd = integrate( exp(A*t)dt, 0, Ts ) * B   Eq. (5-74) Ogata
	const n = 100 // TODO parametrize C2D settings
	dt := Ts / float64(n-1)
	for i := 0; i < n; i++ {
		Aaux.Scale(dt*float64(i), ct.A)
		Aaux.Exp(Aaux)
		Aaux.Scale(dt, Aaux)
		Asum.Add(Asum, Aaux)
	}
	Bd.Mul(Asum, ct.B)
	return &Discrete{dsys}, nil
}

// Propagate propagates returns the next internal state x
// of a linear, continuous-time system given an input vector u and a
// disturbance input z. (wd is process noise, z not implemented yet). It propagates
// the solution by a timestep `dt`.
func (ct *Continuous) Propagate(x, u, wd mat.Vector, dt float64) (mat.Vector, error) {
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

	if wd != nil && wd.Len() == nx { // TODO change _nx to _nz when switching to z and disturbance matrix implementation
		// outZ := new(mat.Dense) // TODO add E disturbance matrix
		// outZ.Mul(b.E, z)
		// out.Add(out, outZ)
		out.Add(out, wd)
	}
	// integrate the first order derivatives calculated: dx/dt = A*x + B*u + wd
	out.Scale(dt, out)
	out.Add(x, out)
	return out.ColView(0), nil
}
