package rts

import (
	"os"
	"testing"

	filter "github.com/milosgajdos/go-estimate"
	"github.com/milosgajdos/go-estimate/estimate"
	"github.com/milosgajdos/go-estimate/noise"
	"github.com/milosgajdos/go-estimate/sim"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

type invalidModel struct {
	filter.DiscreteControlSystem
	r int
	c int
}

func (m *invalidModel) SystemDims() (nx, nu, ny, nz int) {
	return m.r, 0, m.c, 0
}

var (
	okModel  *sim.Discrete
	badModel *invalidModel
	ic       *sim.InitCond
	q        filter.Noise
	ex       []filter.Estimate
	ux       []mat.Vector
	n        int
)

func setup() {
	// initial condition
	initState := mat.NewVecDense(2, []float64{1.0, 3.0})
	initCov := mat.NewSymDense(2, []float64{0.25, 0, 0, 0.25})
	ic = sim.NewInitCond(initState, initCov)

	// state and output noise
	q, _ = noise.NewGaussian([]float64{0, 0}, initCov)

	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	n = 2
	// generate some estimates
	for i := 0; i < 5; i++ {
		e, _ := estimate.NewBaseWithCov(
			mat.NewVecDense(n, []float64{1.0, 3.0}),
			mat.NewSymDense(n, []float64{0.25, 0, 0, 0.25}))
		u := mat.NewVecDense(1, []float64{-1.0})
		ex = append(ex, e)
		ux = append(ux, u)
	}

	okModel = &sim.Discrete{System: sim.System{A: A, B: B, C: C, D: D}}
	badModel = &invalidModel{DiscreteControlSystem: okModel, r: 10, c: 10}
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// call with result of m.Run()
	os.Exit(retCode)
}

func TestNewRTS(t *testing.T) {
	assert := assert.New(t)

	s, err := New(okModel, ic, q)
	assert.NotNil(s)
	assert.NoError(err)

	// nil noise
	s, err = New(okModel, ic, nil)
	assert.NotNil(s)
	assert.NoError(err)

	// invalid model: negative dimensions
	badModel.r, badModel.c = -10, 20
	s, err = New(badModel, ic, q)
	assert.Nil(s)
	assert.Error(err)

	// invalid state noise dimension
	_q := q
	q, _ = noise.NewZero(20)
	s, err = New(okModel, ic, q)
	assert.Nil(s)
	assert.Error(err)
	q = _q
}

func TestRTSSmooth(t *testing.T) {
	assert := assert.New(t)

	s, err := New(okModel, ic, q)
	assert.NotNil(s)
	assert.NoError(err)

	sx, err := s.Smooth(nil, ux)
	assert.Nil(sx)
	assert.Error(err)

	sx, err = s.Smooth(ex, ux[0:1])
	assert.Nil(sx)
	assert.Error(err)

	sx, err = s.Smooth(ex, ux)
	assert.NotNil(sx)
	assert.NoError(err)
}
