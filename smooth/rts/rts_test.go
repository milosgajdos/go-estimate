package rts

import (
	"os"
	"testing"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/go-filter/sim"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var (
	ic *sim.InitCond
	ex []filter.Estimate
	mx []mat.Matrix
	n  int
)

func setup() {
	// initial condition
	initState := mat.NewVecDense(2, []float64{1.0, 3.0})
	initCov := mat.NewSymDense(2, []float64{0.25, 0, 0, 0.25})
	ic = sim.NewInitCond(initState, initCov)

	n = 2
	// generate some estimates
	for i := 0; i < 5; i++ {
		e, _ := estimate.NewBaseWithCov(
			mat.NewVecDense(n, []float64{1.0, 3.0}),
			mat.NewSymDense(n, []float64{0.25, 0, 0, 0.25}))
		mx = append(mx, mat.NewDense(2, 2, []float64{1.0, 1.0, 1.0, 1.0}))
		ex = append(ex, e)
	}
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

	s, err := New(ic)
	assert.NotNil(s)
	assert.NoError(err)
}

func TestRTSSmooth(t *testing.T) {
	assert := assert.New(t)

	s, err := New(ic)
	assert.NotNil(s)
	assert.NoError(err)

	sx, err := s.Smooth(ex[0:1], ex, mx)
	assert.Nil(sx)
	assert.Error(err)

	sx, err = s.Smooth(ex, ex, mx)
	assert.NotNil(sx)
	assert.NoError(err)
}
