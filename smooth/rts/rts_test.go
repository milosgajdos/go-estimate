package rts

import (
	"os"
	"testing"

	"github.com/milosgajdos83/go-filter/sim"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var (
	ic *sim.InitCond
)

func setup() {
	// initial condition
	initState := mat.NewVecDense(2, []float64{1.0, 3.0})
	initCov := mat.NewSymDense(2, []float64{0.25, 0, 0, 0.25})
	ic = sim.NewInitCond(initState, initCov)
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
