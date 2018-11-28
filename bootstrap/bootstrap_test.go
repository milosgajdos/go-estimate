package bootstrap

import (
	"errors"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type mockPropagator struct{}

func (p *mockPropagator) Propagate(x, u *mat.Dense) (*mat.Dense, error) {
	if x == nil {
		return nil, errors.New("MockPropagator Error")
	}

	return x, nil
}

type mockObserver struct{}

func (o *mockObserver) Observe(x, u *mat.Dense) (*mat.Dense, error) {
	if u == nil {
		return nil, errors.New("MockObserver Error")
	}

	return x, nil
}

var (
	pCount   int
	alpha    float64
	config   *Config
	initCond *InitCond
)

func setup() {
	// BF parameters
	pCount = 10
	alpha = 10.0
	errDist, _ := distmv.NewNormal([]float64{0, 0}, mat.NewSymDense(2, []float64{1, 0, 0, 1}), nil)
	covVals := []float64{1.0, 0.0, 0.0, 1.0}
	cov := mat.NewDense(2, 2, covVals)
	// initial condition
	stateVals := []float64{1.0, 1.0}
	state := mat.NewDense(2, 1, stateVals)
	initCond = &InitCond{State: state}

	config = &Config{
		Propagator:    &mockPropagator{},
		Observer:      &mockObserver{},
		ParticleCount: pCount,
		Alpha:         alpha,
		Cov:           cov,
		ErrDist:       errDist,
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

func TestNew(t *testing.T) {
	assert := assert.New(t)

	// invalid count
	config.ParticleCount = -10
	f, err := New(config, initCond)
	assert.Nil(f)
	assert.Error(err)

	config.ParticleCount = pCount
	f, err = New(config, initCond)
	assert.NotNil(f)
	assert.NoError(err)

	// test negative alpha
	_alpha := config.Alpha
	config.Alpha = -10
	config.ParticleCount = pCount
	f, err = New(config, initCond)
	assert.NotNil(f)
	assert.NoError(err)
	// reset Alpha back to original value
	config.Alpha = _alpha
}

func TestRun(t *testing.T) {
	assert := assert.New(t)

	f, err := New(config, initCond)
	assert.NotNil(f)
	assert.NoError(err)

	data := []float64{1.0, 1.0}
	x := mat.NewDense(2, 1, data)
	// for simplicity:
	// - we set system input u to x
	// - we consider output the same as x
	xNew, err := f.Run(x, x, x)
	assert.NoError(err)
	assert.NotNil(xNew)

	// we simulate propagator error by setting filter particles to nil
	_x := f.x
	f.x = nil
	xNew, err = f.Run(x, nil, x)
	assert.Nil(xNew)
	assert.Error(err)

	// we simulate observer error by setting input to nil
	f.x = _x
	xNew, err = f.Run(x, nil, nil)
	assert.Nil(xNew)
	assert.Error(err)
}
