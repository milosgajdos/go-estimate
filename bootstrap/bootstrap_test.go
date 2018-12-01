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
	pCount int
	config *Config
	start  *InitCond
)

func setup() {
	// BF parameters
	pCount = 10
	err, _ := distmv.NewNormal([]float64{0, 0}, mat.NewSymDense(2, []float64{1, 0, 0, 1}), nil)

	config = &Config{
		Propagator:    &mockPropagator{},
		Observer:      &mockObserver{},
		ParticleCount: pCount,
		Err:           err,
	}
	// initial condition
	stateVals := []float64{1.0, 1.0}
	state := mat.NewDense(2, 1, stateVals)
	vals := []float64{1.0, 0.0, 0.0, 1.0}
	cov := mat.NewDense(2, 2, vals)

	start = &InitCond{
		State: state,
		Cov:   cov,
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

func TestNewInit(t *testing.T) {
	assert := assert.New(t)

	// invalid count
	config.ParticleCount = -10
	f, err := New(config)
	assert.Nil(f)
	assert.Error(err)
	// valid config should succeed
	config.ParticleCount = pCount
	f, err = New(config)
	assert.NotNil(f)
	assert.NoError(err)
	// initialize filter
	err = f.Init(start)
	assert.NoError(err)
}

func TestRun(t *testing.T) {
	assert := assert.New(t)

	f, err := New(config)
	assert.NotNil(f)
	assert.NoError(err)

	err = f.Init(start)
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

func TestResample(t *testing.T) {
	assert := assert.New(t)

	f, err := New(config)
	assert.NotNil(f)
	assert.NoError(err)

	err = f.Init(start)
	assert.NoError(err)

	alpha := 10.0
	err = f.Resample(alpha)
	assert.NoError(err)
}
