package bootstrap

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type mockModel struct {
	A *mat.Dense
	B *mat.Dense
	C *mat.Dense
	D *mat.Dense
}

func mxFormat(m *mat.Dense) fmt.Formatter {
	return mat.Formatted(m, mat.Prefix(""), mat.Squeeze())
}

// Propagate propagates internal state x of falling ball to the next step
func (m *mockModel) Propagate(x, u mat.Matrix) (*mat.Dense, error) {
	if reflect.ValueOf(x).IsNil() {
		return nil, errors.New("MockPropagator Error")
	}

	out := new(mat.Dense)
	out.Mul(m.A, x)

	outU := new(mat.Dense)
	outU.Mul(m.B, u)

	out.Add(out, outU)

	return out, nil
}

func (m *mockModel) Observe(x, u mat.Matrix) (*mat.Dense, error) {
	if reflect.ValueOf(u).IsNil() {
		return nil, errors.New("MockObserver Error")
	}

	out := new(mat.Dense)
	out.Mul(m.C, x)

	outU := new(mat.Dense)
	outU.Mul(m.D, u)

	out.Add(out, outU)

	return out, nil
}

func (m *mockModel) Dims() (int, int) {
	_, aCols := m.A.Dims()
	dRows, _ := m.D.Dims()

	return aCols, dRows
}

var (
	pCount int
	config *Config
	start  *InitCond
	model  *mockModel
	u      *mat.Dense
)

func setup() {
	// BF parameters
	pCount = 10
	measCov := mat.NewSymDense(1, []float64{0.25})
	errOut, _ := distmv.NewNormal([]float64{0}, measCov, nil)

	u = mat.NewDense(1, 1, []float64{-1.0})
	// initial condition
	state := mat.NewDense(2, 1, []float64{1.0, 1.0})
	stateCov := mat.NewSymDense(2, []float64{1, 0, 0, 1})

	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	model := &mockModel{A, B, C, D}

	config = &Config{
		Model:         model,
		ParticleCount: pCount,
		Err:           errOut,
	}

	start = &InitCond{
		State: state,
		Cov:   stateCov,
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

func TestNewFilterInit(t *testing.T) {
	assert := assert.New(t)

	// invalid count
	config.ParticleCount = -10
	f, err := NewFilter(config)
	assert.Nil(f)
	assert.Error(err)
	// valid config should succeed
	config.ParticleCount = pCount
	f, err = NewFilter(config)
	assert.NotNil(f)
	assert.NoError(err)
	// initialize filter
	err = f.Init(start)
	assert.NoError(err)
}

func TestRun(t *testing.T) {
	assert := assert.New(t)

	f, err := NewFilter(config)
	assert.NotNil(f)
	assert.NoError(err)

	err = f.Init(start)
	assert.NoError(err)

	data := []float64{1.0, 1.0}
	x := mat.NewDense(2, 1, data)
	// for simplicity:
	// - we set measurement to u
	xNew, err := f.Run(x, u, u)
	assert.NoError(err)
	assert.NotNil(xNew)

	// we simulate propagator error by setting input to nil
	xNew, err = f.Run(nil, u, u)
	assert.Nil(xNew)
	assert.Error(err)
}

func TestResample(t *testing.T) {
	assert := assert.New(t)

	f, err := NewFilter(config)
	assert.NotNil(f)
	assert.NoError(err)

	err = f.Init(start)
	assert.NoError(err)

	alpha := 10.0
	err = f.Resample(alpha)
	assert.NoError(err)
}
