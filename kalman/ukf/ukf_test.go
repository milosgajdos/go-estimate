package ukf

import (
	"fmt"
	"os"
	"testing"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/noise"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

type mockModel struct {
	A *mat.Dense
	B *mat.Dense
	C *mat.Dense
	D *mat.Dense
	Q filter.Noise
	R filter.Noise
}

func (m *mockModel) Propagate(x, u mat.Vector) (mat.Vector, error) {
	_in, _out := m.Dims()
	if u.Len() != _out {
		return nil, fmt.Errorf("Invalid input vector")
	}

	if x.Len() != _in {
		return nil, fmt.Errorf("Invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(m.A, x)

	outU := new(mat.Dense)
	outU.Mul(m.B, u)

	out.Add(out, outU)

	if m.Q != nil {
		out.Add(out, m.Q.Sample())
	}

	return out.ColView(0), nil
}

func (m *mockModel) Observe(x, u mat.Vector) (mat.Vector, error) {
	_in, _out := m.Dims()
	if u.Len() != _out {
		return nil, fmt.Errorf("Invalid input vector")
	}

	if x.Len() != _in {
		return nil, fmt.Errorf("Invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(m.C, x)

	outU := new(mat.Dense)
	outU.Mul(m.D, u)

	out.Add(out, outU)

	if m.R != nil {
		out.Add(out, m.R.Sample())
	}

	return out.ColView(0), nil
}

func (m *mockModel) Dims() (int, int) {
	_, in := m.A.Dims()
	out, _ := m.D.Dims()

	return in, out
}

func (m *mockModel) StateNoise() filter.Noise {
	return m.Q
}

func (m *mockModel) OutputNoise() filter.Noise {
	return m.R
}

type invalidModel struct{}

func (m *invalidModel) Propagate(x, u mat.Vector) (mat.Vector, error) {
	return new(mat.VecDense), nil
}

func (m *invalidModel) Observe(x, u mat.Vector) (mat.Vector, error) {
	return new(mat.VecDense), nil
}

func (m *invalidModel) Dims() (int, int) {
	return -10, 8
}

func (m *invalidModel) StateNoise() filter.Noise {
	return nil
}

func (m *invalidModel) OutputNoise() filter.Noise {
	return nil
}

type initCond struct {
	state mat.Vector
	cov   mat.Symmetric
}

func (c *initCond) State() mat.Vector {
	return c.state
}

func (c *initCond) Cov() mat.Symmetric {
	return c.cov
}

var (
	c        *Config
	ic       *initCond
	okModel  *mockModel
	badModel *invalidModel
	u        *mat.VecDense
	z        *mat.VecDense
	sNoise   filter.Noise
	oNoise   filter.Noise
)

func setup() {
	u = mat.NewVecDense(1, []float64{-1.0})
	z = mat.NewVecDense(1, []float64{-1.5})
	// initial condition
	var state mat.Vector = mat.NewVecDense(2, []float64{1.0, 1.0})
	var stateCov mat.Symmetric = mat.NewSymDense(2, []float64{1, 0, 0, 1})
	sNoise, _ = noise.NewGaussian([]float64{0, 0}, stateCov)
	var outCov mat.Symmetric = mat.NewSymDense(1, []float64{0.25})
	oNoise, _ = noise.NewGaussian([]float64{0}, outCov)

	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})
	Q := sNoise
	R := oNoise

	okModel = &mockModel{A, B, C, D, Q, R}
	badModel = &invalidModel{}

	c = &Config{
		Alpha: 0.75,
		Beta:  2.0,
		Kappa: 3.0,
	}

	ic = &initCond{
		state: state,
		cov:   stateCov,
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

	// invalid model
	f, err := New(badModel, ic, c)
	assert.Nil(f)
	assert.Error(err)
	// invalid config
	_alpha := c.Alpha
	c.Alpha = -10.0
	f, err = New(okModel, ic, c)
	assert.Nil(f)
	assert.Error(err)
	c.Alpha = _alpha
	// valid parameters
	f, err = New(okModel, ic, c)
	assert.NotNil(f)
	assert.NoError(err)
}

func TestCovariance(t *testing.T) {
	assert := assert.New(t)

	// valid parameters
	f, err := New(okModel, ic, c)
	assert.NotNil(f)
	assert.NoError(err)

	cov := f.Covariance()
	assert.NotNil(cov)
}

func TestGain(t *testing.T) {
	assert := assert.New(t)

	// valid parameters
	f, err := New(okModel, ic, c)
	assert.NotNil(f)
	assert.NoError(err)

	gain := f.Gain()
	assert.NotNil(gain)
}

func TestPredict(t *testing.T) {
	//assert := assert.New(t)
}
