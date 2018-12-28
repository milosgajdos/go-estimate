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
}

func (m *mockModel) Propagate(x, u, q mat.Vector) (mat.Vector, error) {
	_in, _out := m.Dims()
	if u.Len() != _out {
		return nil, fmt.Errorf("Invalid input vector")
	}

	if x.Len() != _in {
		return nil, fmt.Errorf("Invalid state vector")
	}

	if q != nil && q.Len() != _in {
		return nil, fmt.Errorf("Invalid state noise")
	}

	out := new(mat.Dense)
	out.Mul(m.A, x)

	outU := new(mat.Dense)
	outU.Mul(m.B, u)

	out.Add(out, outU)

	if q != nil {
		out.Add(out, q)
	}

	return out.ColView(0), nil
}

func (m *mockModel) Observe(x, u, r mat.Vector) (mat.Vector, error) {
	_in, _out := m.Dims()
	if u.Len() != _out {
		return nil, fmt.Errorf("Invalid input vector")
	}

	if x.Len() != _in {
		return nil, fmt.Errorf("Invalid state vector")
	}

	if r != nil && r.Len() != _out {
		return nil, fmt.Errorf("Invalid output noise")
	}

	out := new(mat.Dense)
	out.Mul(m.C, x)

	outU := new(mat.Dense)
	outU.Mul(m.D, u)

	out.Add(out, outU)

	if r != nil {
		out.Add(out, r)
	}

	return out.ColView(0), nil
}

func (m *mockModel) Dims() (int, int) {
	_, in := m.A.Dims()
	out, _ := m.D.Dims()

	return in, out
}

type invalidModel struct{}

func (m *invalidModel) Propagate(x, u, q mat.Vector) (mat.Vector, error) {
	return new(mat.VecDense), nil
}

func (m *invalidModel) Observe(x, u, r mat.Vector) (mat.Vector, error) {
	return new(mat.VecDense), nil
}

func (m *invalidModel) Dims() (int, int) {
	return -10, 8
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
	okModel  *mockModel
	badModel *invalidModel
	ic       *initCond
	q        filter.Noise
	r        filter.Noise
	c        *Config
	u        *mat.VecDense
	z        *mat.VecDense
)

func setup() {
	u = mat.NewVecDense(1, []float64{-1.0})
	z = mat.NewVecDense(1, []float64{-1.5})

	// initial condition
	initState := mat.NewVecDense(2, []float64{1.0, 3.0})
	initCov := mat.NewSymDense(2, []float64{0.25, 0, 0, 0.25})

	// state and output noise
	q, _ = noise.NewGaussian([]float64{0, 0}, initCov)
	r, _ = noise.NewGaussian([]float64{0}, mat.NewSymDense(1, []float64{0.25}))

	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	okModel = &mockModel{A, B, C, D}
	badModel = &invalidModel{}

	c = &Config{
		Alpha: 0.75,
		Beta:  2.0,
		Kappa: 3.0,
	}

	ic = &initCond{
		state: initState,
		cov:   initCov,
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

func TestUKFNew(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, c)
	assert.NotNil(f)
	assert.NoError(err)

	// invalid model: incorrect dimensions
	f, err = New(badModel, ic, q, r, c)
	assert.Nil(f)
	assert.Error(err)

	// invalid config
	_alpha := c.Alpha
	c.Alpha = -10.0
	f, err = New(okModel, ic, q, r, c)
	assert.Nil(f)
	assert.Error(err)
	c.Alpha = _alpha

	// invalid state noise dimension
	_q := q
	q, _ = noise.NewZero(20)
	f, err = New(okModel, ic, q, r, c)
	assert.Nil(f)
	assert.Error(err)
	q = _q

	// invalid output noise dimension
	_r := r
	r, _ = noise.NewZero(20)
	f, err = New(okModel, ic, q, r, c)
	assert.Nil(f)
	assert.Error(err)
	r = _r

	// zero [state and output] noise
	f, err = New(okModel, ic, nil, nil, c)
	assert.NotNil(f)
	assert.NoError(err)
}

func TestUKFGenSigmaPoints(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, c)
	//f, err := New(okModel, ic, nil, r, c)
	assert.NotNil(f)
	assert.NoError(err)

	x := mat.VecDenseCopyOf(ic.state)
	sp, err := f.GenSigmaPoints(x)
	assert.NotNil(sp)
	assert.NoError(err)
}

func TestUKFPredict(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, c)
	assert.NotNil(f)
	assert.NoError(err)

	x := mat.VecDenseCopyOf(ic.state)
	est, err := f.Predict(x, u)
	assert.NotNil(est)
	assert.NoError(err)

	// invalid input vector
	_u := mat.NewVecDense(3, nil)
	est, err = f.Predict(x, _u)
	assert.Nil(est)
	assert.Error(err)

	// sigma point propagation error
	_x := mat.NewDense(5, 2, nil)
	sp := &SigmaPoints{X: _x}
	spp, err := f.propagateSigmaPoints(sp, _u)
	assert.Nil(spp)
	assert.Error(err)
}

func TestUKFUpdate(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, c)
	assert.NotNil(f)
	assert.NoError(err)

	x := mat.VecDenseCopyOf(ic.state)
	est, err := f.Update(x, u, z)
	assert.NotNil(est)
	assert.NoError(err)

	// invalid input vector
	_u := mat.NewVecDense(3, nil)
	est, err = f.Update(x, _u, z)
	assert.Nil(est)
	assert.Error(err)

	// invalid measurement vector
	_z := mat.NewVecDense(3, nil)
	est, err = f.Update(x, u, _z)
	assert.Nil(est)
	assert.Error(err)
}

func TestUKFRun(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, c)
	assert.NotNil(f)
	assert.NoError(err)

	x := mat.VecDenseCopyOf(ic.state)
	est, err := f.Run(x, u, z)
	assert.NotNil(est)
	assert.NoError(err)

	// invalid input vector
	_u := mat.NewVecDense(3, nil)
	est, err = f.Run(x, _u, z)
	assert.Nil(est)
	assert.Error(err)

	// invalid measurement vector
	_z := mat.NewVecDense(3, nil)
	est, err = f.Run(x, u, _z)
	assert.Nil(est)
	assert.Error(err)
}

func TestUKFCovariance(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, c)
	assert.NotNil(f)
	assert.NoError(err)

	cov := f.Covariance()
	assert.NotNil(cov)
}

func TestUKFGain(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, c)
	assert.NotNil(f)
	assert.NoError(err)

	gain := f.Gain()
	assert.NotNil(gain)
}
