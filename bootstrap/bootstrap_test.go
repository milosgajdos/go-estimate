package bootstrap

import (
	"fmt"
	"os"
	"testing"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/noise"
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

func (m *mockModel) Propagate(x, u, q mat.Vector) (mat.Vector, error) {
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
	p        int
	u        *mat.VecDense
	z        *mat.VecDense
	q        filter.Noise
	r        filter.Noise
	errPDF   distmv.LogProber
)

func setup() {
	// BF parameters
	p = 10
	outCov := mat.NewSymDense(1, []float64{0.25})
	errPDF, _ = distmv.NewNormal([]float64{0}, outCov, nil)

	u = mat.NewVecDense(1, []float64{-1.0})
	z = mat.NewVecDense(1, []float64{-1.5})

	// initial condition
	var state mat.Vector = mat.NewVecDense(2, []float64{1.0, 1.0})
	var stateCov mat.Symmetric = mat.NewSymDense(2, []float64{1, 0, 0, 1})

	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	okModel = &mockModel{A, B, C, D}
	badModel = &invalidModel{}

	q, _ = noise.NewZero(state.Len())
	r, _ = noise.NewZero(z.Len())

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

	// invalid particle count
	f, err := New(okModel, ic, q, r, -10, errPDF)
	assert.Nil(f)
	assert.Error(err)

	// invalid model
	f, err = New(badModel, ic, q, r, p, errPDF)
	assert.Nil(f)
	assert.Error(err)

	// invalid state noise
	_q := q
	q, _ = noise.NewZero(20)
	f, err = New(okModel, ic, q, r, p, errPDF)
	assert.Nil(f)
	assert.Error(err)
	q = _q

	// invalid output noise
	_r := r
	r, _ = noise.NewZero(20)
	f, err = New(okModel, ic, q, r, p, errPDF)
	assert.Nil(f)
	assert.Error(err)
	r = _r

	// nil state and output noise
	f, err = New(okModel, ic, nil, nil, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)

	// valid parameters
	f, err = New(okModel, ic, q, r, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)
}

func TestPredict(t *testing.T) {
	assert := assert.New(t)

	// create bootstrap filter
	f, err := New(okModel, ic, q, r, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)

	data := []float64{1.0, 1.0}
	x := mat.NewVecDense(2, data)
	_u := mat.NewVecDense(3, nil)

	// state propagation error
	est, err := f.Predict(x, _u)
	assert.Nil(est)
	assert.Error(err)

	// particle propagation error
	_x := mat.NewDense(5, 5, nil)
	particles := f.x
	f.x = _x
	est, err = f.Predict(x, u)
	assert.Nil(est)
	assert.Error(err)

	f.x = particles
	est, err = f.Predict(x, u)
	assert.NotNil(est)
	assert.NoError(err)
}

func TestUpdate(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)

	// incorrect state dimensions
	_x := mat.NewVecDense(3, nil)
	est, err := f.Update(_x, u, z)
	assert.Nil(est)
	assert.Error(err)

	data := []float64{1.0, 1.0}
	x := mat.NewVecDense(2, data)
	est, err = f.Update(x, u, z)
	assert.NotNil(est)
	assert.NoError(err)
}

func TestRun(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 1.0}
	x := mat.NewVecDense(2, data)

	f, err := New(okModel, ic, q, r, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)

	// Predict error
	_u := mat.NewVecDense(3, nil)
	est, err := f.Run(x, _u, z)
	assert.Nil(est)
	assert.Error(err)

	_z := mat.NewVecDense(3, nil)
	est, err = f.Run(x, u, _z)
	assert.Nil(est)
	assert.Error(err)

	est, err = f.Run(x, u, z)
	assert.NotNil(est)
	assert.NoError(err)
}

func TestResample(t *testing.T) {
	assert := assert.New(t)

	// create bootstrap filter
	f, err := New(okModel, ic, q, r, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)

	var _w []float64
	weights := f.w
	f.w = _w
	err = f.Resample(0.0)
	assert.Error(err)
	f.w = weights

	err = f.Resample(5.0)
	assert.NoError(err)

	err = f.Resample(0.0)
	assert.NoError(err)
}

func TestAlphaGauss(t *testing.T) {
	assert := assert.New(t)

	alpha := AlphaGauss(1, 2)
	assert.True(alpha > 0.0)
}
