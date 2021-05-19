package ekf

import (
	"os"
	"testing"

	filter "github.com/milosgajdos/go-estimate"
	"github.com/milosgajdos/go-estimate/noise"
	"github.com/milosgajdos/go-estimate/sim"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

type invalidModel struct {
	filter.DiscreteModel
}

func (m *invalidModel) SystemDims() (nx, nu, ny, nz int) {
	return -10, 0, 8, 0 // a system may have 0 inputs, this is not "invalid". Negative dimension is invalid
}

var (
	okModel  *sim.Discrete
	badModel *invalidModel
	ic       *sim.InitCond
	q        filter.Noise
	r        filter.Noise
	u        *mat.VecDense
	z        *mat.VecDense
)

func setup() {
	u = mat.NewVecDense(1, []float64{-1.0})
	z = mat.NewVecDense(1, []float64{-1.5})

	// initial condition
	initState := mat.NewVecDense(2, []float64{1.0, 3.0})
	initCov := mat.NewSymDense(2, []float64{0.25, 0, 0, 0.25})
	ic = sim.NewInitCond(initState, initCov)

	// state and output noise
	q, _ = noise.NewGaussian([]float64{0, 0}, initCov)
	r, _ = noise.NewGaussian([]float64{0}, mat.NewSymDense(1, []float64{0.25}))

	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	okModel = &sim.Discrete{System: sim.System{A: A, B: B, C: C, D: D}}
	badModel = &invalidModel{okModel}
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// call with result of m.Run()
	os.Exit(retCode)
}

func TestEKFNew(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	// invalid model: incorrect dimensions
	f, err = New(badModel, ic, q, r)
	assert.Nil(f)
	assert.Error(err)

	// invalid state noise dimension
	_q := q
	q, _ = noise.NewZero(20)
	f, err = New(okModel, ic, q, r)
	assert.Nil(f)
	assert.Error(err)
	q = _q

	// invalid output noise dimension
	_r := r
	r, _ = noise.NewZero(20)
	f, err = New(okModel, ic, q, r)
	assert.Nil(f)
	assert.Error(err)
	r = _r

	// zero [state and output] noise
	f, err = New(okModel, ic, nil, nil)
	assert.NotNil(f)
	assert.NoError(err)
}

func TestEKFPredict(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	x := mat.VecDenseCopyOf(ic.State())
	est, err := f.Predict(x, u)
	assert.NotNil(est)
	assert.NoError(err)

	// invalid input vector
	_u := mat.NewVecDense(3, nil)
	est, err = f.Predict(x, _u)
	assert.Nil(est)
	assert.Error(err)
}

func TestEKFUpdate(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	x := mat.VecDenseCopyOf(ic.State())
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

func TestEKFRun(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	x := mat.VecDenseCopyOf(ic.State())
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

func TestEKFModel(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	m := f.Model()
	assert.NotNil(m)
}

func TestEKFNoise(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	sn := f.StateNoise()
	assert.NotNil(sn)

	on := f.OutputNoise()
	assert.NotNil(on)
}

func TestEKFCov(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	cov := f.Cov()
	assert.NotNil(cov)

	err = f.SetCov(nil)
	assert.Error(err)

	err = f.SetCov(mat.NewSymDense(30, nil))
	assert.Error(err)

	err = f.SetCov(mat.NewSymDense(f.p.Symmetric(), nil))
	assert.NoError(err)
}

func TestEKFGain(t *testing.T) {
	assert := assert.New(t)

	f, err := New(okModel, ic, q, r)
	assert.NotNil(f)
	assert.NoError(err)

	gain := f.Gain()
	assert.NotNil(gain)
}
