package bf

import (
	"os"
	"testing"

	filter "github.com/milosgajdos/go-estimate"
	"github.com/milosgajdos/go-estimate/noise"
	"github.com/milosgajdos/go-estimate/sim"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type invalidModel struct {
	filter.DiscreteModel
}

func (m *invalidModel) SystemDims() (nx, nu, ny, nz int) {
	return -10, 0, 8, 0
}

var (
	okModel  *sim.Discrete
	badModel *invalidModel
	ic       *sim.InitCond
	p        int
	u        *mat.VecDense
	z        *mat.VecDense
	q        filter.Noise
	r        filter.Noise
	errPDF   distmv.LogProber
)

func setup() {
	// PF parameters
	p = 10
	outCov := mat.NewSymDense(1, []float64{0.25})
	errPDF, _ = distmv.NewNormal([]float64{0}, outCov, nil)

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

	data := []float64{1.0, 1.0}
	x := mat.NewVecDense(2, data)
	est, err := f.Update(x, u, z)
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

func TestParticles(t *testing.T) {
	assert := assert.New(t)

	// create bootstrap filter
	f, err := New(okModel, ic, q, r, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)

	p := f.Particles()
	r, c := p.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			assert.InDelta(f.x.At(i, j), p.At(i, j), 0.001)
		}
	}
}

func TestWeights(t *testing.T) {
	assert := assert.New(t)

	// create bootstrap filter
	f, err := New(okModel, ic, q, r, p, errPDF)
	assert.NotNil(f)
	assert.NoError(err)

	weights := f.Weights()
	for i := range f.w {
		assert.InDelta(f.w[i], weights.At(i, 0), 0.001)
	}
}

func TestAlphaGauss(t *testing.T) {
	assert := assert.New(t)

	alpha := AlphaGauss(1, 2)
	assert.True(alpha > 0.0)
}
