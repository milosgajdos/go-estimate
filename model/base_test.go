package model

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var (
	x, u, z, q, r *mat.VecDense
	A, B, C, D    *mat.Dense
)

func setup() {
	x = mat.NewVecDense(2, []float64{0.5, 0.6})
	u = mat.NewVecDense(1, []float64{-1.0})
	z = mat.NewVecDense(1, []float64{-1.5})

	// state and output noise
	q = mat.NewVecDense(2, nil)
	r = mat.NewVecDense(1, nil)

	A = mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B = mat.NewDense(2, 1, []float64{0.5, 1.0})
	C = mat.NewDense(1, 2, []float64{1.0, 0.0})
	D = mat.NewDense(1, 1, []float64{0.0})
}

func TestMain(m *testing.M) {
	// set up tests
	setup()
	// run the tests
	retCode := m.Run()
	// call with result of m.Run()
	os.Exit(retCode)
}

func TestInitCond(t *testing.T) {
	assert := assert.New(t)

	state := mat.NewVecDense(2, []float64{1.0, 3.0})
	cov := mat.NewSymDense(2, []float64{0.25, 0, 0, 0.25})

	ic := NewInitCond(state, cov)

	s := ic.State()
	for i := 0; i < state.Len(); i++ {
		assert.Equal(state.AtVec(i), s.AtVec(i))
	}

	c := ic.Cov()
	for i := 0; i < cov.Symmetric(); i++ {
		for j := 0; j < cov.Symmetric(); j++ {
			assert.Equal(cov.At(i, j), c.At(i, j))
		}
	}
}

func TestBase(t *testing.T) {
	assert := assert.New(t)

	f, err := NewBase(A, B, C, D)
	assert.NotNil(f)
	assert.NoError(err)
}

func TestBasePropagate(t *testing.T) {
	assert := assert.New(t)

	f, err := NewBase(A, B, C, D)
	assert.NotNil(f)
	assert.NoError(err)

	v, err := f.Propagate(x, u, q)
	assert.NotNil(v)
	assert.NoError(err)

	_u := mat.NewVecDense(10, nil)
	v, err = f.Propagate(x, _u, q)
	assert.Nil(v)
	assert.Error(err)

	_x := mat.NewVecDense(10, nil)
	v, err = f.Propagate(_x, u, q)
	assert.Nil(v)
	assert.Error(err)

	v, err = f.Propagate(x, u, nil)
	assert.NotNil(v)
	assert.NoError(err)
}

func TestBaseObserve(t *testing.T) {
	assert := assert.New(t)

	f, err := NewBase(A, B, C, D)
	assert.NotNil(f)
	assert.NoError(err)

	v, err := f.Observe(x, u, r)
	assert.NotNil(v)
	assert.NoError(err)

	_u := mat.NewVecDense(10, nil)
	v, err = f.Observe(x, _u, r)
	assert.Nil(v)
	assert.Error(err)

	_x := mat.NewVecDense(10, nil)
	v, err = f.Observe(_x, u, r)
	assert.Nil(v)
	assert.Error(err)

	v, err = f.Observe(x, u, nil)
	assert.NotNil(v)
	assert.NoError(err)
}

func TestBaseSystemMatrices(t *testing.T) {
	assert := assert.New(t)

	f, err := NewBase(A, B, C, D)
	assert.NotNil(f)
	assert.NoError(err)

	m := f.StateMatrix()
	assert.True(mat.EqualApprox(m, A, 0.001))

	m = f.StateCtlMatrix()
	assert.True(mat.EqualApprox(m, B, 0.001))

	m = f.OutputMatrix()
	assert.True(mat.EqualApprox(m, C, 0.001))

	m = f.OutputCtlMatrix()
	assert.True(mat.EqualApprox(m, D, 0.001))
}

func TestBaseDims(t *testing.T) {
	assert := assert.New(t)

	f, err := NewBase(A, B, C, D)
	assert.NotNil(f)
	assert.NoError(err)

	in, out := f.Dims()
	_, _in := A.Dims()
	_out, _ := D.Dims()
	assert.Equal(_in, in)
	assert.Equal(_out, out)
}
