package sim

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var (
	x, u, z, q, r *mat.VecDense
	A, B, C, D, E *mat.Dense
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
	E = mat.NewDense(2, 1, []float64{1.0, 0})
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

	f, err := NewDiscrete(A, B, C, D, E)
	assert.NotNil(f)
	assert.NoError(err)
}

func TestDiscretePropagate(t *testing.T) {
	assert := assert.New(t)

	f, err := NewDiscrete(A, B, C, D, E)
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

func TestDiscreteObserve(t *testing.T) {
	assert := assert.New(t)

	f, err := NewDiscrete(A, B, C, D, E)
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

func TestSystemMatrices(t *testing.T) {
	assert := assert.New(t)
	f := System{A, B, C, D, E}
	assert.NotNil(f)

	m := f.SystemMatrix()
	assert.True(mat.EqualApprox(m, A, 0.001))

	m = f.ControlMatrix()
	assert.True(mat.EqualApprox(m, B, 0.001))

	m = f.OutputMatrix()
	assert.True(mat.EqualApprox(m, C, 0.001))

	m = f.FeedForwardMatrix()
	assert.True(mat.EqualApprox(m, D, 0.001))
}

func TestSystemDims(t *testing.T) {
	assert := assert.New(t)
	f := System{A, B, C, D, E}
	assert.NotNil(f)

	nx, nu, ny, nz := f.SystemDims()
	r, c := A.Dims()
	assert.Equal(nx, r) // A is square [n,n]
	assert.Equal(nx, c)
	r, c = B.Dims()
	assert.Equal(nx, r) // B [n,p]
	assert.Equal(nu, c)
	r, c = C.Dims()
	assert.Equal(ny, r) // C [q,n]
	assert.Equal(nx, c)
	r, c = D.Dims()
	assert.Equal(ny, r) // D [q,p]
	assert.Equal(nu, c)
	r, c = E.Dims()
	assert.Equal(nx, r) // E [n,r]
	assert.Equal(nz, c)
}
