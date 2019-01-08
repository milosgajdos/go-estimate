package ekf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewIterEKF(t *testing.T) {
	assert := assert.New(t)

	f, err := NewIter(okModel, ic, q, r, 5)
	assert.NotNil(f)
	assert.NoError(err)

	// invalid number of iterations
	f, err = NewIter(badModel, ic, q, r, -5)
	assert.Nil(f)
	assert.Error(err)

	// invalid model: incorrect dimensions
	f, err = NewIter(badModel, ic, q, r, 5)
	assert.Nil(f)
	assert.Error(err)
}

func TestIEKFUpdate(t *testing.T) {
	assert := assert.New(t)

	f, err := NewIter(okModel, ic, q, r, 3)
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

func TestIEKFRun(t *testing.T) {
	assert := assert.New(t)

	f, err := NewIter(okModel, ic, q, r, 3)
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
