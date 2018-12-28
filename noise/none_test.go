package noise

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewNone(t *testing.T) {
	assert := assert.New(t)

	e, err := NewNone()
	assert.NotNil(e)
	assert.NoError(err)
}

func TestNoneMeanCov(t *testing.T) {
	assert := assert.New(t)

	e, err := NewNone()
	assert.NotNil(e)
	assert.NoError(err)

	assert.True(e.Cov().(*mat.SymDense).IsZero())
	assert.Equal(0, len(e.Mean()))
}

func TestNoneSample(t *testing.T) {
	assert := assert.New(t)

	e, err := NewNone()
	assert.NotNil(e)
	assert.NoError(err)

	sample := e.Sample()
	assert.Equal(0, sample.(*mat.VecDense).Len())
}

func TestNoneReset(t *testing.T) {
	assert := assert.New(t)

	e, err := NewNone()
	assert.NotNil(e)
	assert.NoError(err)
}

func TestNoneString(t *testing.T) {
	assert := assert.New(t)

	str := `None{
Mean=[]
Cov=
}`

	e, err := NewNone()
	assert.NotNil(e)
	assert.NoError(err)
	assert.Equal(str, e.String())
}
