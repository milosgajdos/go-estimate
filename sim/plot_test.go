package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNew2DPlot(t *testing.T) {
	assert := assert.New(t)

	model := mat.NewDense(3, 2, nil)
	measure := mat.NewDense(3, 2, nil)
	filter := mat.NewDense(3, 2, nil)

	plt, err := New2DPlot(model, measure, filter)
	assert.NotNil(plt)
	assert.NoError(err)

	plt, err = New2DPlot(nil, nil, nil)
	assert.Nil(plt)
	assert.Error(err)

	plt, err = New2DPlot(mat.NewDense(3, 1, nil), measure, filter)
	assert.Nil(plt)
	assert.Error(err)

}
