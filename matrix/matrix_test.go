package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestRowColSums(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	rowSums := []float64{4.6, 11.2, 18.9}
	colSums := []float64{14.6, 20.1}
	delta := 0.001

	m := mat.NewDense(3, 2, data)
	assert.NotNil(m)

	// check rows
	resRows := RowSums(m)
	assert.NotNil(resRows)
	assert.InDeltaSlice(rowSums, resRows, delta)
	// check cols
	resCols := ColSums(m)
	assert.NotNil(resCols)
	assert.InDeltaSlice(colSums, resCols, delta)
	// should panic
	assert.Panics(func() { RowSums(nil) })
	assert.Panics(func() { ColSums(nil) })
}
