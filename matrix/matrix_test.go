package matrix

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestFormat(t *testing.T) {
	assert := assert.New(t)

	out := `⎡1.2  3.4⎤
⎣4.5  6.7⎦`
	data := []float64{1.2, 3.4, 4.5, 6.7}
	m := mat.NewDense(2, 2, data)
	assert.NotNil(m)

	format := Format(m)
	tstOut := fmt.Sprintf("%v", format)
	assert.Equal(out, tstOut)
}

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

func TestRowsColsMean(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.2, 3.4, 4.5, 6.7, 8.9, 10.0}
	mRow := []float64{4.8667, 6.7}
	mCol := []float64{2.3000, 5.6, 9.45}
	delta := 0.001

	m := mat.NewDense(3, 2, data)
	assert.NotNil(m)

	// check rows mean
	meanRow := RowsMean(m)
	assert.NotNil(meanRow)
	assert.InDeltaSlice(mRow, meanRow, delta)

	// check cols mean
	meanCol := ColsMean(m)
	assert.NotNil(meanCol)
	assert.InDeltaSlice(mCol, meanCol, delta)

	// should panic
	assert.Panics(func() { RowSums(nil) })
	assert.Panics(func() { ColSums(nil) })
}

func TestCov(t *testing.T) {
	assert := assert.New(t)
	data := []float64{1, 2, 2, 4}
	delta := 0.001

	rowCov := mat.NewDense(2, 2, []float64{1.25, -1.25, -1.25, 1.25})
	colCov := mat.NewDense(2, 2, []float64{0.5, 1.0, 1.0, 2.0})

	m := mat.NewDense(2, 2, data)
	assert.NotNil(m)

	cov, err := Cov(m, "rows")
	assert.NotNil(cov)
	assert.NoError(err)

	rows, cols := cov.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			assert.InDelta(rowCov.At(r, c), cov.At(r, c), delta)
		}
	}

	cov, err = Cov(m, "cols")
	assert.NotNil(cov)
	assert.NoError(err)

	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			assert.InDelta(colCov.At(r, c), cov.At(r, c), delta)
		}
	}
}

func TestToSymDense(t *testing.T) {
	assert := assert.New(t)

	badMx := mat.NewDense(2, 1, []float64{0.5, 1.0})
	notSymMx := mat.NewDense(2, 2, []float64{0.5, 1.0, 2.0, 2.0})
	symMx := mat.NewDense(2, 2, []float64{0.5, 1.0, 1.0, 2.0})

	sym, err := ToSymDense(badMx)
	assert.Nil(sym)
	assert.Error(err)

	sym, err = ToSymDense(notSymMx)
	assert.Nil(sym)
	assert.Error(err)

	sym, err = ToSymDense(symMx)
	assert.NotNil(sym)
	assert.NoError(err)
}
