package matrix

import (
	"errors"
	"fmt"
	"strings"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Format returns matrix formatter for printing matrices
func Format(m mat.Matrix) fmt.Formatter {
	return mat.Formatted(m, mat.Prefix(""), mat.Squeeze())
}

// RowSums returns a slice containing m row sums.
// It panics if m is nil.
func RowSums(m *mat.Dense) []float64 {
	rows, _ := m.Dims()
	sum := make([]float64, rows)

	for i := 0; i < rows; i++ {
		sum[i] = floats.Sum(m.RawRowView(i))
	}

	return sum
}

// ColSums returns a slice containing m column sums.
// It panics if m is nil.
func ColSums(m *mat.Dense) []float64 {
	_, cols := m.Dims()
	sum := make([]float64, cols)

	for i := 0; i < cols; i++ {
		sum[i] = mat.Sum(m.ColView(i))
	}

	return sum
}

// RowsMean returns a slice containing m row mean values.
// It panics if m is nil
func RowsMean(m *mat.Dense) []float64 {
	rows, _ := m.Dims()
	mean := ColSums(m)

	floats.Scale(1/float64(rows), mean)

	return mean
}

// ColsMean returns a slice containing m column mean values.
// It panics if m is nil
func ColsMean(m *mat.Dense) []float64 {
	_, cols := m.Dims()
	mean := RowSums(m)

	floats.Scale(1/float64(cols), mean)

	return mean
}

// Cov calculates a covariance matrix of data stored across dim dimension.
// It returns error if the covariance could not be calculated.
func Cov(m *mat.Dense, dim string) (*mat.SymDense, error) {
	// 1. We will calculate zero mean matrix x of the data
	// 2. 1/(n-1)(x * x^T) will give us covariance of the data
	rows, cols := m.Dims()

	// calculate mean data vector across dimension dim
	var mean []float64
	var count float64
	if strings.EqualFold(dim, "rows") {
		mean = RowsMean(m)
		count = float64(rows)
	} else {
		mean = ColsMean(m)
		count = float64(cols)
	}

	// x is zero-mean matrix of data stored in dimension dim
	x := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if strings.EqualFold(dim, "rows") {
				x.Set(r, c, m.At(r, c)-mean[c])
			} else {
				x.Set(r, c, m.At(r, c)-mean[r])
			}
		}
	}

	cov := new(mat.Dense)
	cov.Mul(x, x.T())
	cov.Scale(1/(count-1.0), cov)

	return ToSymDense(cov)
}

// ToSymDense converts m to SymDense (symmetric Dense matrix) if possible.
// It returns error if the provided Dense matrix is not symmetric.
func ToSymDense(m *mat.Dense) (*mat.SymDense, error) {
	r, c := m.Dims()
	if r != c {
		return nil, errors.New("Matrix must be square")
	}

	mT := m.T()
	vals := make([]float64, r*c)
	idx := 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if i != j && !floats.EqualWithinAbsOrRel(mT.At(i, j), m.At(i, j), 1e-6, 1e-2) {
				return nil, fmt.Errorf("Matrix not symmetric (%d, %d): %.40f != %.40f\n%v",
					i, j, mT.At(i, j), m.At(i, j), Format(m))
			}
			vals[idx] = m.At(i, j)
			idx++
		}
	}

	return mat.NewSymDense(r, vals), nil
}
