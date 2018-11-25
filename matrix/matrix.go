package matrix

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

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
