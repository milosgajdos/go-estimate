package rand

import (
	"fmt"
	"math"
	rnd "math/rand"
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// WithCovN draws n random samples from a zero-mean Normal (aka Gaussian) distribution with covariance cov.
// It returns matrix which contains the randomly generated samples stored in its columns.
// It fails with error if n is non-positive and/or smaller than 1 or if SVD factorization of cov fails.
func WithCovN(cov mat.Symmetric, n int) (*mat.Dense, error) {
	if n <= 0 {
		return nil, fmt.Errorf("Invalid number of samples requested: %d", n)
	}

	// Use SVD instead of Cholesky as Cholesky can be numerically unstable if cov is (almost) singular
	var svd mat.SVD
	ok := svd.Factorize(cov, mat.SVDFull)
	if !ok {
		return nil, fmt.Errorf("SVD factorization failed")
	}

	U := new(mat.Dense)
	svd.UTo(U)
	vals := svd.Values(nil)
	for i := range vals {
		vals[i] = math.Sqrt(vals[i])
	}
	diag := mat.NewDiagDense(len(vals), vals)
	U.Mul(U, diag)

	rows, _ := cov.Dims()
	data := make([]float64, rows*n)
	for i := range data {
		data[i] = rnd.NormFloat64()
	}
	samples := mat.NewDense(rows, n, data)
	samples.Mul(U, samples)

	return samples, nil
}

// RouletteDrawN draws n numbers randomly from a probability mass function (PMF) defined by weights in p.
// RouletteDrawN implements the Roulette Wheel Draw a.k.a. Fitness Proportionate Selection:
// - https://en.wikipedia.org/wiki/Fitness_proportionate_selection
// - http://www.keithschwarz.com/darts-dice-coins/
// It returns a slice of n indices into the vector p.
// It fails with error if p is empty or nil.
func RouletteDrawN(p []float64, n int) ([]int, error) {
	if len(p) == 0 {
		return nil, fmt.Errorf("Invalid probability weights: %v", p)
	}

	// Initialization: create the discrete CDF
	// We know that cdf is sorted in ascending order
	cdf := make([]float64, len(p))
	floats.CumSum(cdf, p)

	// Generation:
	// 1. Generate a uniformly-random value x in the range [0,1)
	// 2. Using a binary search, find the index of the smallest element in cdf larger than x
	var val float64
	indices := make([]int, n)
	for i := range indices {
		// multiply the sample with the largest CDF value; easier than normalizing to [0,1)
		val = distuv.UnitUniform.Rand() * cdf[len(cdf)-1]
		// Search returns the smallest index i such that cdf[i] > val
		indices[i] = sort.Search(len(cdf), func(i int) bool { return cdf[i] > val })
	}

	return indices, nil
}
