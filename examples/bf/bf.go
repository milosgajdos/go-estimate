package main

import (
	"fmt"
	"log"

	filter "github.com/milosgajdos83/go-estimate"
	"github.com/milosgajdos83/go-estimate/estimate"
	"github.com/milosgajdos83/go-estimate/noise"
	"github.com/milosgajdos83/go-estimate/particle/bf"
	"github.com/milosgajdos83/go-estimate/sim"
	"github.com/milosgajdos83/matrix"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/plot/vg"
)

func main() {
	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	// ball is the model of the system we will simulate
	ball, err := sim.NewBaseModel(A, B, C, D)
	if err != nil {
		log.Fatalf("Failed to create ball: %v", err)
	}

	// initial system state and control input
	var x mat.Vector = mat.NewVecDense(2, []float64{100.0, 0.0})
	u := mat.NewVecDense(1, []float64{-1.0})

	// number of simulation steps
	steps := 14

	// modelOut measurements i.e. true model output state
	modelOut := mat.NewDense(steps, 2, nil)

	// output measurement i.e. output + error
	measOut := mat.NewDense(steps, 2, nil)

	// measurement noise used to simulate real system
	measMean := []float64{0.0}
	measCov := mat.NewSymDense(1, []float64{0.25})
	measNoise, err := noise.NewGaussian(measMean, measCov)
	if err != nil {
		log.Fatalf("Failed to create measurement noise: %v", err)
	}

	// output corrected by filter
	filterOut := mat.NewDense(steps, 2, nil)

	// initial condition
	stateCov := mat.NewSymDense(2, []float64{1, 0, 0, 1})
	initCond := sim.NewInitCond(x, stateCov)

	p := 100
	errPDF, _ := distmv.NewNormal([]float64{0}, measCov, nil)
	// create new particle filter
	f, err := bf.New(ball, initCond, nil, nil, p, errPDF)
	if err != nil {
		log.Fatalf("Failed to create particle filter: %v", err)
	}

	// z stores real system measurement: y+noise
	z := new(mat.VecDense)
	// filter initial estimate
	var est filter.Estimate
	est, err = estimate.NewBase(x)
	if err != nil {
		log.Fatalf("Failed to create initial estimate: %v", err)
	}

	for i := 0; i < steps; i++ {
		// ground truth propagation
		x, err = ball.Propagate(x, u, nil)
		if err != nil {
			log.Fatalf("Model Propagation error: %v", err)
		}

		fmt.Printf("TRUTH State %d:\n%v\n", i, matrix.Format(x))

		// ground truth observation
		y, err := ball.Observe(x, u, nil)
		if err != nil {
			log.Fatalf("Model Observation error: %v", err)
		}

		fmt.Printf("TRUTH Output %d:\n%v\n", i, matrix.Format(y))

		// store results for plotting
		modelOut.Set(i, 0, float64(i))
		modelOut.Set(i, 1, y.AtVec(0))

		// measurement: z = y+noise
		z.AddVec(y, measNoise.Sample())
		// store results for plotting
		measOut.Set(i, 0, float64(i))
		measOut.Set(i, 1, z.AtVec(0))

		fmt.Printf("Measurement %d:\n%v\n", i, matrix.Format(z))

		// propagate particle filters to the next step
		pred, err := f.Predict(est.Val(), u)
		if err != nil {
			log.Fatalf("Filter Prediction error: %v", err)
		}

		// correct state estimate using measurement z
		est, err = f.Update(pred.Val(), u, z)
		if err != nil {
			log.Fatalf("Filter Correction error: %v", err)
		}

		// get corrected output
		yFilter, err := ball.Observe(est.Val(), u, nil)
		if err != nil {
			log.Fatalf("Model Observation error: %v", err)
		}
		fmt.Printf("FILTER Output %d:\n%v\n", i, matrix.Format(yFilter))

		// store results for plotting
		filterOut.Set(i, 0, float64(i))
		filterOut.Set(i, 1, est.Val().AtVec(0))

		fmt.Printf("CORRECTED State  %d:\n%v\n", i, matrix.Format(est.Val()))
		fmt.Printf("CORRECTED Output %d:\n%v\n", i, matrix.Format(yFilter))
		fmt.Println("----------------")

		// resample every other step
		if i%2 == 0 {
			if err := f.Resample(0.0); err != nil {
				log.Fatalf("Resampling failed: %v", err)
			}
		}
	}

	plt, err := sim.New2DPlot(modelOut, measOut, filterOut)
	if err != nil {
		log.Fatalf("Failed to make plot: %v", err)
	}

	name := "system.png"
	// Save the plot to a PNG file.
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, name); err != nil {
		log.Fatalf("Failed to save plot to %s: %v", name, err)
	}
}
