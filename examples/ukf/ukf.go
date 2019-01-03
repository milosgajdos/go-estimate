package main

import (
	"fmt"
	"image/color"
	"log"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/go-filter/kalman/ukf"
	"github.com/milosgajdos83/go-filter/model"
	"github.com/milosgajdos83/go-filter/noise"
	"github.com/milosgajdos83/matrix"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func NewSystemPlot(model, meas, filter *mat.Dense) (*plot.Plot, error) {
	p, err := plot.New()
	if err != nil {
		return nil, err
	}
	p.Title.Text = "Falling Ball"
	p.X.Label.Text = "time"
	p.Y.Label.Text = "position"

	// Make a scatter plotter for model data
	modelData := makePoints(model)
	modelScatter, err := plotter.NewScatter(modelData)
	if err != nil {
		return nil, err
	}
	modelScatter.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	modelScatter.Shape = draw.PyramidGlyph{}
	modelScatter.GlyphStyle.Radius = vg.Points(3)

	p.Add(modelScatter)
	p.Legend.Add("model", modelScatter)

	// Make a scatter plotter for measurement data
	measData := makePoints(meas)
	measScatter, err := plotter.NewScatter(measData)
	if err != nil {
		return nil, err
	}
	measScatter.GlyphStyle.Color = color.RGBA{G: 255, A: 128}
	measScatter.GlyphStyle.Radius = vg.Points(3)

	p.Add(measScatter)
	p.Legend.Add("measurement", measScatter)

	// Make a scatter plotter for filter data
	filterPoints := makePoints(filter)
	filterScatter, err := plotter.NewScatter(filterPoints)
	if err != nil {
		log.Fatalf("Failed to create partcle scatter: %v", err)
	}
	filterScatter.GlyphStyle.Color = color.RGBA{R: 169, G: 169, B: 169}
	filterScatter.Shape = draw.CrossGlyph{}
	filterScatter.GlyphStyle.Radius = vg.Points(3)

	p.Add(filterScatter)
	p.Legend.Add("filtered", filterScatter)

	return p, nil
}

func makePoints(m *mat.Dense) plotter.XYs {
	r, _ := m.Dims()
	pts := make(plotter.XYs, r)
	for i := 0; i < r; i++ {
		pts[i].X = m.At(i, 0)
		pts[i].Y = m.At(i, 1)
	}

	return pts
}

func main() {
	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	// ball is the model of the system we will simulate
	ball, err := model.NewBase(A, B, C, D)
	if err != nil {
		log.Fatalf("Failed to created ball: %v", err)
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
	measCov := mat.NewSymDense(1, []float64{0.25})
	measNoise, err := noise.NewGaussian([]float64{0.0}, measCov)
	if err != nil {
		log.Fatalf("Failed to create measurement noise: %v", err)
	}

	// output corrected by filter
	filterOut := mat.NewDense(steps, 2, nil)

	// initial state covariance
	stateCov := mat.NewSymDense(2, []float64{0.25, 0, 0, 0.05})
	stateNoise, err := noise.NewGaussian([]float64{0.0, 0.0}, stateCov)
	if err != nil {
		log.Fatalf("Failed to create state noise: %v", err)
	}

	// initial condition of UKF
	initCond := model.NewInitCond(x, stateCov)

	// z stores real system measurement: y+noise
	z := new(mat.VecDense)

	// filter initial estimate
	initX := &mat.VecDense{}
	initX.AddVec(x, stateNoise.Sample())
	var est filter.Estimate
	est, err = estimate.NewBase(initX)
	if err != nil {
		log.Fatalf("Failed to create initial estimate: %v", err)
	}

	// UKF configuration
	c := &ukf.Config{
		Alpha: 0.95,
		Beta:  2.0,
		Kappa: 2.75,
	}

	f, err := ukf.New(ball, initCond, nil, measNoise, c)
	if err != nil {
		log.Fatalf("Failed to create UKF filter: %v", err)
	}

	// filter state error
	filterErr := 0.0

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
		noise := measNoise.Sample()
		fmt.Println("NOISE:", matrix.Format(noise))
		z.AddVec(y, noise)
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
			log.Fatalf("Filter Udpate error: %v", err)
		}

		// calculate filter state error
		xErr := &mat.Dense{}
		xErr.Sub(x, est.Val())
		pInv := &mat.Dense{}
		pInv.Inverse(est.Cov())
		xerrPinv := &mat.Dense{}
		xerrPinv.Mul(xErr.T(), pInv)
		res := &mat.Dense{}
		res.Mul(xerrPinv, xErr)
		filterErr += res.At(0, 0)

		// get corrected output
		yFilter, err := ball.Observe(est.Val(), u, nil)
		if err != nil {
			log.Fatalf("Model Observation error: %v", err)
		}
		fmt.Printf("FILTER Output %d:\n%v\n", i, matrix.Format(yFilter))

		// store results for plotting
		filterOut.Set(i, 0, float64(i))
		filterOut.Set(i, 1, yFilter.AtVec(0))

		fmt.Printf("CORRECTED State  %d:\n%v\n", i, matrix.Format(est.Val()))
		fmt.Printf("CORRECTED Output %d:\n%v\n", i, matrix.Format(yFilter))
		fmt.Println("----------------")
	}

	fmt.Println("XERR:", filterErr)

	plt, err := NewSystemPlot(modelOut, measOut, filterOut)
	if err != nil {
		log.Fatalf("Failed to make plot: %v", err)
	}

	name := "system.png"
	// Save the plot to a PNG file.
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, name); err != nil {
		log.Fatalf("Failed to save plot to %s: %v", name, err)
	}
}
