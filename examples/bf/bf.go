package main

import (
	"fmt"
	"image/color"
	"log"

	filter "github.com/milosgajdos83/go-filter"
	"github.com/milosgajdos83/go-filter/bootstrap"
	"github.com/milosgajdos83/go-filter/estimate"
	"github.com/milosgajdos83/go-filter/matrix"
	"github.com/milosgajdos83/go-filter/noise"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Fall is a model of a falling ball
type Fall struct {
	// A is internal state matrix
	A *mat.Dense
	// B is control matrix
	B *mat.Dense
	// C is output state matrix
	C *mat.Dense
	// D is output control matrix
	D *mat.Dense
}

// NewFall creates a model of falling ball and returns it
func NewFall(A, B, C, D *mat.Dense) (*Fall, error) {
	return &Fall{A: A, B: B, C: C, D: D}, nil
}

// Propagate propagates internal state x of falling ball to the next step
func (b *Fall) Propagate(x, u, q mat.Vector) (mat.Vector, error) {
	out := new(mat.Dense)
	out.Mul(b.A, x)

	outU := new(mat.Dense)
	outU.Mul(b.B, u)

	out.Add(out, outU)

	return out.ColView(0), nil
}

// Observe observes external state of falling ball given internal state x and input u
func (b *Fall) Observe(x, u, r mat.Vector) (mat.Vector, error) {
	out := new(mat.Dense)
	out.Mul(b.C, x)

	outU := new(mat.Dense)
	outU.Mul(b.D, u)

	out.Add(out, outU)

	return out.ColView(0), nil
}

// Dims returns input and output model dimensions
func (b *Fall) Dims() (int, int) {
	_, in := b.A.Dims()
	out, _ := b.D.Dims()

	return in, out
}

type initCnd struct {
	state mat.Vector
	cov   mat.Symmetric
}

func (c *initCnd) State() mat.Vector {
	return c.state
}

func (c *initCnd) Cov() mat.Symmetric {
	return c.cov
}

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
	ball, err := NewFall(A, B, C, D)
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
	measMean := []float64{0.0}
	measCov := mat.NewSymDense(1, []float64{0.25})
	measNoise, err := noise.NewGaussian(measMean, measCov)
	if err != nil {
		log.Fatalf("Failed to create measurement noise: %v", err)
	}

	// output corrected by filter
	filterOut := mat.NewDense(steps, 2, nil)

	// initial condition
	var stateCov mat.Symmetric = mat.NewSymDense(2, []float64{1, 0, 0, 1})
	initCond := &initCnd{
		state: x,
		cov:   stateCov,
	}

	p := 100
	errPDF, _ := distmv.NewNormal([]float64{0}, measCov, nil)
	// create new bootstrap filter
	f, err := bootstrap.New(ball, initCond, nil, nil, p, errPDF)
	if err != nil {
		log.Fatalf("Failed to create bootstrap filter: %v", err)
	}

	// z stores real system measurement: y+noise
	z := new(mat.VecDense)
	// filter initial estimate
	var est filter.Estimate
	est, err = estimate.NewBase(x, nil)
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
		pred, err := f.Predict(est.State(), u)
		if err != nil {
			log.Fatalf("Filter Prediction error: %v", err)
		}

		fmt.Printf("FILTER Output %d:\n%v\n", i, matrix.Format(pred.Output()))

		// correct state estimate using measurement z
		est, err = f.Update(est.State(), u, z)
		if err != nil {
			log.Fatalf("Filter Correction error: %v", err)
		}

		// store results for plotting
		filterOut.Set(i, 0, float64(i))
		filterOut.Set(i, 1, est.Output().AtVec(0))

		fmt.Printf("CORRECTED State  %d:\n%v\n", i, matrix.Format(est.State()))
		fmt.Printf("CORRECTED Output %d:\n%v\n", i, matrix.Format(est.Output()))
		fmt.Println("----------------")

		// resample every other step
		if i%2 == 0 {
			if err := f.Resample(0.0); err != nil {
				log.Fatalf("Resampling failed: %v", err)
			}
		}
	}

	plt, err := NewSystemPlot(modelOut, measOut, filterOut)
	if err != nil {
		log.Fatalf("Failed to make plot: %v", err)
	}

	//name := "system.svg"
	name := "system.png"
	// Save the plot to a PNG file.
	if err := plt.Save(10*vg.Inch, 10*vg.Inch, name); err != nil {
		log.Fatalf("Failed to save plot to %s: %v", name, err)
	}
}
