package main

import (
	"fmt"
	"image/color"
	"log"

	"github.com/milosgajdos83/go-filter/bootstrap"
	"github.com/milosgajdos83/go-filter/rnd"
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
func (b *Fall) Propagate(x, u mat.Matrix) (*mat.Dense, error) {
	out := new(mat.Dense)
	out.Mul(b.A, x)

	outU := new(mat.Dense)
	outU.Mul(b.B, u)

	out.Add(out, outU)

	return out, nil
}

// Observe observes external state of falling ball based with internal state x
func (b *Fall) Observe(x, u mat.Matrix) (*mat.Dense, error) {
	out := new(mat.Dense)
	out.Mul(b.C, x)

	outU := new(mat.Dense)
	outU.Mul(b.D, u)

	out.Add(out, outU)

	return out, nil
}

// Dims returns input and output dimensions
func (b *Fall) Dims() (int, int) {
	_, aCols := b.A.Dims()
	dRows, _ := b.D.Dims()

	return aCols, dRows
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

func mxFormat(m *mat.Dense) fmt.Formatter {
	return mat.Formatted(m, mat.Prefix(""), mat.Squeeze())
}

func main() {
	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	ball, err := NewFall(A, B, C, D)
	if err != nil {
		log.Fatalf("Failed to created ball: %v", err)
	}

	// initial system state and control input
	x := mat.NewDense(2, 1, []float64{100.0, 0.0})
	u := mat.NewDense(1, 1, []float64{-1.0})

	steps := 14
	// modelOut measurement i.e. true output state
	modelOut := mat.NewDense(steps, 2, nil)
	// output measurement i.e. output + error
	meas := mat.NewDense(steps, 2, nil)
	measCov := mat.NewSymDense(1, []float64{0.25})
	// corrected output by filter
	filterOut := mat.NewDense(steps, 2, nil)

	// create Bootstrap Filter
	pCount := 100
	errOut, _ := distmv.NewNormal([]float64{0}, measCov, nil)
	config := &bootstrap.Config{
		Model:         ball,
		ParticleCount: pCount,
		Err:           errOut,
	}

	stateCov := mat.NewSymDense(2, []float64{1, 0, 0, 1})
	start := &bootstrap.InitCond{
		State: x,
		Cov:   stateCov,
	}
	// create new filter and initialize it
	f, err := bootstrap.NewFilter(config)
	if err != nil {
		log.Fatalf("Failed to create filter: %v", err)
	}
	if err := f.Init(start); err != nil {
		log.Fatalf("Failed to initialise filter: %v", err)
	}

	// z is system measurement: y+noise
	z := new(mat.Dense)

	// filter initial internal state
	xFilter := x
	// filter output state
	yFilter := new(mat.Dense)

	for i := 0; i < steps; i++ {
		// model internal state ground truth
		x, err = ball.Propagate(x, u)
		if err != nil {
			log.Fatalf("Model Propagation error: %v", err)
		}

		fmt.Printf("TRUTH State %d:\n%v\n", i, mxFormat(x))

		// model output state ground truth
		y, err := ball.Observe(x, u)
		if err != nil {
			log.Fatalf("Model Observation error: %v", err)
		}
		modelOut.Set(i, 0, float64(i))
		modelOut.Set(i, 1, y.At(0, 0))

		fmt.Printf("TRUTH Output %d:\n%v\n", i, mxFormat(y))

		// measurement noise
		noise, err := rnd.WithCovN(measCov, 1)
		if err != nil {
			log.Fatalf("Measurement error: %v", err)
		}
		// measurement: z = y+noise
		z.Add(y, noise)
		meas.Set(i, 0, float64(i))
		meas.Set(i, 1, z.At(0, 0))

		fmt.Printf("Measurement %d:\n%v\n", i, mxFormat(z))

		// propagate particle filters to the next step
		yFilter, err = f.Predict(xFilter, u)
		if err != nil {
			log.Fatalf("Filter Prediction error: %v", err)
		}

		fmt.Printf("FILTER Output %d:\n%v\n", i, mxFormat(yFilter))

		// system model i.e. ground TRUTH
		xFilter, err = f.Correct(xFilter, z)
		if err != nil {
			log.Fatalf("Filter Correction error: %v", err)
		}

		fmt.Printf("CORRECTED State %d:\n%v\n", i, mxFormat(xFilter))

		yCorr, err := f.Model.Observe(xFilter, u)
		if err != nil {
			log.Fatalf("Filter Observation error: %v", err)
		}
		filterOut.Set(i, 0, float64(i))
		filterOut.Set(i, 1, yCorr.At(0, 0))

		fmt.Printf("CORRECTED Output %d:\n%v\n", i, mxFormat(yCorr))
		fmt.Println("----------------")
	}

	p, err := NewSystemPlot(modelOut, meas, filterOut)
	if err != nil {
		log.Fatalf("Failed to make plot: %v", err)
	}

	//name := "system.svg"
	name := "system.png"
	// Save the plot to a PNG file.
	if err := p.Save(10*vg.Inch, 10*vg.Inch, name); err != nil {
		log.Fatalf("Failed to save plot to %s: %v", name, err)
	}

}
