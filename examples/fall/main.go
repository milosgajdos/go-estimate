package main

import (
	"fmt"
	"image/color"
	"log"

	"github.com/milosgajdos83/go-filter/rnd"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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
func (b *Fall) Propagate(x, u *mat.Dense) (*mat.Dense, error) {
	out := new(mat.Dense)
	out.Mul(b.A, x)

	outU := new(mat.Dense)
	outU.Mul(b.B, u)

	out.Add(out, outU)

	return out, nil
}

// Observe observes external state of falling ball based with internal state x
func (b *Fall) Observe(x, u *mat.Dense) (*mat.Dense, error) {
	out := new(mat.Dense)
	out.Mul(b.C, x)

	outU := new(mat.Dense)
	outU.Mul(b.D, u)

	out.Add(out, outU)

	return out, nil
}

func plotSystem(model, meas *mat.Dense, name string) error {
	p, err := plot.New()
	if err != nil {
		return err
	}
	p.Title.Text = "Falling Ball"
	p.X.Label.Text = "time"
	p.Y.Label.Text = "position"

	// Make a scatter plotter for model data
	modelData := makePoints(model)
	modelScatter, err := plotter.NewScatter(modelData)
	if err != nil {
		return err
	}
	modelScatter.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	modelScatter.GlyphStyle.Radius = vg.Points(2)

	p.Add(modelScatter)
	p.Legend.Add("model", modelScatter)

	// Make a scatter plotter for measurement data
	measData := makePoints(meas)
	measScatter, err := plotter.NewScatter(measData)
	if err != nil {
		return err
	}
	measScatter.GlyphStyle.Color = color.RGBA{G: 255, A: 128}
	measScatter.GlyphStyle.Radius = vg.Points(2)

	p.Add(measScatter)
	p.Legend.Add("measurement", measScatter)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, name); err != nil {
		return err
	}

	return nil
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

	steps := 10
	// model measurement i.e. true state
	model := mat.NewDense(steps, 2, nil)
	// output measurement i.e. output + error
	meas := mat.NewDense(steps, 2, nil)
	measCov := mat.NewDense(1, 1, []float64{6.25})

	for i := 0; i < steps; i++ {
		x, err = ball.Propagate(x, u)
		if err != nil {
			log.Fatalf("Propagation error: %v", err)
		}

		//fmt.Printf("State %d:\n%v\n", i, mxFormat(x))

		y, err := ball.Observe(x, u)
		if err != nil {
			log.Fatalf("Observation error: %v", err)
		}
		model.Set(i, 1, y.At(0, 0))
		model.Set(i, 0, float64(i))

		// generate measurement error and perturb output with it
		measErr, err := rnd.WithCovN(measCov, 1)
		if err != nil {
			log.Fatalf("Measurement error: %v", err)
		}
		y.Add(y, measErr)
		meas.Set(i, 1, y.At(0, 0))
		meas.Set(i, 0, float64(i))

		//fmt.Printf("Output %d:\n%v\n", i, mxFormat(y))
		//fmt.Println("----------------")
	}

	if err := plotSystem(model, meas, "system.png"); err != nil {
		log.Fatalf("Failed to plot system: %v", err)
	}
}
