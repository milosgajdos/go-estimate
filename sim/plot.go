package sim

import (
	"fmt"
	"image/color"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// New2DPlot creates new plot of the simulation from the three data sources:
// model:   idealised model values
// measure: measurement values
// filter:  filter values
// It returns error if the plot fails to be created. This can be due to either of the following conditions:
// * either of the supplied data matrices is nil
// * either of the supplied data matrices does not have at least 2 columns
// * gonum plot fails to be created
func New2DPlot(model, measure, filter *mat.Dense) (*plot.Plot, error) {
	if model == nil || measure == nil || filter == nil {
		return nil, fmt.Errorf("Invalid data supplied")
	}

	_, cmd := model.Dims()
	_, cms := model.Dims()
	_, cmf := model.Dims()

	if cmd < 2 || cms < 2 || cmf < 2 {
		return nil, fmt.Errorf("Invalid data dimensions")
	}

	p := plot.New()

	p.Title.Text = "Simulation"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	legend := plot.NewLegend()

	legend.Top = true

	p.Legend = legend
	p.X.Max = 55

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
	measData := makePoints(measure)
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
		return nil, fmt.Errorf("Failed to create scatter: %v", err)
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
