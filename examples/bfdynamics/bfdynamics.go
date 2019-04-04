package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"time"

	filter "github.com/milosgajdos83/go-estimate"
	"github.com/milosgajdos83/go-estimate/estimate"
	"github.com/milosgajdos83/go-estimate/noise"
	"github.com/milosgajdos83/go-estimate/particle/bf"
	"github.com/milosgajdos83/go-estimate/sim"
	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

var (
	// resample requests particle resampling
	resample bool
)

func init() {
	flag.BoolVar(&resample, "resample", false, "Reasmple BF particles every other step")
}

func DrawMarker(img *gocv.Mat, center image.Point, c color.RGBA, d int) {
	gocv.Line(img, image.Pt(center.X-d, center.Y-d), image.Pt(center.X+d, center.Y+d), c, 2)
	gocv.Line(img, image.Pt(center.X+d, center.Y-d), image.Pt(center.X-d, center.Y+d), c, 2)
}

func main() {
	flag.Parse()

	// system dynamics matrices
	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	B := mat.NewDense(2, 1, []float64{0.5, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})
	D := mat.NewDense(1, 1, []float64{0.0})

	// ball is the model of the system we will simulate
	ball, err := sim.NewBaseModel(A, B, C, D)
	if err != nil {
		log.Fatalf("Failed to create ball model: %v", err)
	}

	// Y coordinate of the initial system state
	startY := 400.0
	// initial system state
	var x mat.Vector = mat.NewVecDense(2, []float64{startY, 0.0})
	// initial system input
	u := mat.NewVecDense(1, []float64{-9.81})

	// system state covariance
	stateCov := mat.NewSymDense(2, []float64{10, 0, 0, 10})
	stateNoise, err := noise.NewGaussian([]float64{0.0, 0.0}, stateCov)
	if err != nil {
		log.Fatalf("Failed to create state noise: %v", err)
	}

	// initial condition: initial system state + its covariance
	// Note: initial condition has 0 mean - we center particles around it
	initCond := sim.NewInitCond(x, stateCov)

	// number of simulation steps: 9 steps cover free fall
	// within the chosen size of the simulation frame
	steps := 9

	// measurement noise used to simulate real life measurements
	measCov := mat.NewSymDense(1, []float64{10})
	measNoise, err := noise.NewGaussian([]float64{0.0}, measCov)
	if err != nil {
		log.Fatalf("Failed to create measurement noise: %v", err)
	}

	// number of BF particles
	p := 100

	// system state error Probability Distribution Function (PDF)
	errPDF, _ := distmv.NewNormal([]float64{0}, measCov, nil)

	// create new Bootstrap Filter: note we don't consider noise here
	f, err := bf.New(ball, initCond, nil, nil, p, errPDF)
	if err != nil {
		log.Fatalf("Failed to create bootstrap filter: %v", err)
	}

	// z stores real system measurement: y+noise
	z := new(mat.VecDense)

	// initial filter estimate: our initial guess about position of the ball
	// Note: Our estimate will be off the model x value
	initX := &mat.VecDense{}
	initX.CloneVec(x)
	initX.AddVec(initX, stateNoise.Sample())
	var est filter.Estimate
	est, err = estimate.NewBase(initX)
	if err != nil {
		log.Fatalf("Failed to create initial estimate: %v", err)
	}

	///////////////////////////
	// Simulation parameters //
	///////////////////////////

	// velocity in X direction
	velX := 90.0
	// sampling time: measurement frequency
	timeStep := 0.5
	// initial X coordinates for model, measurement and filter
	modelX, measX, filterX := 20.0, 20.0, 20.0
	// Y offset from the top frame of the simulation window
	offsetY := 100.0

	// initial particle coordinates: particles are centered around initial system state
	// Note: particles are our "hypothesis" around model (ground truth) value hence the noise
	particles := f.Particles()
	partPts := make([]image.Point, p)
	for i := 0; i < p; i++ {
		partPts[i].X = int(math.Round(modelX + stateNoise.Sample().At(0, 0)))
		partPts[i].Y = int(math.Round((startY - particles.At(0, i)) + offsetY))
	}

	////////////////
	// SIMULATION //
	////////////////

	// GoCV simulation environment
	img := gocv.NewMatWithSize(500, 500, gocv.MatTypeCV8UC3)
	// create simple window to show the simulation
	window := gocv.NewWindow("Bootstrap Filter")

	for i := 0; i < steps; i++ {
		// model propagation i.e. ground truth propagation
		x, err = ball.Propagate(x, u, nil)
		if err != nil {
			log.Fatalf("Model propagation failed: %v", err)
		}

		// model observation i.e. ground truth observation
		y, err := ball.Observe(x, u, nil)
		if err != nil {
			log.Fatalf("Model observation failed: %v", err)
		}

		// calculate model coordinates
		modelX += velX * timeStep
		modelY := (startY - y.AtVec(0)) + offsetY
		modelPt := image.Point{
			X: int(math.Round(modelX)),
			Y: int(math.Round(modelY)),
		}

		// measurement: z = y+noise
		z.AddVec(y, measNoise.Sample())

		// measurement coordinates
		measX += velX * timeStep
		measY := (startY - z.AtVec(0)) + offsetY
		measPt := image.Point{
			X: int(math.Round(measX + measNoise.Sample().AtVec(0))),
			Y: int(math.Round(measY + measNoise.Sample().AtVec(0))),
		}

		// propagate filter particles to the next step
		pred, err := f.Predict(est.Val(), u)
		if err != nil {
			log.Fatalf("Failed to predict next filter state: %v", err)
		}

		// correct predicted estimate using measurement z
		est, err = f.Update(pred.Val(), u, z)
		if err != nil {
			log.Fatalf("Failed to update the filter state: %v", err)
		}

		// get corrected output
		yFilter, err := ball.Observe(est.Val(), u, nil)
		if err != nil {
			log.Fatalf("Failed to observe filter output: %v", err)
		}

		// filter output coordinates i.e. corrected estimate
		filterX += velX * timeStep
		filterY := (startY - yFilter.AtVec(0)) + offsetY
		filterPt := image.Point{
			//X: int(math.Round(filterX + measNoise.Sample().AtVec(0))),
			X: int(math.Round(filterX)),
			Y: int(math.Round(filterY)),
		}

		// reset all pixels to 255 i.e. to white background
		img.SetTo(gocv.Scalar{255, 255, 255, 255})

		// draw particles in grey-ish color
		particles := f.Particles()
		for j := 0; j < p; j++ {
			partPts[j].X += int(math.Round(velX * timeStep))
			partPts[j].Y = int(math.Round((startY - particles.At(0, j)) + offsetY))
			gocv.Circle(&img, partPts[j], 1, color.RGBA{169, 169, 169, 0}, 1)
		}

		// draw model (ground truth) point marker
		DrawMarker(&img, modelPt, color.RGBA{0, 255, 0, 0}, 2)
		// draw measurement point marker
		DrawMarker(&img, measPt, color.RGBA{255, 0, 0, 0}, 2)
		// draw filter point marker
		DrawMarker(&img, filterPt, color.RGBA{100, 149, 237, 0}, 2)

		window.IMShow(img)
		if window.WaitKey(int(500)) == 27 {
			fmt.Printf("Shutting down: ESC pressed\n")
			break
		}

		time.Sleep(600 * time.Millisecond)

		if resample && i > 0 && i%2 == 0 {
			if err := f.Resample(0.0); err != nil {
				log.Fatalf("Failed to resample filter particles: %v", err)
			}
		}
	}
}
