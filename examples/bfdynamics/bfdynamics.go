package main

import (
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
	"github.com/milosgajdos83/matrix"
	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

func DrawMarker(img *gocv.Mat, center image.Point, c color.RGBA, d int) {
	gocv.Line(img, image.Pt(center.X-d, center.Y-d), image.Pt(center.X+d, center.Y+d), c, 2)
	gocv.Line(img, image.Pt(center.X+d, center.Y-d), image.Pt(center.X-d, center.Y+d), c, 2)
}

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
	startY := 400.0
	var x mat.Vector = mat.NewVecDense(2, []float64{startY, 0.0})
	u := mat.NewVecDense(1, []float64{-9.81})

	// state covariance
	stateMean := []float64{0.0, 0.0}
	stateCov := mat.NewSymDense(2, []float64{10, 0, 0, 10})
	stateNoise, err := noise.NewGaussian(stateMean, stateCov)
	if err != nil {
		log.Fatalf("Failed to create state noise: %v", err)
	}

	// initial condition
	initCond := sim.NewInitCond(x, stateCov)

	// number of simulation steps
	steps := 9

	// measurement noise used to simulate real system
	measMean := []float64{0.0}
	measCov := mat.NewSymDense(1, []float64{10})
	measNoise, err := noise.NewGaussian(measMean, measCov)
	if err != nil {
		log.Fatalf("Failed to create measurement noise: %v", err)
	}

	// output corrected by filter
	filterOut := mat.NewDense(steps, 2, nil)

	// number of BF particles
	p := 100

	// system state error PDF
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

	fmt.Println("------------------------------------------------------")

	// velocity in X direction
	xVel := 90.0
	timestep := 0.5
	xDist, xMeasDist, xFilterDist := 20.0, 20.0, 20.0
	yOffset := 100.0

	// particle coordinates
	particles := f.Particles()
	partPts := make([]image.Point, p)
	for i := 0; i < p; i++ {
		partPts[i].X = int(math.Round(xDist + stateNoise.Sample().At(0, 0)))
		partPts[i].Y = int(math.Round((startY - particles.At(0, i)) + yOffset))
		//fmt.Printf("%v\n", partPts[i])
	}

	// GoCV simulation environment
	img := gocv.NewMatWithSize(500, 500, gocv.MatTypeCV8UC3)
	// create simple window to show the simulation
	window := gocv.NewWindow("Bootstrap Filter")

	for i := 0; i < steps; i++ {
		// ground truth propagation
		x, err = ball.Propagate(x, u, nil)
		if err != nil {
			log.Fatalf("Model Propagation error: %v", err)
		}

		fmt.Printf("Truth State %d:\n%v\n", i, matrix.Format(x))

		// ground truth observation
		y, err := ball.Observe(x, u, nil)
		if err != nil {
			log.Fatalf("Model Observation error: %v", err)
		}

		fmt.Printf("Truth Output %d:\n%v\n", i, matrix.Format(y))

		xDist += xVel * timestep
		yDist := (startY - y.AtVec(0)) + yOffset
		truthPt := image.Point{X: int(math.Round(xDist)), Y: int(math.Round(yDist))}

		fmt.Printf("Truth point: %v\n", truthPt)

		// measurement: z = y+noise
		z.AddVec(y, measNoise.Sample())

		fmt.Printf("Measurement %d:\n%v\n", i, matrix.Format(z))

		xMeasDist += xVel * timestep
		yMeasDist := (startY - z.AtVec(0)) + yOffset
		measPt := image.Point{
			X: int(math.Round(xMeasDist + measNoise.Sample().AtVec(0))),
			Y: int(math.Round(yMeasDist)),
		}

		fmt.Printf("Truth point: %v\n", measPt)

		// propagate particle filters to the next step
		pred, err := f.Predict(est.Val(), u)
		if err != nil {
			log.Fatalf("Failed to predict next filter state: %v", err)
		}

		// correct state estimate using measurement z
		est, err = f.Update(pred.Val(), u, z)
		if err != nil {
			log.Fatalf("Failed to update the filter state: %v", err)
		}

		// get corrected output
		yFilter, err := ball.Observe(est.Val(), u, nil)
		if err != nil {
			log.Fatalf("Failed to observe filter output: %v", err)
		}

		// store results for plotting
		filterOut.Set(i, 0, float64(i))
		filterOut.Set(i, 1, est.Val().AtVec(0))

		fmt.Printf("CORRECTED State  %d:\n%v\n", i, matrix.Format(est.Val()))
		fmt.Printf("CORRECTED Output %d:\n%v\n", i, matrix.Format(yFilter))
		fmt.Println("------------------------------------------------------")

		// reset all pixels to 255
		img.SetTo(gocv.Scalar{255, 255, 255, 255})

		// draw markers
		DrawMarker(&img, truthPt, color.RGBA{0, 255, 0, 0}, 2)
		DrawMarker(&img, measPt, color.RGBA{255, 0, 0, 0}, 2)

		// draw particles in grey-ish color
		particles := f.Particles()
		for j := 0; j < p; j++ {
			partPts[j].X += int(math.Round(xVel * timestep))
			partPts[j].Y = int(math.Round((startY - particles.At(0, j)) + yOffset))
			gocv.Circle(&img, partPts[j], 1, color.RGBA{169, 169, 169, 0}, 1)
		}

		xFilterDist += xVel * timestep
		yFilterDist := (startY - yFilter.AtVec(0)) + yOffset
		filterPt := image.Point{
			X: int(math.Round(xFilterDist + measNoise.Sample().AtVec(0))),
			Y: int(math.Round(yFilterDist)),
		}
		DrawMarker(&img, filterPt, color.RGBA{100, 149, 237, 0}, 2)

		window.IMShow(img)
		if window.WaitKey(int(500)) == 27 {
			fmt.Printf("Shutting down: ESC pressed\n")
			break
		}

		time.Sleep(600 * time.Millisecond)

		// resample every other step
		//if i > 0 && i%2 == 0 {
		//	if err := f.Resample(0.0); err != nil {
		//		log.Fatalf("Resampling failed: %v", err)
		//	}
		//}
	}
}
