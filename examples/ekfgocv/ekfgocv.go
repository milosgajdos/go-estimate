package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"math/rand"

	filter "github.com/milosgajdos83/go-estimate"
	"github.com/milosgajdos83/go-estimate/estimate"
	"github.com/milosgajdos83/go-estimate/kalman/ekf"
	"github.com/milosgajdos83/go-estimate/noise"
	"github.com/milosgajdos83/go-estimate/sim"
	"github.com/milosgajdos83/matrix"
	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
)

func GetDotPos(center image.Point, r, angle float64) image.Point {
	p := image.Pt(center.X, center.Y)
	x := math.Cos(angle) * r
	y := (-math.Sin(angle)) * r

	return p.Add(image.Pt(int(x), int(y)))
}

func DrawMarker(img *gocv.Mat, center image.Point, c color.RGBA, d int) {
	gocv.Line(img, image.Pt(center.X-d, center.Y-d), image.Pt(center.X+d, center.Y+d), c, 3)
	gocv.Line(img, image.Pt(center.X+d, center.Y-d), image.Pt(center.X-d, center.Y+d), c, 3)
}

func main() {
	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})

	// dot is the model of the system we will simulate
	dot, err := sim.NewBaseModel(A, nil, C, nil)
	if err != nil {
		log.Fatalf("Failed to created dot model: %v", err)
	}

	// measurement noise used to simulate real system
	measCov := mat.NewSymDense(1, []float64{1e-1})
	measNoise, err := noise.NewGaussian([]float64{0.0}, measCov)
	if err != nil {
		log.Fatalf("Failed to create measurement noise: %v", err)
	}

	// initial state covariance
	stateCov := mat.NewSymDense(2, []float64{1e-5, 0, 0, 1e-5})
	stateNoise, err := noise.NewGaussian([]float64{0.0, 0.0}, stateCov)
	if err != nil {
		log.Fatalf("Failed to create state noise: %v", err)
	}

	// initial system state: we simply generate some random numbers
	x1, x2 := rand.NormFloat64()*0.1, rand.NormFloat64()*0.1
	var x mat.Vector = mat.NewVecDense(2, []float64{x1, x2})
	fmt.Println("Initial Model State: \n", matrix.Format(x))

	// initial condition of KF
	initCond := sim.NewInitCond(x, stateCov)

	// z stores real system measurement: y+noise
	z := new(mat.VecDense)

	// filter initial estimate
	initX := &mat.VecDense{}
	initX.CloneVec(x)
	initX.AddVec(initX, stateNoise.Sample())
	fmt.Println("Initial KF State: \n", matrix.Format(initX))

	var est filter.Estimate
	est, err = estimate.NewBase(initX)
	if err != nil {
		log.Fatalf("Failed to create initial estimate: %v", err)
	}

	// create Extended Kalman Filter
	f, err := ekf.New(dot, initCond, stateNoise, measNoise)
	if err != nil {
		log.Fatalf("Failed to create EKF filter: %v", err)
	}

	fmt.Println("=============================================")

	// GoCV simulation environment
	img := gocv.NewMatWithSize(500, 500, gocv.MatTypeCV8UC3)
	center := image.Pt(img.Cols()/2, img.Rows()/2)
	r := float64(img.Cols()) / 3.0
	// create simple window to show the simulation
	window := gocv.NewWindow("Extended Kalman Filter")

	for {
		// ground truth propagation
		x, err = dot.Propagate(x, nil, nil)
		if err != nil {
			log.Fatalf("Model Propagation error: %v", err)
		}

		fmt.Printf("TRUTH State:\n%v\n", matrix.Format(x))

		// ground truth observation
		y, err := dot.Observe(x, nil, nil)
		if err != nil {
			log.Fatalf("Model Observation error: %v", err)
		}

		fmt.Printf("TRUTH Output:\n%v\n", matrix.Format(y))

		yPt := GetDotPos(center, r, y.At(0, 0))
		fmt.Printf("Model Out Point: %v\n", yPt)

		// measurement: z = y+noise
		noise := measNoise.Sample()
		fmt.Println("NOISE:", matrix.Format(noise))
		z.AddVec(y, noise)
		fmt.Printf("Measurement:\n%v\n", matrix.Format(z))

		measPt := GetDotPos(center, r, z.At(0, 0))
		fmt.Printf("Meas Point: %v\n", measPt)

		// propagate particle filters to the next step
		pred, err := f.Predict(est.Val(), nil)
		if err != nil {
			log.Fatalf("Filter Prediction error: %v", err)
		}

		// correct state estimate using measurement z
		est, err = f.Update(pred.Val(), nil, z)
		if err != nil {
			log.Fatalf("Filter Udpate error: %v", err)
		}

		fmt.Printf("CORRECTED State:\n%v\n", matrix.Format(est.Val()))

		corrPt := GetDotPos(center, r, est.Val().At(0, 0))
		fmt.Printf("Corr Point: %v\n", corrPt)

		fmt.Println("---------------------------------------------")

		// reset all pixels to 0
		img.SetTo(gocv.Scalar{Val1: 0, Val2: 0, Val3: 0, Val4: 0})
		// draw markers
		DrawMarker(&img, yPt, color.RGBA{0, 255, 0, 0}, 3)
		DrawMarker(&img, measPt, color.RGBA{255, 0, 0, 0}, 3)
		DrawMarker(&img, corrPt, color.RGBA{100, 149, 237, 0}, 3)

		window.IMShow(img)
		if window.WaitKey(int(500)) == 27 {
			fmt.Printf("Shutting down: ESC pressed\n")
			break
		}
	}
}
