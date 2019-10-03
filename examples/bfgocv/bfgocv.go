package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"log"
	"math"
	"math/rand"
	"os"

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
	resample int
	// particle count
	particles int
)

func init() {
	flag.IntVar(&resample, "resample", 0, "BF particles resampling rate")
	flag.IntVar(&particles, "particles", 100, "Number of filter particles")
}

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

func AppendToGIF(out *gif.GIF, img *gocv.Mat, delay int) error {
	buf := new(bytes.Buffer)

	goImg, err := img.ToImage()
	if err != nil {
		return err
	}

	if err = gif.Encode(buf, goImg, nil); err != nil {
		return err
	}

	in, err := gif.Decode(buf)
	if err != nil {
		return err
	}

	out.Image = append(out.Image, in.(*image.Paletted))
	out.Delay = append(out.Delay, delay)

	return nil
}

func main() {
	flag.Parse()

	// system dynamics matrices
	A := mat.NewDense(2, 2, []float64{1.0, 1.0, 0.0, 1.0})
	C := mat.NewDense(1, 2, []float64{1.0, 0.0})

	// ship is a model of the system we will simulate
	ship, err := sim.NewBaseModel(A, nil, C, nil)
	if err != nil {
		log.Fatalf("Failed to create ship model: %v", err)
	}

	// initial system state: we simply generate some random numbers
	// Note: there is no system input
	x1, x2 := rand.NormFloat64()*0.1, rand.NormFloat64()*0.1
	var x mat.Vector = mat.NewVecDense(2, []float64{x1, x2})

	// covariance of system state variables
	covX, covY := 1e-4, 1e-4
	// system state covariance
	stateCov := mat.NewSymDense(2, []float64{covX, 0, 0, covY})
	stateNoise, err := noise.NewGaussian([]float64{0.0, 0.0}, stateCov)
	if err != nil {
		log.Fatalf("Failed to create state noise: %v", err)
	}

	// initial condition: initial system state + its covariance
	// Note: initial condition has 0 mean - we center particles around it
	initCond := sim.NewInitCond(x, stateCov)

	// measurement noise used to simulate real life measurements
	measCov := mat.NewSymDense(1, []float64{1e-2})
	measNoise, err := noise.NewGaussian([]float64{0.0}, measCov)
	if err != nil {
		log.Fatalf("Failed to create measurement noise: %v", err)
	}

	// number of BF particles
	p := particles

	// system state error Probability Distribution Function (PDF)
	errPDF, _ := distmv.NewNormal([]float64{0}, measCov, nil)

	// create Bootstrap Filter
	f, err := bf.New(ship, initCond, nil, nil, p, errPDF)
	if err != nil {
		log.Fatalf("Failed to create bootstrap filter: %v", err)
	}

	// z stores real system measurement: y+noise
	z := new(mat.VecDense)

	// initial filter estimate: our initial guess about position of the ship
	//  Note: Our estimate will be off the model a bit
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

	width, height := 500, 500
	// center of angular motion
	center := image.Pt(width/2, height/2)
	// radius of angular motion
	r := float64(width) / 3.0

	////////////////
	// SIMULATION //
	////////////////

	// GoCV simulation environment
	img := gocv.NewMatWithSize(width, height, gocv.MatTypeCV8UC3)
	// create simple window to show the simulation
	window := gocv.NewWindow("Bootstrap Filter")
	// reset all pixels to 0
	img.SetTo(gocv.Scalar{Val1: 0, Val2: 0, Val3: 0, Val4: 0})

	/////////////////////////
	/////// Make a GIF //////
	/////////////////////////

	// output GIF image
	outGif := &gif.GIF{}
	// append first image to outGIF
	if err := AppendToGIF(outGif, &img, 50); err != nil {
		log.Fatalf("Failed to create GIF image: %v", err)
	}

	// resample counter: it gets reset every other run
	rsCount := 0

	for {
		// model propagation i.e. ground truth propagation
		x, err = ship.Propagate(x, nil, nil)
		if err != nil {
			log.Fatalf("Model propagation failed: %v", err)
		}

		// model observation i.e. ground truth observation
		y, err := ship.Observe(x, nil, nil)
		if err != nil {
			log.Fatalf("Model observation failed: %v", err)
		}

		// model coordinates
		modelPt := GetDotPos(center, r, y.At(0, 0))

		// measurement: z = y+noise
		z.AddVec(y, measNoise.Sample())

		// measurement coordinates
		measPt := GetDotPos(center, r, z.At(0, 0))
		measPt.X += int(math.Round(measNoise.Sample().AtVec(0)))
		measPt.Y += int(math.Round(measNoise.Sample().AtVec(0)))

		// propagate particle filters to the next step
		pred, err := f.Predict(est.Val(), nil)
		if err != nil {
			log.Fatalf("Failed to predict next filter state: %v", err)
		}

		// correct state estimate using measurement z
		est, err = f.Update(pred.Val(), nil, z)
		if err != nil {
			log.Fatalf("Failed to update the filter state: %v", err)
		}

		// filter output coordinates i.e. corrected estimate
		filterPt := GetDotPos(center, r, est.Val().At(0, 0))

		// reset all pixels to 0
		img.SetTo(gocv.Scalar{Val1: 0, Val2: 0, Val3: 0, Val4: 0})

		// draw particles in grey-ish color
		particles := f.Particles()
		for j := 0; j < p; j++ {
			partPt := GetDotPos(center, r, particles.At(0, j))
			gocv.Circle(&img, partPt, 1, color.RGBA{220, 220, 220, 0}, 1)
		}

		// draw model (ground truth) point marker
		DrawMarker(&img, modelPt, color.RGBA{0, 255, 0, 0}, 2)
		// draw measurement point marker
		DrawMarker(&img, measPt, color.RGBA{255, 0, 0, 0}, 2)
		// draw filter point marker
		DrawMarker(&img, filterPt, color.RGBA{100, 149, 237, 0}, 3)

		window.IMShow(img)
		if window.WaitKey(int(500)) == 27 {
			fmt.Printf("Shutting down: ESC pressed\n")
			break
		}

		// append image to outGIF
		if err := AppendToGIF(outGif, &img, 50); err != nil {
			log.Fatalf("Failed to create GIF image: %v", err)
		}

		if resample > 0 && rsCount == resample {
			if err := f.Resample(0.0); err != nil {
				log.Fatalf("Failed to resample filter particles: %v", err)
			}
			// reset the resample counter
			rsCount = 0
		}
		// increment resample counter
		rsCount += 1
	}

	fGIF, err := os.Create("out.gif")
	if err != nil {
		log.Fatal(err)
	}
	defer fGIF.Close()

	if err = gif.EncodeAll(fGIF, outGif); err != nil {
		log.Fatal(err)
	}
}
