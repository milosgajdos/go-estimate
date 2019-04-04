# go-estimate

[![GoDoc](https://godoc.org/github.com/milosgajdos83/go-estimate?status.svg)](https://godoc.org/github.com/milosgajdos83/go-estimate)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/go-estimate.svg?branch=master)](https://travis-ci.org/milosgajdos83/go-estimate)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/go-estimate)](https://goreportcard.com/report/github.com/milosgajdos83/go-estimate)
[![codecov](https://codecov.io/gh/milosgajdos83/go-estimate/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/go-estimate)

This package offers a small suite of basic filtering algorithms written in Go. It currently provides the implementations of the following filters and estimators:

* [Bootstrap Filter](https://en.wikipedia.org/wiki/Particle_filter#The_bootstrap_filter) also known as SIR Particle filter
* [Unscented Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter) also known as Sigma-point filter
* [Extended Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Extended_Kalman_filter) also known as Non-linear Kalman Filter
  * [Iterated Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter#Iterated_extended_Kalman_filter)
* [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) also known as Linear Kalman Filter

In addition it provides an implementation of [Rauch–Tung–Striebel](https://en.wikipedia.org/wiki/Kalman_filter#Rauch%E2%80%93Tung%E2%80%93Striebel) smoothing for Kalman filter, which is an optimal Gaussian smoothing algorithm. There are variants for both `LKF` (Linear Kalman Filter) and `EKF` (Extended Kalman Filter) implemented in the `smooth` package. `UKF` smoothing will be implemented in the future.

# Get started

Get the package:
```shell
$ go get -u github.com/milosgajdos83/go-estimate
```

Get dependencies:
```shell
$ make godep && make dep
```

Run unit tests:
```shell
$ make test
```

# Examples

The project provides a few [example](examples) programs which demonstrate basic usage of the `go-estimate` packages.

You can build the examples by running the following command:
```shell
$ make examples
```

This will create a directory called `_build` in your current working directory and places the newly built binaries into it. You can now run the programs by executing any of the binaries from the `_build` directory.

Alternatively, you can also install the examples by either running `go install` for each of the examples or do it all with one command:
```shell
$ make install
```

Most of the examples are static i.e. they generate a plot which shows how filter estimates new values from the noise measurements.

There are however [two](examples/bfgocv) [examples](examples/kfgocv) which use the wonderful [gocv](https://gocv.io/). They allow you to watch the filter live in action.

Example of bootstrap filter in action:

<img src="./examples/bfgocv/bootstrap_filter.gif" alt="Bootstrap filter in action" width="200">

# TODO

- [ ] [Square Root filter](https://en.wikipedia.org/wiki/Kalman_filter#Square_root_form)
- [ ] [Information Filter](https://en.wikipedia.org/wiki/Kalman_filter#Information_filter)
- [x] [Smoothing](https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
    - [Rauch–Tung–Striebel](https://en.wikipedia.org/wiki/Kalman_filter#Rauch%E2%80%93Tung%E2%80%93Striebel) for both KF and EKF has been implemented in `smooth` package

# Contributing

**YES PLEASE!**
