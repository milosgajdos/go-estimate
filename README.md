# go-estimate

[![GoDoc](https://godoc.org/github.com/milosgajdos/go-estimate?status.svg)](https://godoc.org/github.com/milosgajdos/go-estimate)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos/go-estimate.svg?branch=master)](https://travis-ci.org/milosgajdos/go-estimate)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos/go-estimate)](https://goreportcard.com/report/github.com/milosgajdos/go-estimate)

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
$ go get github.com/milosgajdos/go-estimate
```

Get dependencies:
```shell
$ make dep
```

Run unit tests:
```shell
$ make test
```
You can find various examples of usage in [go-estimate-examples](https://github.com/milosgajdos/go-estimate-examples).

# TODO

- [ ] [Square Root filter](https://en.wikipedia.org/wiki/Kalman_filter#Square_root_form)
- [ ] [Information Filter](https://en.wikipedia.org/wiki/Kalman_filter#Information_filter)
- [x] [Smoothing](https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
    - [Rauch–Tung–Striebel](https://en.wikipedia.org/wiki/Kalman_filter#Rauch%E2%80%93Tung%E2%80%93Striebel) for both KF and EKF has been implemented in `smooth` package

# Contributing

**YES PLEASE!**
