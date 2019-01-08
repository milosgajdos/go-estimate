# go-filter

[![GoDoc](https://godoc.org/github.com/milosgajdos83/go-filter?status.svg)](https://godoc.org/github.com/milosgajdos83/go-filter)
[![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://travis-ci.org/milosgajdos83/go-filter.svg?branch=master)](https://travis-ci.org/milosgajdos83/go-filter)
[![Go Report Card](https://goreportcard.com/badge/milosgajdos83/go-filter)](https://goreportcard.com/report/github.com/milosgajdos83/go-filter)
[![codecov](https://codecov.io/gh/milosgajdos83/go-filter/branch/master/graph/badge.svg)](https://codecov.io/gh/milosgajdos83/go-filter)

This package offers a small suite of basic filtering algorithms written in Go. It currently provides the implementations of the following filters and estimators:

* [Bootstrap Filter](https://en.wikipedia.org/wiki/Particle_filter#The_bootstrap_filter) also known as SIR Particle filter
* [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) also known as Linear Kalman Filter
* [Extended Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Extended_Kalman_filter) also known as Non-linear Kalman Filter
* [Iterated Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter#Iterated_extended_Kalman_filter)
* [Unscented Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter) also known as Sigma-point filter

# Get started

Get the package:
```shell
$ go get -u github.com/milosgajdos83/go-filter
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

The package ships few [example](examples) programs which demonstrate the basic usage of the libraries.

You can build the examples by running the following command:
```shell
$ make examples
```

This will create a directory called `_build` in your current working directory and places the newly built binaries into it. You can now run the programs by executing any of the binaries from the `_build` directory.

Alternatively, you can also install the examples by either running `go install` for each of the examples or do it all with one command:
```shell
$ make install
```

# TODO

- [ ] [Square Root filter](https://en.wikipedia.org/wiki/Kalman_filter#Square_root_form)
- [ ] [Information Filter](https://en.wikipedia.org/wiki/Kalman_filter#Information_filter)
- [ ] [Smoothing](https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)

# Contributing

**YES PLEASE!**
