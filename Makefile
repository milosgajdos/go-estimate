BUILD=go build
CLEAN=go clean
BUILDPATH=./_build
GO111MODULE=on
PACKAGES=$(shell go list ./... | grep -v /examples/)

all: dep check test

builddir:
	mkdir -p $(BUILDPATH)

clean:
	rm -rf $(BUILDPATH)
	go clean

dep:
	go get ./...

check:
	go vet ./...

test:
	for pkg in ${PACKAGES}; do \
		go test -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg || exit; \
	done

.PHONY: clean
