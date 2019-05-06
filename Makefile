BUILD=go build
CLEAN=go clean
INSTALL=go install
BUILDPATH=./_build
PACKAGES=$(shell go list ./... | grep -v /examples/)
EXAMPLES=$(shell find examples/* -maxdepth 0 -type d -exec basename {} \;)

examples: builddir
	for example in $(EXAMPLES); do \
		go build -o "$(BUILDPATH)/$$example" "examples/$$example/$$example.go"; \
	done

all: dep check test examples

builddir:
	mkdir -p $(BUILDPATH)

install:
	$(INSTALL) ./$(EXDIR)/...

clean:
	rm -rf $(BUILDPATH)
	go clean

godep:
ifneq ($(GO111MODULE),"on")
	echo "Installing Go dep resolver"
	wget -O- https://raw.githubusercontent.com/golang/dep/master/install.sh | sh
endif

dep:
ifeq ($(GO111MODULE),"on")
	go mod vendor
else
	dep ensure -v
endif

check:
	for pkg in ${PACKAGES}; do \
		go vet $$pkg || exit ; \
		golint $$pkg || exit ; \
	done

test:
	for pkg in ${PACKAGES}; do \
		go test -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg || exit; \
	done

.PHONY: clean examples
