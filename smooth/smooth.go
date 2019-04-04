package smooth

import filter "github.com/milosgajdos83/go-estimate"

// RauchTungStriebel is optimal filter smoother
type RTS interface {
	// filter.Smoother is filter smoother
	filter.Smoother
}
