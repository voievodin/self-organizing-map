package som

import (
	"math/rand"
	"sort"
)

type DataVector []float64

// DataSet is in-memory collection of data vectors.
type DataSet struct {
	Vectors []DataVector
}

// Add adds vector to this data-set.
func (ds *DataSet) Add(vector DataVector) {
	if len(ds.Vectors) != 0 && ds.Width() != len(vector) {
		panic("data set must contain vectors of the same length")
	}
	ds.Vectors = append(ds.Vectors, vector)
}

// AddRaw adds data vector to this data set, created from the given raw values.
func (ds *DataSet) AddRaw(vector ...float64) {
	ds.Add(DataVector(vector))
}

// Len returns the number of vectors carried by this data set.
func (ds *DataSet) Len() int {
	return len(ds.Vectors)
}

// Width returns the length of a single vector from this data set
// (all data vectors have the same length).
func (ds *DataSet) Width() int {
	if ds.Len() == 0 {
		panic("data set contains no elements")
	}
	return len(ds.Vectors[0])
}

// Shuffle shuffles data vectors in this data set.
func (ds *DataSet) Shuffle() {
	shuffled := make([]DataVector, ds.Len())
	for i, j := range rand.Perm(ds.Len()) {
		shuffled[i] = ds.Vectors[j]
	}
	ds.Vectors = shuffled
}

// Copy copies data set vectors and returns a new instance of data set.
func (ds *DataSet) Copy() *DataSet {
	vectorsCopy := make([]DataVector, ds.Len())
	for i := range ds.Vectors {
		vectorCopy := make(DataVector, len(ds.Vectors[i]))
		copy(vectorCopy, ds.Vectors[i])
		vectorsCopy[i] = vectorCopy
	}
	return &DataSet{Vectors: vectorsCopy}
}

// Sort sorts this data set in ascending order.
// Vector A < Vector B, when A[k] < B[k] for the first met such k, where k [0 -> len(A)-1]
func (ds *DataSet) Sort() {
	sort.Slice(ds.Vectors, func(i, j int) bool {
		for k := 0; k < ds.Width(); k++ {
			if ds.Vectors[i][k] != ds.Vectors[j][k] {
				return ds.Vectors[i][k] < ds.Vectors[j][k]
			}
		}
		return false
	})
}

// Reduce reduces the size of this data set,
// divides data set on newLen segments, leaves those vectors
// which indexes are in the middle of each divided segment.
func (ds *DataSet) Reduce(newLen int) {
	if ds.Len() > newLen {
		step := float64(ds.Len()) / float64(newLen)
		vectors := make([]DataVector, newLen)
		for i := 0; i < newLen; i++ {
			left := int(float64(i) * step)
			right := int(float64(i+1) * step)
			vectors[i] = ds.Vectors[(left+right)>>1]
		}
		ds.Vectors = vectors
	}
}
