package som

import "math/rand"

type DataVector []float64

type DataSet struct {
	Vectors []DataVector
}

func (ds *DataSet) Add(vector DataVector) {
	if len(ds.Vectors) != 0 && ds.Width() != len(vector) {
		panic("data set must contain vectors of the same length")
	}
	ds.Vectors = append(ds.Vectors, vector)
}

func (ds *DataSet) AddRaw(vector ...float64) {
	ds.Add(DataVector(vector))
}

func (ds *DataSet) Len() int {
	return len(ds.Vectors)
}

func (ds *DataSet) Width() int {
	if ds.Len() == 0 {
		panic("data set contains no elements")
	}
	return len(ds.Vectors[0])
}

func (ds *DataSet) Shuffle() {
	shuffled := make([]DataVector, ds.Len())
	for i, j := range rand.Perm(ds.Len()) {
		shuffled[i] = ds.Vectors[j]
	}
	ds.Vectors = shuffled
}
