// Package som provides experimental implementation of Self-Organizing Map.
// See https://en.wikipedia.org/wiki/Self-organizing_map.
//
// SOM - Self-Organizing Map
// BMU - Best Matching Unit
package som

import (
	"errors"
	"math"
	"math/rand"
)

var (
	// ErrNoDataLeft is returned by selector when there is
	// nothing to select from the corresponding data set.
	ErrNoDataLeft = errors.New("no data left")
)

// RestraintFunc calculates learning restraint coefficient
// based on current iteration and overall iterations number.
// The coefficient value indicates how much neurons weights
// will change at corresponding iteration.
type RestraintFunc interface {
	// currentIt => [0, iterationsNumber)
	Apply(currentIt, iterationsNumber int) float64
}

// InfluenceFunc calculates the coefficient which indicates how much
// the weights of each neuron will be changed according to the BMU position.
type InfluenceFunc interface {
	// currentIt => [0, iterationsNumber)
	Apply(bmu *Neuron, currentIt, iterationsNumber, x, y int) float64
}

// DistanceFunc calculates Distance between two points
// represented as float vectors.
type DistanceFunc interface {
	Apply(xVector, yVector []float64) float64
}

// Selector is an interface for selecting data vectors
// from the data set in implementation specific manner.
type Selector interface {
	// Init initializes this Selector with the data set.
	Init(set *DataSet)

	// Next returns the next data vector from the data set,
	// or an error if there are no data vectors left.
	Next() (DataVector, error)
}

// NeuronsInitializer initializes neurons, for example sets
// initial values of weights. Called within Learn func before anything else.
type NeuronsInitializer interface {
	Init(set *DataSet, neurons [][]*Neuron)
}

// ProgressMonitor handler for monitoring learning progress.
type ProgressMonitor interface {
	// ItCompleted called each time iteration completes.
	// Iteration is within bounds [1, itNum].
	ItCompleted(it, itNum int, som *SOM)
}

// DataAdapter adapts a data vector in implementation specific manner.
type DataAdapter interface {
	Adapt(vector []float64) []float64
}

// DataAdapterFunc is an adapter that allows to use
// regular functions as DataAdapters.
type DataAdapterFunc func(vector []float64) []float64

func (f DataAdapterFunc) Adapt(vector []float64) []float64 { return f(vector) }

// Neuron is a build unit in SOM.
// One neuron manages number of weights equal to the number of input vector elements(data set width).
// Each neuron is indexed and has its unique place in a map.
type Neuron struct {
	Weights  []float64
	Distance float64
	X, Y     int
}

// New creates new 2 dimensional X*Y size SOM.
func New(X, Y int) *SOM {
	neurons := make([][]*Neuron, X)
	for i := 0; i < X; i++ {
		neurons[i] = make([]*Neuron, Y)
		for j := 0; j < Y; j++ {
			neurons[i][j] = &Neuron{X: i, Y: j}
		}
	}

	return &SOM{
		Neurons:       neurons,
		Initializer:   &ZeroValueWeightsInitializer{},
		Selector:      &SequentialSelector{},
		Restraint:     &NoRestraintFunc{},
		Influence:     &BMUOnlyInfluencedFunc{},
		Distance:      &EuclideanDistanceFunc{},
		Monitor:       &NoOpProgressMonitor{},
		InDataAdapter: &NoOpAdapter{},
	}
}

// SOM is a map itself.
// Currently it carries double dimension array of neurons,
// provides ability to teach the map and then use results.
type SOM struct {
	Neurons [][]*Neuron

	Initializer   NeuronsInitializer
	Selector      Selector
	Restraint     RestraintFunc
	Influence     InfluenceFunc
	Distance      DistanceFunc
	Monitor       ProgressMonitor
	InDataAdapter DataAdapter
}

// Learn does learning of this SOM from the given data set,
// making as many iterations as iterationsNumber value is.
func (som *SOM) Learn(set *DataSet, iterationsNumber int) {
	som.Initializer.Init(set, som.Neurons)
	som.Selector.Init(set)
	for it := 0; it < iterationsNumber; it++ {
		vector, err := som.Selector.Next()
		if err != nil {
			break
		}
		vector = som.InDataAdapter.Adapt(vector)

		som.computeDistance(vector)
		bmu := som.findBMU()
		som.fixWeights(it, iterationsNumber, bmu, vector)

		som.Monitor.ItCompleted(it+1, iterationsNumber, som)
	}
}

// LearnEntire does learning of this SOM from the given
// data set, making as many iterations as data set length is.
func (som *SOM) LearnEntire(dataSet *DataSet) {
	som.Learn(dataSet, dataSet.Len())
}

// Test finds BMU (Neuron) and returns it.
// Note that this func DOES CHANGE the values of neuron.Distance props,
// so they become equal to the distance between the given vector
// and corresponding neurons.
func (som *SOM) Test(vector DataVector) *Neuron {
	som.computeDistance(som.InDataAdapter.Adapt(vector))
	return som.findBMU()
}

// ComputeDistanceMatrix computes distance from the given vector
// to each neuron and returns a matrix of such values.
// The value at position (x, y) is a distance to the neuron at position (x, y).
// Note that this func:
//   - DOES NOT CHANGE the values of neuron.Distance props;
//   - ADAPTS input vector using som.InDataAdapter.
func (som *SOM) ComputeDistanceMatrix(vector DataVector) [][]float64 {
	vector = som.InDataAdapter.Adapt(vector)
	distances := make([][]float64, len(som.Neurons))
	for i := 0; i < len(som.Neurons); i++ {
		distances[i] = make([]float64, len(som.Neurons[i]))
		for j := 0; j < len(som.Neurons[i]); j++ {
			distances[i][j] = som.Distance.Apply(vector, som.Neurons[i][j].Weights)
		}
	}
	return distances
}

// SeparateWeights creates and returns N matrices of neurons weights.
// Each matrix in the result describes neurons weights at corresponding
// index position, for example:
//
// for the following matrix of neurons weights:
//    [ [1, 2] [3, 4] ]
//    [ [5, 6] [7, 8] ]
//
// result will be:
//  result[0]:   result[1]:
//     [ 1 3 ]         [ 2 4 ]
//     [ 5 7 ]         [ 6 8 ]
//
func (som *SOM) SeparateWeights() [][][]float64 {
	separations := make([][][]float64, len(som.Neurons[0][0].Weights))
	for si := 0; si < len(separations); si++ {
		separations[si] = make([][]float64, len(som.Neurons))
		for i := 0; i < len(separations[si]); i++ {
			separations[si][i] = make([]float64, len(som.Neurons[i]))
			for j := 0; j < len(separations[si][i]); j++ {
				separations[si][i][j] = som.Neurons[i][j].Weights[si]
			}
		}
	}
	return separations
}

func (som *SOM) computeDistance(vector DataVector) {
	for i := 0; i < len(som.Neurons); i++ {
		for j := 0; j < len(som.Neurons[i]); j++ {
			som.Neurons[i][j].Distance = som.Distance.Apply(vector, som.Neurons[i][j].Weights)
		}
	}
}

func (som *SOM) findBMU() *Neuron {
	bmu := som.Neurons[0][0]
	minDistance := bmu.Distance
	candidatesCount := 1
	for i := 0; i < len(som.Neurons); i++ {
		for j := 0; j < len(som.Neurons[i]); j++ {
			candidate := som.Neurons[i][j]
			if minDistance > candidate.Distance {
				bmu = candidate
				minDistance = bmu.Distance
				candidatesCount = 1
			} else if minDistance == candidate.Distance {
				candidatesCount++
			}
		}
	}

	if candidatesCount == 1 {
		return bmu
	}

	candidates := make([]*Neuron, 0, 2)
	for i := 0; i < len(som.Neurons); i++ {
		for j := 0; j < len(som.Neurons[i]); j++ {
			if minDistance == som.Neurons[i][j].Distance {
				candidates = append(candidates, som.Neurons[i][j])
			}
		}
	}

	return candidates[rand.Intn(len(candidates))]
}

func (som *SOM) fixWeights(t, T int, bmu *Neuron, input DataVector) {
	for i := 0; i < len(som.Neurons); i++ {
		for j := 0; j < len(som.Neurons[i]); j++ {
			neuron := som.Neurons[i][j]
			for k := 0; k < len(neuron.Weights); k++ {
				cof := som.Restraint.Apply(t, T) * som.Influence.Apply(bmu, t, T, i, j)
				neuron.Weights[k] += cof * (input[k] - neuron.Weights[k])
			}
		}
	}
}

type EuclideanDistanceFunc struct{}

func (ed *EuclideanDistanceFunc) Apply(xVector, yVector []float64) float64 {
	var sum float64
	for i := 0; i < len(xVector); i++ {
		sum += math.Pow(xVector[i]-yVector[i], 2)
	}
	return math.Sqrt(sum)
}

// See https://en.wikipedia.org/wiki/Taxicab_geometry.
type ManhattanDistanceFunc struct{}

func (md *ManhattanDistanceFunc) Apply(xVector, yVector []float64) float64 {
	var sum float64
	for i := 0; i < len(xVector); i++ {
		sum += math.Abs(xVector[i]-yVector[i])
	}
	return sum
}

// BMUOnlyInfluencedFunc is implementation of InfluenceFunc which
// allows modification of BMU neuron only.
type BMUOnlyInfluencedFunc struct{}

func (calc *BMUOnlyInfluencedFunc) Apply(bmu *Neuron, currentIt, iterationsNumber, i, j int) float64 {
	if bmu.X == i && bmu.Y == j {
		return 1
	} else {
		return 0
	}
}

// NoRestraintFunc is RestraintFunc implementation which always returns 1,
// thus doesn't effect weights modification at all.
type NoRestraintFunc struct{}

func (rc *NoRestraintFunc) Apply(currentIt, iterationsNumber int) float64 { return 1 }

type SequentialSelector struct {
	set *DataSet
	idx int
}

func (sel *SequentialSelector) Init(set *DataSet) {
	sel.set = set
}

func (sel *SequentialSelector) Next() (DataVector, error) {
	if sel.idx >= sel.set.Len() {
		return nil, ErrNoDataLeft
	}
	vector := sel.set.Vectors[sel.idx]
	sel.idx++
	return vector, nil
}

// RandSelector randomly selects a data vector from the corresponding data set,
// the selection is infinite, thus Next() never returns error. If data set size is X
// then X calls to Next() will return X different random vectors from the data set.
type RandSelector struct {
	dataSet *DataSet
	perm    []int
	idx     int
}

func (sel *RandSelector) Init(dataSet *DataSet) {
	sel.dataSet = dataSet
	sel.perm = rand.Perm(dataSet.Len())
}

func (sel *RandSelector) Next() (DataVector, error) {
	if sel.idx == len(sel.perm) {
		sel.idx = 0
		sel.perm = rand.Perm(sel.dataSet.Len())
	}
	vector := sel.dataSet.Vectors[sel.perm[sel.idx]]
	sel.idx++
	return vector, nil
}

// ZeroValueWeightsInitializer adjusts weight arrays length based on data set width.
type ZeroValueWeightsInitializer struct{}

func (initializer *ZeroValueWeightsInitializer) Init(set *DataSet, neurons [][]*Neuron) {
	inputSize := set.Width()
	for i := 0; i < len(neurons); i++ {
		for j := 0; j < len(neurons[i]); j++ {
			neurons[i][j].Weights = make([]float64, inputSize)
		}
	}
}

// RandWeightsInitializer sets weights values to small [0.0,1.0) random values.
type RandWeightsInitializer struct{}

func (initializer *RandWeightsInitializer) Init(set *DataSet, neurons [][]*Neuron) {
	zeroInitializer := &ZeroValueWeightsInitializer{}
	zeroInitializer.Init(set, neurons)

	for i := 0; i < len(neurons); i++ {
		for j := 0; j < len(neurons[i]); j++ {
			neuron := neurons[i][j]
			for k := 0; k < len(neuron.Weights); k++ {
				neuron.Weights[k] = rand.Float64()
			}
		}
	}
}

// RandDataSetVectorsWeightsInitializer sets weights values to random vectors from data set.
type RandDataSetVectorsWeightsInitializer struct{}

func (initializer *RandDataSetVectorsWeightsInitializer) Init(dataSet *DataSet, neurons [][]*Neuron) {
	zeroInitializer := &ZeroValueWeightsInitializer{}
	zeroInitializer.Init(dataSet, neurons)

	matrixSize := len(neurons) * len(neurons[0])
	if matrixSize < dataSet.Len() {
		dataSet = dataSet.Copy()
		dataSet.Sort()
		dataSet.Reduce(matrixSize)
	}

	selector := &RandSelector{}
	selector.Init(dataSet)

	for i := 0; i < len(neurons); i++ {
		for j := 0; j < len(neurons[i]); j++ {
			neuron := neurons[i][j]
			vector, _ := selector.Next()
			for k := 0; k < len(neuron.Weights); k++ {
				neuron.Weights[k] = vector[k]
			}
		}
	}
}

// RadiusReducingConstantInfluenceFunc influences only neurons in a given radius around BMU.
// Radius is reduced at each iteration, so the influence area becomes smaller,
// but not smaller than r/2, so R >= influence area > R/2.
type RadiusReducingConstantInfluenceFunc struct {
	Radius float64
}

func (influence *RadiusReducingConstantInfluenceFunc) Apply(bmu *Neuron, currentIt, iterationsNumber, x, y int) float64 {
	t := float64(currentIt)
	T := float64(iterationsNumber)
	qt := influence.Radius / (1 + t/T)

	d := math.Sqrt(math.Pow(float64(bmu.X-x), 2) + math.Pow(float64(bmu.Y-y), 2))

	if d > qt {
		return 0
	} else {
		return 1
	}
}

// Calculates influence coefficient g(t) using gaussian function
// with exp decay function to reduce neighbourhood width.
// The calculation is done in the following way:
// g(t) = exp( - d*d/(2*q(t)*q(t)) )
// q(t) = InitialWidth * exp( -currentIt/iterationsNumber )
// d - distance from the BMU to the neuron at position (x, y)
type GaussianExpDecayInfluenceFunc struct {
	// InitialWidth is the initial width of the neighbourhood.
	InitialWidth float64
}

func (f *GaussianExpDecayInfluenceFunc) Apply(bmu *Neuron, currentIt, iterationsNumber, x, y int) float64 {
	xx := float64(bmu.X - x)
	yy := float64(bmu.Y - y)
	d := math.Sqrt(xx*xx + yy*yy)
	q := f.InitialWidth * math.Exp(-float64(currentIt)/float64(iterationsNumber))
	return math.Exp(-(d * d) / (2 * q * q))
}

// GaussianInfluenceFunc calculates influence coefficient g(t) using gaussian function
// with custom neighbourhood function.
// g(t) = exp( -d**2/ (2*q(t)**2) )
// where q(T) - is neighbourhood function
// where d is euclidean distance from the BMU to [i][j] neuron
type GaussianInfluenceFunc struct {
	// Q - neighbourhood function.
	// currentIt => [currentIt, iterationsNumber)
	Q func(currentIt, iterationsNumber int) float64
}

func (f *GaussianInfluenceFunc) Apply(bmu *Neuron, currentIt, iterationsNumber, x, y int) float64 {
	xx := float64(bmu.X - x)
	yy := float64(bmu.Y - y)
	d := math.Sqrt(xx*xx + yy*yy)
	q := f.Q(currentIt, iterationsNumber)
	return math.Exp(-(d * d) / (2 * q * q))
}

// SimpleRestraintFunc calculates coefficient as => A / (B + t).
type SimpleRestraintFunc struct {
	A, B float64
}

func (rc *SimpleRestraintFunc) Apply(currentIt, iterationsNumber int) float64 {
	return rc.A / (rc.B + float64(currentIt))
}

// ExpRestraintFunc calculates coefficient as => InitialRate * exp(-t/N),
// if N is <= 0 (not set) then iterationsNumber will be used.
type ExpRestraintFunc struct {
	InitialRate, N float64
}

func (erf *ExpRestraintFunc) Apply(currentIt, iterationsNumber int) float64 {
	t := float64(currentIt)
	var denominator float64
	if erf.N <= 0 {
		denominator = float64(iterationsNumber)
	} else {
		denominator = erf.N
	}
	return erf.InitialRate * math.Exp(-t/denominator)
}

// NoOpProgressMonitor is a default implementation of ProgressMonitor, does nothing.
type NoOpProgressMonitor struct{}

func (pm *NoOpProgressMonitor) ItCompleted(it, itNum int, som *SOM) {}

// NoOpAdapter is an implementation of DataAdapter which returns
// input vector without any modifications.
type NoOpAdapter struct{}

func (adapter *NoOpAdapter) Adapt(vector []float64) []float64 {
	return vector
}

func NewScalingDataAdapter(min, max []float64) *ScalingDataAdapter {
	maxMinDiff := make([]float64, len(min))
	for i := range min {
		maxMinDiff[i] = max[i] - min[i]
	}
	return &ScalingDataAdapter{Min: min, MaxMinDiff: maxMinDiff}
}

// ScalingDataAdapter scales input vector values to be in range [0, 1].
// Note that the original vector is modified.
type ScalingDataAdapter struct {
	Min, MaxMinDiff []float64
}

func (adapter *ScalingDataAdapter) Adapt(vector []float64) []float64 {
	for i := range vector {
		vector[i] -= adapter.Min[i]
		vector[i] /= adapter.MaxMinDiff[i]
	}
	return vector
}
