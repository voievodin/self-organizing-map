package som_test

import (
	"bytes"
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/voievodin/self-organizing-map/som"
)

const (
	dataDir   = "testdata"
	reportDir = "testreport"
)

func TestMain(m *testing.M) {
	rand.Seed(time.Now().UnixNano())
	os.Exit(m.Run())
}

func TestRandSelectorDoesNotSelectTheSameVectorTwice(t *testing.T) {
	dataSet := &som.DataSet{}
	for i := 0; i < 100; i++ {
		dataSet.AddRaw(float64(i))
	}

	selector := &som.RandSelector{}
	selector.Init(dataSet)

	selected := make([]int, dataSet.Len())
	for i := 0; i < dataSet.Len(); i++ {
		vector, _ := selector.Next()
		selected[int(vector[0])]++
	}

	for i := 0; i < len(selected); i++ {
		if selected[i] != 1 {
			t.Fatal("All the elements from the data set must be selected")
		}
	}
}

func TestRandDataSetVectorsWeightsInitializer(t *testing.T) {
	dataSet := &som.DataSet{}
	for i := 0; i < 100; i++ {
		dataSet.AddRaw(rand.Float64(), rand.Float64(), rand.Float64())
	}

	somap := som.New(5, 5)

	initializer := &som.RandDataSetVectorsWeightsInitializer{}
	initializer.Init(dataSet, somap.Neurons)

	for i := 0; i < len(somap.Neurons); i++ {
		for j := 0; j < len(somap.Neurons[i]); j++ {
			neuron := somap.Neurons[i][j]
			found := false
			for k := 0; k < dataSet.Len() && !found; k++ {
				found = reflect.DeepEqual(dataSet.Vectors[k], som.DataVector(neuron.Weights))
			}
			if !found {
				t.Fatal("Weights values must be initialized from data set vectors")
			}
		}
	}
}

func TestSOMComputesDistanceMatrix(t *testing.T) {
	dataSet := &som.DataSet{Vectors: []som.DataVector{{0.1, 0.2, 0.3}, {0.9, 0.8, 0.7}}}

	somap := som.New(5, 5)
	somap.Initializer = &som.RandWeightsInitializer{}
	somap.LearnEntire(dataSet)

	vector := som.DataVector{0.4, 0.5, 0.6}

	// test vector before, so values of neuron.Distance are set
	_ = somap.Test(vector)
	distances := somap.ComputeDistanceMatrix(vector)

	for i := 0; i < len(distances); i++ {
		for j := 0; j < len(distances[i]); j++ {
			if distances[i][j] != somap.Neurons[i][j].Distance {
				t.Fatalf(
					"The distances at position (%d, %d) %f != %f (neuron distance)",
					i,
					j,
					distances[i][j],
					somap.Neurons[i][j].Distance,
				)
			}
		}
	}
}

func TestSOMSeparatesWeights(t *testing.T) {
	dataSet := &som.DataSet{Vectors: []som.DataVector{{0.1, 0.2, 0.3}}}

	somap := som.New(5, 5)
	somap.Initializer = &som.RandWeightsInitializer{}
	somap.LearnEntire(dataSet)

	separations := somap.SeparateWeights()

	for i := 0; i < len(somap.Neurons); i++ {
		for j := 0; j < len(somap.Neurons[i]); j++ {
			for k := 0; k < len(somap.Neurons[i][j].Weights); k++ {
				if separations[k][i][j] != somap.Neurons[i][j].Weights[k] {
					t.Fatalf(
						"Wrong snapshot values separations[%d][%d][%d] != neuron[%d][%d].weights[%d]",
						k, i, j, i, j, k,
					)

				}
			}
		}
	}
}

func TestSOMGobSerialization(t *testing.T) {
	dataSet := &som.DataSet{Vectors: []som.DataVector{{0.1, 0.2, 0.3}}}

	somap := som.New(5, 5)
	somap.Initializer = &som.RandWeightsInitializer{}
	somap.LearnEntire(dataSet)

	buf := &bytes.Buffer{}
	encoder := gob.NewEncoder(buf)
	decoder := gob.NewDecoder(buf)

	if err := encoder.Encode(&som.SOM{Neurons: somap.Neurons}); err != nil {
		t.Fatal(err)
	}

	decodedSOM := &som.SOM{}
	if err := decoder.Decode(decodedSOM); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < len(somap.Neurons); i++ {
		for j := 0; j < len(somap.Neurons[i]); j++ {
			actual := somap.Neurons[i][j]
			decoded := decodedSOM.Neurons[i][j]
			if !reflect.DeepEqual(actual, decoded) {
				t.Fatalf("Expected neurons to be equal but %v != %v", actual, decoded)
			}
		}
	}
}

func TestInDataAdapterIsAppliedWhileLearning(t *testing.T) {
	dataSet := &som.DataSet{Vectors: []som.DataVector{{1}}}

	somap := som.New(1, 1)
	somap.Initializer = &som.RandDataSetVectorsWeightsInitializer{}
	somap.InDataAdapter = som.DataAdapterFunc(func(vector []float64) []float64 {
		return []float64{5}
	})
	somap.LearnEntire(dataSet)

	if somap.Neurons[0][0].Weights[0] != 5 {
		t.Fatalf("Expected weights[0] to be 5, but it is %f", somap.Neurons[0][0].Weights[0])
	}
}

func TestInDataAdapterIsAppliedWhileTesting(t *testing.T) {
	dataSet := &som.DataSet{Vectors: []som.DataVector{{1, 2, 3}}}

	somap := som.New(1, 1)
	somap.LearnEntire(dataSet)

	neuron := somap.Test(som.DataVector{100, 100, 100})
	if neuron.Distance == 0 {
		t.Fatalf("Expected distance to be different from 0")
	}

	somap.InDataAdapter = som.DataAdapterFunc(func(vector []float64) []float64 {
		return dataSet.Vectors[0]
	})

	neuron = somap.Test(som.DataVector{100, 100, 100})
	if neuron.Distance != 0 {
		t.Fatalf("Adapation was not applied, expected distance to be 0, but it is %f", neuron.Distance)
	}
}

func TestInDataAdapterIsAppliedWhileComputingDistanceMatrix(t *testing.T) {
	dataSet := &som.DataSet{Vectors: []som.DataVector{{1}}}

	somap := som.New(1, 1)
	somap.LearnEntire(dataSet)

	distance := somap.ComputeDistanceMatrix(som.DataVector{5})[0][0]
	if distance != 4 {
		t.Fatalf("Expected distance to be 4, but it is %f", distance)
	}
}

func TestScalingDataAdapterAdaptsValues(t *testing.T) {
	cases := []struct {
		min, max, vector, expected []float64
	}{
		{
			min:      []float64{0},
			max:      []float64{10},
			vector:   []float64{5},
			expected: []float64{0.5},
		},
		{
			min:      []float64{10},
			max:      []float64{20},
			vector:   []float64{12},
			expected: []float64{0.2},
		},
		{
			min:      []float64{-50},
			max:      []float64{50},
			vector:   []float64{0},
			expected: []float64{0.5},
		},
		{
			min:      []float64{0, 0, 0},
			max:      []float64{10, 20, 40},
			vector:   []float64{10, 10, 10},
			expected: []float64{1, 0.5, 0.25},
		},
	}

	for _, aCase := range cases {
		adapter := som.NewScalingDataAdapter(aCase.min, aCase.max)

		adapted := adapter.Adapt(aCase.vector)
		if !reflect.DeepEqual(adapted, aCase.expected) {
			t.Fatalf("Expected %v != actual %v", adapter, aCase.expected)
		}
	}
}

func TestNeuronsAreOnTheRightPositions(t *testing.T) {
	N, M := 15, 7
	sm := som.New(N, M)
	for x := 0; x < N; x++ {
		for y := 0; y < M; y++ {
			neuron := sm.Neurons[x][y]
			if neuron.X != x || neuron.Y != y {
				t.Fatalf("Expected neuron to be on (%d, %d) position, but it is on (%d, %d)", x, y, neuron.X, neuron.Y)
			}
		}
	}
}

func TestChebyshevDistanceFunc(t *testing.T) {
	f := som.ChebyshevDistanceFunc{}

	distance := f.Apply([]float64{1, 4}, []float64{2, 4.5})
	if 1 != distance {
		t.Fatalf("Wrong distance '%f', expected '%f'", distance, 1.0)
	}
}

func BenchmarkDistanceCalculationUsingMathPow(b *testing.B) {
	// simulating the case with neuron in the influence functions
	neuron := &som.Neuron{X: 10, Y: 10}
	x, y := 5, 5

	for i := 0; i < b.N; i++ {
		_ = math.Sqrt(math.Pow(float64(neuron.X-x), 2) + math.Pow(float64(neuron.Y-y), 2))
	}
}

func BenchmarkDistanceCalculationUsingMultiplication(b *testing.B) {
	// simulating the case with neuron in the influence functions
	neuron := &som.Neuron{X: 10, Y: 10}
	x, y := 5, 5

	for i := 0; i < b.N; i++ {
		xx := float64(neuron.X - x)
		yy := float64(neuron.Y - y)
		_ = math.Sqrt(xx*xx + yy*yy)
	}
}
