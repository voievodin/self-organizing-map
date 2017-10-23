package som_test

import (
	"testing"

	"github.com/voievodin/self-organizing-map/som"
)

func TestDataSetReduce(t *testing.T) {
	dataSet := &som.DataSet{}
	for i := 0; i < 9; i++ {
		dataSet.AddRaw(float64(i))
	}

	dataSet.Reduce(3)

	// 0 1 2 3 4 5 6 7 8  (len = 9)
	// * ^   * ^   * ^   *
	//
	// [0] -> (0 + 3) / 2 = int(1.5) = 1
	// [1] -> (3 + 6) / 2 = int(4.5) = 4
	// [2] -> (6 + 9) / 2 = int(7.5) = 7
	assertEq(t, dataSet.Vectors[0][0], 1.0)
	assertEq(t, dataSet.Vectors[1][0], 4.0)
	assertEq(t, dataSet.Vectors[2][0], 7.0)
}

func assertEq(t *testing.T, a, b interface{}) {
	if a != b {
		t.Fatalf("Expected elements to be equals, but %T% v != %T %v", a, a, b, b)
	}
}
