package som_test

import (
	"encoding/json"
	"image"
	"image/color"
	"io"
	"os"
	"testing"

	"github.com/voievodin/self-organizing-map/som"
)

const (
	irisSetosa     = "Iris-setosa"
	irisVersicolor = "Iris-versicolor"
	irisVirginica  = "Iris-virginica"

	irisDataDir     = dataDir + "/iris"
	irisDataSetPath = irisDataDir + "/iris.json"
	irisReportDir   = reportDir + "/iris"
)

func TestIrisesClustering(t *testing.T) {
	irises := readIrisData(t)
	ds := &som.DataSet{}
	for _, iris := range irises {
		ds.Add(iris.toDataVector())
	}
	ds.Shuffle()

	somap := som.New(10, 10)
	somap.Initializer = &som.RandWeightsInitializer{}
	somap.Influence = &som.RadiusReducingConstantInfluenceFunc{Radius: 3}
	somap.Learn(ds, ds.Len())

	compareDispersion(t, irises, somap, "sl-sw", func(iris iris) (float64, float64) {
		return iris.SepalLength, iris.SepalWidth
	})
	compareDispersion(t, irises, somap, "pl-pw", func(iris iris) (float64, float64) {
		return iris.PetalLength, iris.PetalWidth
	})
	compareDispersion(t, irises, somap, "pl-sw", func(iris iris) (float64, float64) {
		return iris.PetalLength, iris.SepalWidth
	})
	compareDispersion(t, irises, somap, "pl-sl", func(iris iris) (float64, float64) {
		return iris.PetalLength, iris.SepalLength
	})
	compareDispersion(t, irises, somap, "pw-sw", func(iris iris) (float64, float64) {
		return iris.PetalWidth, iris.SepalWidth
	})
	compareDispersion(t, irises, somap, "pw-sl", func(iris iris) (float64, float64) {
		return iris.PetalWidth, iris.SepalLength
	})
}

func compareDispersion(t *testing.T, irises []iris, somap *som.SOM, classifier string, xyExt func(iris) (float64, float64)) {
	imgW := 200
	imgH := 100

	colors := map[string]color.RGBA{
		irisSetosa:     {G: 0xff, A: 0xff},
		irisVersicolor: {R: 0xff, A: 0xff},
		irisVirginica:  {B: 0xff, A: 0xff},
		"som":          {R: 0xff, B: 0xff, A: 0xff}, //pink
	}

	img := image.NewRGBA(image.Rect(0, 0, imgW, imgH))

	// fill white
	for i := 0; i < imgW; i++ {
		for j := 0; j < imgH; j++ {
			img.Set(i, j, color.White)
		}
	}

	// draw irises
	for _, iris := range irises {
		x, y := xyExt(iris)
		img.SetRGBA(int(x*10), int(y*10), colors[iris.Name])
	}

	// draw som (+imgW x offset)
	for i := 0; i < len(somap.Neurons); i++ {
		for j := 0; j < len(somap.Neurons[i]); j++ {
			x, y := xyExt(toIris(somap.Neurons[i][j].Weights))
			img.SetRGBA(int(x*10)+100, int(y*10), colors["som"])
		}
	}

	savePNG(t, img, pngpath(t, irisReportDir, classifier))
}

type iris struct {
	SepalLength float64
	SepalWidth  float64
	PetalLength float64
	PetalWidth  float64
	Name        string
}

func (ir *iris) toDataVector() som.DataVector {
	return som.DataVector{
		ir.SepalLength,
		ir.SepalWidth,
		ir.PetalLength,
		ir.PetalWidth,
	}
}

func toIris(vector []float64) iris {
	return iris{
		SepalLength: vector[0],
		SepalWidth:  vector[1],
		PetalLength: vector[2],
		PetalWidth:  vector[3],
	}
}

func readIrisData(t *testing.T) []iris {
	f, err := os.Open(irisDataSetPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	irises := make([]iris, 0)
	decoder := json.NewDecoder(f)
	for {
		iris := iris{}
		if err := decoder.Decode(&iris); err == nil {
			irises = append(irises, iris)
		} else if err != io.EOF {
			t.Fatal(err)
		} else {
			break
		}
	}
	return irises
}
