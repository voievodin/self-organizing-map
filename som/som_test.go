package som_test

import (
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/voievodin/self-organizing-map/som"
)

const (
	dataDir         = "testdata"
	reportDir       = "testreport"
	irisDataDir     = dataDir + "/iris"
	irisDataSetPath = irisDataDir + "/iris.json"
	irisReportDir   = reportDir + "/iris"
	colorsReportDir = reportDir + "/colors"

	irisSetosa     = "Iris-setosa"
	irisVersicolor = "Iris-versicolor"
	irisVirginica  = "Iris-virginica"

	defaultImgWidth  = 300
	defaultImgHeight = 300
)

func TestMain(m *testing.M) {
	rand.Seed(time.Now().UnixNano())
	os.Exit(m.Run())
}

func TestColorsClustering(t *testing.T) {
	xLen, yLen := 30, 30

	dataSet := genRandDataSet(xLen*yLen, 3)
	saveDataSetAsColorsPNG(t, dataSet, xLen, yLen)

	somap := som.New(xLen, yLen)
	somap.Initializer = &som.RandWeightsInitializer{}
	somap.Influence = &som.RadiusReducingConstantInfluenceFunc{Radius: 4}
	somap.Restraint = &som.ExpRestraintFunc{InitialRate: 1}
	somap.Selector = &som.RandSelector{}
	somap.Learn(dataSet, 2000)
	saveSOMAsColorsPNG(t, somap, "30x30_radius_reducing_const_func.png")

	somap.Influence = &som.GaussianInfluenceFunc{InitialWidth: 4}
	somap.Learn(dataSet, 2000)
	saveSOMAsColorsPNG(t, somap, "30x30_gaussian_func.png")
}

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

func genRandDataSet(count, vectorLen int) *som.DataSet {
	ds := &som.DataSet{}
	for i := 0; i < count; i++ {
		vector := make(som.DataVector, vectorLen)
		for j := 0; j < vectorLen; j++ {
			vector[j] = rand.Float64()
		}
		ds.Add(vector)
	}
	return ds
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

func toRGBA(vector []float64) color.RGBA {
	return color.RGBA{
		R: uint8(255 * vector[0]),
		G: uint8(255 * vector[1]),
		B: uint8(255 * vector[2]),
		A: 255,
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

	savePNG(t, img, irisReportDir+"/"+classifier+".png")
}

func saveSOMAsColorsPNG(t *testing.T, somap *som.SOM, filename string) {
	img := createImg(defaultImgWidth, defaultImgHeight, len(somap.Neurons), len(somap.Neurons[0]), func(x, y, xLen, yLen int) color.RGBA {
		return toRGBA(somap.Neurons[x][y].Weights)
	})
	savePNG(t, img, colorsReportDir+"/"+filename)
}

func saveDataSetAsColorsPNG(t *testing.T, ds *som.DataSet, xLen, yLen int) {
	img := createImg(defaultImgWidth, defaultImgHeight, xLen, yLen, func(x, y, xLen, yLen int) color.RGBA {
		return toRGBA(ds.Vectors[x*xLen+y])
	})
	savePNG(t, img, colorsReportDir+"/data-set.png")
}

func savePNG(t *testing.T, img *image.RGBA, filename string) {
	f, err := os.Create(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}

func createImg(imgW, imgH, xLen, yLen int, rgbaProvider func(x, y, xLen, yLen int) color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, imgW, imgH))

	subW := imgW / xLen
	subH := imgH / yLen

	for i := 0; i < xLen; i++ {
		for j := 0; j < yLen; j++ {
			rgba := rgbaProvider(i, j, xLen, yLen)
			fillRect(img, rgba, subW*i, subH*j, subW, subH)
		}
	}

	return img
}

func fillRect(img *image.RGBA, c color.RGBA, x, y, w, h int) {
	for i := 0; i < w; i++ {
		for j := 0; j < h; j++ {
			img.SetRGBA(i+x, j+y, c)
		}
	}
}
