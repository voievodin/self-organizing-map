package som_test

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/voievodin/self-organizing-map/som"
)

const (
	colorsReportDir = reportDir + "/colors"

	defaultImgWidth  = 300
	defaultImgHeight = 300
)

type rgbaExtractor func(x, y, xLen, yLen int) color.RGBA

func TestColorsClusteringUsingConstantInfluenceRadius4AndExpRestraintRate1(t *testing.T) {
	xLen, yLen := 30, 30

	dataSet := genRandDataSet(xLen*yLen, 3)
	dataSetImg := createImg(xLen, yLen, dataSetRGBAExtractor(dataSet))
	savePNG(t, dataSetImg, pngpath(t, colorsReportDir, "data-set"))

	somap := som.New(xLen, yLen)
	somap.Initializer = &som.RandWeightsInitializer{}
	somap.Influence = &som.RadiusReducingConstantInfluenceFunc{Radius: 4}
	somap.Restraint = &som.ExpRestraintFunc{InitialRate: 1}
	somap.Selector = &som.RandSelector{}
	somap.Learn(dataSet, 2000)

	somImg := createImg(xLen, yLen, somRGBAExtractor(somap))
	savePNG(t, somImg, pngpath(t, colorsReportDir, "30x30_radius_reducing_const_func"))
}

func TestColorsClusteringUsingGaussianInfluenceWidth4AndExpRestraintRate1(t *testing.T) {
	xLen, yLen := 30, 30

	dataSet := genRandDataSet(xLen*yLen, 3)
	dataSetImg := createImg(xLen, yLen, dataSetRGBAExtractor(dataSet))
	savePNG(t, dataSetImg, pngpath(t, colorsReportDir, "data-set"))

	somap := som.New(xLen, yLen)
	somap.Initializer = &som.RandWeightsInitializer{}
	somap.Influence = &som.GaussianInfluenceFunc{InitialWidth: 4}
	somap.Restraint = &som.ExpRestraintFunc{InitialRate: 1}
	somap.Selector = &som.RandSelector{}
	somap.Learn(dataSet, 2000)

	somImg := createImg(xLen, yLen, somRGBAExtractor(somap))
	savePNG(t, somImg, pngpath(t, colorsReportDir, "30x30_gaussian_func"))
}

func pngpath(t *testing.T, repDir, name string) string {
	return fmt.Sprintf("%s%c%s%c%s.png", repDir, filepath.Separator, t.Name(), filepath.Separator, name)
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

func dataSetRGBAExtractor(ds *som.DataSet) rgbaExtractor {
	return func(x, y, xLen, yLen int) color.RGBA {
		return toRGBA(ds.Vectors[x*xLen+y])
	}
}

func somRGBAExtractor(sm *som.SOM) rgbaExtractor {
	return func(x, y, xLen, yLen int) color.RGBA {
		return toRGBA(sm.Neurons[x][y].Weights)
	}
}

func savePNG(t *testing.T, img *image.RGBA, filename string) {
	err := os.MkdirAll(filepath.Dir(filename), os.ModePerm)
	if err != nil && err != os.ErrExist {
		t.Fatal(err)
	}
	f, err := os.Create(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}

func createImg(xLen, yLen int, extractor rgbaExtractor) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, defaultImgWidth, defaultImgHeight))

	subW := defaultImgWidth / xLen
	subH := defaultImgHeight / yLen

	for i := 0; i < xLen; i++ {
		for j := 0; j < yLen; j++ {
			rgba := extractor(i, j, xLen, yLen)
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

func toRGBA(vector []float64) color.RGBA {
	return color.RGBA{
		R: uint8(255 * vector[0]),
		G: uint8(255 * vector[1]),
		B: uint8(255 * vector[2]),
		A: 255,
	}
}
