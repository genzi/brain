package brain

import (
	"log"
	"math"
	"math/rand"
	"time"
)

type Brain struct {
	inputsNumber        int
	hiddenNeuronsNumber int
	outputsNumber       int
	inputsValues        []float64
	hiddenValues        []float64
	outputsValues       []float64

	inputWeights             [][]float64
	outputWeights            [][]float64
	inputChangesForMomentum  [][]float64
	outputChangesForMomentum [][]float64
}

func New(InputsNumber, HidenNeuronsNumber, OutputsNumber int) *Brain {
	brain := new(Brain)

	brain.inputsNumber = InputsNumber + 1              // +1 for bias
	brain.hiddenNeuronsNumber = HidenNeuronsNumber + 1 // +1 for bias
	brain.outputsNumber = OutputsNumber

	brain.inputsValues = newVector(brain.inputsNumber, 1.0)
	brain.hiddenValues = newVector(brain.hiddenNeuronsNumber, 1.0)
	brain.outputsValues = newVector(brain.outputsNumber, 1.0)

	brain.inputWeights = newMatrix(brain.inputsNumber, brain.hiddenNeuronsNumber)
	brain.outputWeights = newMatrix(brain.hiddenNeuronsNumber, brain.outputsNumber)

	for i := 0; i < brain.inputsNumber; i++ {
		for j := 0; j < brain.hiddenNeuronsNumber; j++ {
			brain.inputWeights[i][j] = random(-1, 1)
		}
	}

	for i := 0; i < brain.hiddenNeuronsNumber; i++ {
		for j := 0; j < brain.outputsNumber; j++ {
			brain.outputWeights[i][j] = random(-1, 1)
		}
	}

	brain.inputChangesForMomentum = newMatrix(brain.inputsNumber, brain.hiddenNeuronsNumber)
	brain.outputChangesForMomentum = newMatrix(brain.hiddenNeuronsNumber, brain.outputsNumber)

	rand.Seed(time.Now().UTC().UnixNano())

	return brain
}

func (self *Brain) Process(inputs []float64) []float64 {
	if len(inputs) != self.inputsNumber-1 {
		log.Fatalln("Error: wrong number of inputs")
	}

	for i := 0; i < self.inputsNumber-1; i++ {
		self.inputsValues[i] = inputs[i]
	}

	for i := 0; i < self.hiddenNeuronsNumber-1; i++ {
		var sum float64

		for j := 0; j < self.inputsNumber; j++ {
			sum += self.inputsValues[j] * self.inputWeights[j][i]
		}

		self.hiddenValues[i] = sigmoid(sum)
	}

	for i := 0; i < self.outputsNumber; i++ {
		var sum float64
		for j := 0; j < self.hiddenNeuronsNumber; j++ {
			sum += self.hiddenValues[j] * self.outputWeights[j][i]
		}

		self.outputsValues[i] = sigmoid(sum)
	}

	return self.outputsValues
}

func (self *Brain) BackPropagate(targets []float64, lRate, mFactor float64) {
	if len(targets) != self.outputsNumber {
		log.Fatal("Error: wrong number of target values")
	}

	outputDeltas := newVector(self.outputsNumber, 0.0)
	for i := 0; i < self.outputsNumber; i++ {
		outputDeltas[i] = dsigmoid(self.outputsValues[i]) * (targets[i] - self.outputsValues[i])
	}

	hiddenDeltas := newVector(self.hiddenNeuronsNumber, 0.0)
	for i := 0; i < self.hiddenNeuronsNumber; i++ {
		var e float64

		for j := 0; j < self.outputsNumber; j++ {
			e += outputDeltas[j] * self.outputWeights[i][j]
		}

		hiddenDeltas[i] = dsigmoid(self.hiddenValues[i]) * e
	}

	for i := 0; i < self.hiddenNeuronsNumber; i++ {
		for j := 0; j < self.outputsNumber; j++ {
			change := outputDeltas[j] * self.hiddenValues[i]
			self.outputWeights[i][j] = self.outputWeights[i][j] + lRate*change + mFactor*self.outputChangesForMomentum[i][j]
			self.outputChangesForMomentum[i][j] = change
		}
	}

	for i := 0; i < self.inputsNumber; i++ {
		for j := 0; j < self.hiddenNeuronsNumber; j++ {
			change := hiddenDeltas[j] * self.inputsValues[i]
			self.inputWeights[i][j] = self.inputWeights[i][j] + lRate*change + mFactor*self.inputChangesForMomentum[i][j]
			self.inputChangesForMomentum[i][j] = change
		}
	}
}

func (self *Brain) Train(patterns [][][]float64, iterations int, lRate, mFactor float64) {
	for i := 0; i < iterations; i++ {
		for _, p := range patterns {
			self.Process(p[0])
			self.BackPropagate(p[1], lRate, mFactor)
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

//////////////////////////////// UTILS /////////////////////////////////////////

func newVector(I int, fill float64) []float64 {
	v := make([]float64, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func newMatrix(x, y int) [][]float64 {
	matrix := make([][]float64, x)
	for i := 0; i < x; i++ {
		matrix[i] = make([]float64, y)
	}
	return matrix
}

func random(x, y float64) float64 {
	return (y-x)*rand.Float64() + x
}

////////////////////////////////////////////////////////////////////////////////
