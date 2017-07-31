package brain

import (
	"testing"
)

func TestNewBrain(t *testing.T) {
	simpleBrain := New(2, 3, 1)
	if simpleBrain == nil {
		t.Error("Can't create brain!")
	}
	if (simpleBrain.inputsNumber != (2 + 1)) ||
		(simpleBrain.hiddenNeuronsNumber != (3 + 1)) ||
		(simpleBrain.outputsNumber != 1) {

		t.Error("Brain init error!")
	}
}

func TestXor(t *testing.T) {
	// create the XOR representation patter to train the network
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	gobrain := New(2, 2, 1)

	gobrain.Train(patterns, 1000, 0.6, 0.4, false)

	for _, p := range patterns {
		result := gobrain.Process(p[0])
		result[0] += 0.5 // +0.5 for round
		if int(result[0]) != int(p[1][0]) {
			t.Error("Xor error!", int(result[0]), int(p[1][0]))
		}
	}

}
