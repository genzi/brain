package brain

import (
	"log"
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
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	gobrain := New(2, 3, 1)

	gobrain.Train(patterns, 100000, 0.2, 0.1)

	for _, p := range patterns {
		result := gobrain.Process(p[0])
		result[0] += 0.5 // +0.5 for round
		if int(result[0]) != int(p[1][0]) {
			t.Error("Xor error!", int(result[0]), int(p[1][0]))
		}
	}

}

func TestAnd(t *testing.T) {
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {1}},
	}

	gobrain := New(2, 3, 1)

	gobrain.Train(patterns, 100000, 0.2, 0.1)

	for _, p := range patterns {
		result := gobrain.Process(p[0])
		result[0] += 0.5 // +0.5 for round
		if int(result[0]) != int(p[1][0]) {
			t.Error("Xor error!", int(result[0]), int(p[1][0]))
		}
	}

}

/////////////////////////////////helping functions//////////////////////////////
func character(chars string) []float64 {
	flt := make([]float64, len(chars))
	for i := 0; i < len(chars); i++ {
		if chars[i] == '#' {
			flt[i] = 1.0
		} else { // if '.'
			flt[i] = 0.0
		}
	}
	return flt
}

func mapletter(letter byte) []float64 {
	if letter == 'a' {
		return []float64{0.1}
	}
	if letter == 'b' {
		return []float64{0.3}
	}
	if letter == 'c' {
		return []float64{0.5}
	}
	return []float64{0.0}
}

////////////////////////////////////////////////////////////////////////////////

func TestRecognizeCharacter(t *testing.T) {
	var (
		a = character(
			".#####." +
				"#.....#" +
				"#.....#" +
				"#######" +
				"#.....#" +
				"#.....#" +
				"#.....#")
		b = character(
			"######." +
				"#.....#" +
				"#.....#" +
				"######." +
				"#.....#" +
				"#.....#" +
				"######.")
		c = character(
			"#######" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"#######")
	)

	gobrain := New(49, 15, 1)

	gobrain.Train([][][]float64{
		{c, mapletter('c')},
		{b, mapletter('b')},
		{a, mapletter('a')},
	}, 1000000, 0.2, 0.1)

	result := gobrain.Process(
		character(
			"######." +
				"#.....#" +
				"#.....#" +
				"#######" +
				"#.....#" +
				"#.....#" +
				"#.....#"))
	log.Println("Should be around 0.1", result)

	result = gobrain.Process(
		character(
			"#######" +
				"#.....#" +
				"#.....#" +
				"######." +
				"#.....#" +
				"#.....#" +
				"######."))
	log.Println("Should be around 0.3", result)

	result = gobrain.Process(
		character(
			"#######" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"##....." +
				"#######"))
	log.Println("Should be around 0.5", result)
}
