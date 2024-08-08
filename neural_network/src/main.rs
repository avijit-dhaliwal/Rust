use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;

struct NeuralNetwork {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        NeuralNetwork {
            weights: Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen::<f64>() - 0.5),
            bias: Array1::from_shape_fn(hidden_size, |_| rng.gen::<f64>() - 0.5),
        }
    }

    fn forward(&self, input: ArrayView1<f64>) -> f64 {
        let hidden = self.weights.dot(&input) + &self.bias;
        hidden.mapv(|x| 1.0 / (1.0 + (-x).exp())).sum()
    }

    fn train(&mut self, inputs: &Array2<f64>, targets: &Array1<f64>, learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for (input, &target) in inputs.rows().into_iter().zip(targets.iter()) {
                let output = self.forward(input);
                let error = target - output;
                let gradient = input.mapv(|x| x * error * output * (1.0 - output));
                for (mut weight_row, grad) in self.weights.rows_mut().into_iter().zip(gradient.iter()) {
                    weight_row += &(learning_rate * grad * &input);
                }
                self.bias += learning_rate * error;
            }
        }
    }
}

fn main() {
    let inputs = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let targets = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

    let mut nn = NeuralNetwork::new(2, 4);
    nn.train(&inputs, &targets, 0.1, 1000);

    for input in inputs.rows() {
        println!("Input: {:?}, Prediction: {:.4}", input, nn.forward(input));
    }
}