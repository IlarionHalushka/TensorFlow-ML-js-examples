const tf = require('@tensorflow/tfjs-node');

// // Define function
// function predict(input) {
// 	// y = a * x ^ 2 + b * x + c
// 	// More on tf.tidy in the next section
// 	return tf.tidy(() => {
// 		const x = tf.scalar(input);
//
// 		const ax2 = a.mul(x.square());
// 		const bx = b.mul(x);
// 		const y = ax2.add(bx).add(c);
//
// 		return y;
// 	});
// }
//
// // Define constants: y = 2x^2 + 4x + 8
// const a = tf.scalar(2);
// const b = tf.scalar(4);
// const c = tf.scalar(8);
//
// // Predict output for input of 2
// const result = predict(2);
// result.print() // Output: 24

(async () => {
	// Create a simple model.
	const model = tf.sequential();
	model.add(tf.layers.dense({units: 1, inputShape: [1]}));

	// Prepare the model for training: Specify the loss and the optimizer.
	model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

	// Generate some synthetic data for training. (y = 2x - 1)
	const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
	const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

	// Train the model using the data.
	await model.fit(xs, ys, {epochs: 250})
		.then((history) => {
			model.predict(xs).print()
		});
})();
