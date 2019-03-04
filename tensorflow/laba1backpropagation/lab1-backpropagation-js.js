import * as tf from "@tensorflow/tfjs"
import "@tensorflow/tfjs-node"
import dataTesting from "./dataTesting"
import dataTraining from "./dataTraining"

// VIDEO ON YOUTUBE similar to this code but iris https://www.youtube.com/watch?v=XdErOpUzupY
// google machine learning course https://codelabs.developers.google.com/codelabs/neural-tensorflow-js/index.html?index=..%2F..index#5

// function: Math.LN10(Math.abs(Math.cos(x1)) + Math.tan(x2) - (1 / Math.tan(x3) ))
// d1 = ln(|cos(x1)|)+tg(x2)-ctg(x3)

// 5,7,8

// const x1initial = 5;
// const x2initial = 7;
// const x3initial = 8;
//
// const output = [];
// const x1x2x3training = [];
//
// // const modelingFunction = (x1, x2, x3) => Math.log(Math.abs(Math.cos(x1)) + Math.tan(x2) - (1 / Math.tan(x3) ));
// // const modelingFunction = (x1, x2, x3) => Math.log(Math.abs(Math.cos(x1)) + Math.tan(x2));
// const modelingFunction = (x1, x2, x3) => x1 + x2 + x3;
//
// for (let i = 0; i < 1000; i++) {
//   output.push(modelingFunction(x1initial + i, x2initial + i, x3initial + i));
//   x1x2x3training.push([x1initial + i, x2initial + i, x3initial + i]);
// }
//
// console.log(output.toString())

const trainingData = tf.tensor2d(dataTraining.inputs);
const outputData = tf.tensor1d(dataTraining.outputs);
const testingData = tf.tensor2d(dataTesting.inputs);
const testingOutputData = tf.tensor1d(dataTesting.outputs);

// build neural network
(async () => {
  const model = await tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [3],
    units: 3,
  }));
  await model.add(tf.layers.dense({
    inputShape: [3],
    activation: "relu",
    units: 1,
  }));

  await model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(0.5),
  });

// // train/fit our network
  await model.fit(trainingData, outputData, {epochs: 25})
    .then((history) => {
      model.predict(testingData).print()
    });
})();

