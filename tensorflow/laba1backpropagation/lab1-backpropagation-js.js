import * as tf from "@tensorflow/tfjs"
import "@tensorflow/tfjs-node"
import dataTesting from "./dataTesting"
import dataTraining from "./dataTraining"
// import irisTesting from "./iris-testing.json"

// function: Math.LN10(Math.abs(Math.cos(x1)) + Math.tan(x2) - (1 / Math.tan(x3) ))
// d1 = ln(|cos(x1)|)+tg(x2)-ctg(x3)

// 5,7,8

// const x1initial = 5;
// const x2initial = 7;
// const x3initial = 8;
// //
// const output = [];
// const x1x2x3training = [];
//
// // const modelingFunction = (x1, x2, x3) => Math.log(Math.abs(Math.cos(x1)) + Math.tan(x2) - (1 / Math.tan(x3) ));
// // const modelingFunction = (x1, x2, x3) => Math.log(Math.abs(Math.cos(x1)) + Math.tan(x2));
// const modelingFunction = (x1, x2, x3) => x1 + x2 + x3;
//
// for (let i = 27; i < 100; i++) {
//   output.push(modelingFunction(x1initial + i, x2initial + i, x3initial + i));
//   x1x2x3training.push([x1initial + i, x2initial + i, x3initial + i]);
// }
//
// console.log(output)
// console.log(x1x2x3training)

const trainingData = tf.tensor2d(dataTraining.inputs);
const outputData = tf.tensor1d(dataTraining.outputs);
const testingData = tf.tensor2d(dataTesting.inputs);
const testingOutputData = tf.tensor1d(dataTesting.outputs);

// // convert/setup our data
// const trainingData = tf.tensor2d(iris.map(item => [
//   item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
// ]))
// const outputData = tf.tensor2d(iris.map(item => [
//   item.species === "setosa" ? 1 : 0,
//   item.species === "virginica" ? 1 : 0,
//   item.species === "versicolor" ? 1 : 0,
// ]))
// const testingData = tf.tensor2d(irisTesting.map(item => [
//   item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
// ]))
//


// build neural network
(async () => {
  // TODO GENERATE trainigng data till 100


  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [3],
    activation: "relu",
    units: 3,
  }));
  model.add(tf.layers.dense({
    inputShape: [3],
    activation: "relu",
    units: 2,
  }));
  model.add(tf.layers.dense({
    inputShape: [2],
    activation: "relu",
    units: 1,
  }));
  model.add(tf.layers.dense({
    activation: "relu", // relu, sigmoid, softmax, relu6
    units: 1,
  }));


  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(0.01),
  });


// // train/fit our network
  model.fit(trainingData, outputData, {epochs: 5000})
    .then((history) => {
      model.predict(testingData).print()
    });
// // test network
})();



console.log('HELLO')

