import * as tf from "@tensorflow/tfjs"
import "@tensorflow/tfjs-node"
import dataTesting from "./dataTesting"
import dataTraining from "./dataTraining"

import * as tfVis from '@tensorflow/tfjs-vis';


// google machine learning course https://codelabs.developers.google.com/codelabs/neural-tensorflow-js/index.html?index=..%2F..index#5

// 5,7,8

// const x1initial = 5;
// const x2initial = 7;
// const x3initial = 8;
//
// const output = [];
// const x1x2x3training = [];
//

// const modelingFunction = (x1, x2, x3) => x1 + x2 + x3;
// //
//
// for (let i = 0; i < 27; i++) {
//   output.push(modelingFunction(dataTraining.inputs[i][0], dataTraining.inputs[i][1], dataTraining.inputs[i][2]));
//   // x1x2x3training.push([x1initial + i, x2initial + i, x3initial + i]);
// }
// //
// console.log(output.toString())

console.log('HERE')

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
    units: 1,
  }));

  await model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(0.5),
  });

// // train/fit our network
  await model.fit(trainingData, outputData, {epochs: 100})
    .then((history) => {
      return model.predict(testingData).print()

    }).then(predictions => {
      console.log(predictions);
      tfVis.visor().surface({ name: 'My First Surface', tab: 'Input Data' });
      tfVis.show.history(
        {
          name: 'Training History1',
          tab: 'Training1',
        },
        predictions,
        ['loss']
      );
    });
})();

