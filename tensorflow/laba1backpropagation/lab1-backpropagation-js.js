import * as tf from '@tensorflow/tfjs';
// import "@tensorflow/tfjs-node"
import dataTesting from './dataTesting';
import dataTraining from './dataTraining';

import * as tfVis from '@tensorflow/tfjs-vis';

// google machine learning course https://codelabs.developers.google.com/codelabs/neural-tensorflow-js/index.html?index=..%2F..index#5

const trainingData = tf.tensor2d(dataTraining.inputs);
const outputData = tf.tensor1d(dataTraining.outputs);
const testingData = tf.tensor2d(dataTesting.inputs);
const testingOutputData = tf.tensor1d(dataTesting.outputs);

// build neural network
(() => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [3],
      units: 3,
    })
  );
  model.add(
    tf.layers.dense({
      inputShape: [3],
      units: 1,
    })
  );

  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.5),
  });

  // // train/fit our network
  model
    .fit(trainingData, outputData, { epochs: 100 })
    .then(history => {
      return model.predict(testingData).print();
    })
})();

// TO RUN AS NODE PROGRAM UNCOMMENT IMPORT, TO RUN IN BROWSER COMMENT tf-node import!!!

tfVis.show.history(
  {
    name: 'preds',
    tab: 'preds',
  },
  {
    history: {
      pr: [20.759161, 23.4505157, 26.1418648, 27.055582, 27.0666752, 27.0390644],
      expected: [20, 23, 26, 27, 27, 27],
    },
  },
  ['pr', 'expected']
);
