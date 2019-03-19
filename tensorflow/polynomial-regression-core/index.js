/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfVis from '@tensorflow/tfjs-vis';
import { generateData } from './data';
import { plotData, plotDataAndPredictions, renderCoefficients } from './ui';

/**
 * We want to learn the coefficients that give correct solutions to the
 * following cubic equation:
 *      y = a * x^3 + b * x^2 + c * x + d
 * In other words we want to learn values for:
 *      a
 *      b
 *      c
 *      d
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */

// Step 1. Set up variables, these are the things we want the model
// to learn in order to do prediction accurately. We will initialize
// them with random values.
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model performs.
const numIterations = 75;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

// Step 3. Write our training process functions.

/*
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 *
 * @return number predicted y value
 */
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a
      .mul(x.pow(tf.scalar(3, 'int32')))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

/*
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 */
function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  const error = prediction
    .sub(labels)
    .square()
    .mean();
  return error;
}

/*
 * This will iteratively train our model.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, numIterations) {
  for (let iter = 0; iter < numIterations; iter++) {
    // optimizer.minimize is where the training happens.

    // The function it takes must return a numerical estimate (i.e. loss)
    // of how well we are doing using the current state of
    // the variables we created at the start.

    // This optimizer does the 'backward' step of our training process
    // updating variables defined previously in order to minimize the
    // loss.
    optimizer.minimize(() => {
      // Feed the examples into the model
      const pred = predict(xs);
      return loss(pred, ys);
    });

    // Use tf.nextFrame to not block the browser.
    await tf.nextFrame();
  }
}

async function learnCoefficients() {
  tfVis.visor().surface({ name: 'My First Surface', tab: 'Input Data' });

  const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
  const trainingData = generateData(100, trueCoefficients);

  // Plot original data
  renderCoefficients('#data .coeff', trueCoefficients);
  await plotData('#data .plot', trainingData.xs, trainingData.ys);

  // See what the predictions look like with random coefficients
  renderCoefficients('#random .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  const predictionsBefore = predict(trainingData.xs);
  await plotDataAndPredictions(
    '#random .plot',
    trainingData.xs,
    trainingData.ys,
    predictionsBefore
  );

  // Train the model!
  await train(trainingData.xs, trainingData.ys, numIterations);

  // See what the final results predictions are after training.
  renderCoefficients('#trained .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  const predictionsAfter = predict(trainingData.xs);
  await plotDataAndPredictions(
    '#trained .plot',
    trainingData.xs,
    trainingData.ys,
    predictionsAfter
  );

  predictionsBefore.dispose();
  predictionsAfter.dispose();

  (async function watchTraining() {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'show.fitCallbacks',
      tab: 'Training',
      styles: {
        height: '1000px',
      },
    };
    console.log('HERE');
    const callbacks = tfVis.show.fitCallbacks(container, metrics);
    return train(trainingData.xs, trainingData.ys, callbacks);
  })();

  // document.querySelector('#start-training-1').addEventListener('click', () => watchTraining());
}

// learnCoefficients();

const model = tf.sequential();

async function train1() {
  // Create a simple model.
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // Generate some synthetic data for training. (y = 2x - 1)
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // Train the model using the data.
  return model.fit(xs, ys, {
    epochs: 50,
  });
}

async function showTrainingHistory() {
  const trainingHistory = await train1();
  console.log(trainingHistory);

  tfVis.show.history(
    {
      name: 'Training History1',
      tab: 'Training1',
    },
    trainingHistory,
    ['loss']
  );
}

// showTrainingHistory();

import dataTesting from '../../tensorflow/laba1backpropagation/dataTesting';
import dataTraining from '../../tensorflow/laba1backpropagation/dataTraining';

const trainingData = tf.tensor2d(dataTraining.inputs);
const outputData = tf.tensor1d(dataTraining.outputs);

async function train2(layers, rate, epochs) {
  const model2 = tf.sequential();

  await model2.add(
    tf.layers.dense({
      inputShape: [layers[0].in],
      units: layers[0].out,
    })
  );
  await model2.add(
    tf.layers.dense({
      inputShape: [layers[1].in],
      units: layers[1].out,
    })
  );
  // await model2.add(tf.layers.dense({
  //   inputShape: [5],
  //   activation: "softmax",
  //   //   activation: "relu", // relu, sigmoid, softmax, relu6
  //   units: 1,
  // }));
  await model2.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(rate),
  });

  // // train/fit our network
  return model2.fit(trainingData, outputData, { epochs, shuffle: true });
}

async function showTrainingHistory2(layers, rate, epochs) {
  const trainingHistory = await train2(layers, rate, epochs);

  const name = `Model layer0: ${JSON.stringify(layers[0])}; layer[1]: ${JSON.stringify(
    layers[1]
  )}; rate ${rate}; epochs: ${epochs}`;

  console.log(name, ':', trainingHistory);

  trainingHistory.history.loss.splice(0, 5);

  tfVis.show.history(
    {
      name,
      tab: 'Data',
    },
    // slices first 5 entries because the training loss is too high
    trainingHistory,
    ['loss']
  );
}

(async () => {
  await showTrainingHistory2([{ in: 3, out: 2 }, { in: 2, out: 1 }], 1, 50);
})();

(async () => {
  await showTrainingHistory2([{ in: 3, out: 2 }, { in: 2, out: 1 }], 0.01, 50);
})();

(async () => {
  await showTrainingHistory2([{ in: 3, out: 2 }, { in: 2, out: 1 }], 0.0001, 50);
})();

// TO MAKE THIS INDEX.JS run first build it with
// parcel index.js
// AND in index.html require script with source ./dist/index.js
// then open index.html in browser and click ` to open up the charts

