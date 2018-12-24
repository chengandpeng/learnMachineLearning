require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    // dataColumns: ['horsepower'],
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']    
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100,
    batchSize: 10
});

regression.train();

// v1
// console.log('Update M is:', regression.m, 'Updated B is:', regression.b);

// v2
// console.log('Update M is:', regression.weights.get(1, 0), 'Updated B is:', regression.weights.get(0, 0));

const r2 = regression.test(testFeatures, testLabels);

// 生成mseHistory图
// plot({
//     x: regression.mseHistory.reverse(),
//     xLabel: 'Iteration #',
//     yLabel: 'Mean Squared Error',
// });

// plot({
//     x: regression.bHistory,
//     y: regression.mseHistory.reverse(),
//     xLabel: 'Value of B',
//     yLabel: 'Mean Squared Error',
// });

// console.log('MSE History is: ', regression.mseHistory.reverse());
console.log('R2 is:', r2);

// 预测
regression.predict([
    [120, 2, 380],
    [135, 2.1, 420]
]).print();