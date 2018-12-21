require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    // dataColumns: ['horsepower'],
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']    
});

const regression = new LinearRegression(features, labels, {
    learningRate: 10,
    iterations: 1000
});

regression.train();

// v1
// console.log('Update M is:', regression.m, 'Updated B is:', regression.b);

// v2
// console.log('Update M is:', regression.weights.get(1, 0), 'Updated B is:', regression.weights.get(0, 0));

const r2 = regression.test(testFeatures, testLabels);
console.log('R2 is:', r2);