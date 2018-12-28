require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

function loadData(num) {
    const mnistData = mnist.training(0, num);

    const features = mnistData.images.values.map(image => _.flatMap(image));
    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;
    });

    return { features, labels: encodedLabels };
}

const { features, labels } = loadData(10000);

// console.log(mnistData.labels.values);
// console.log(encodedLabels);

const regression = new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 20,
    batchSize: 100
});

regression.train();

const { features: testFeatures, labels: testEncodeLabels } = loadData(1000);

const accuracy = regression.test(testFeatures, testEncodeLabels);
console.log('Accuracy is:', accuracy);

plot({
    x: regression.costHistory.reverse()
})
console.log(regression.costHistory);