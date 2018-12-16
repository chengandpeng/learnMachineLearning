require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

// V1: lat, long
// let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
//     shuffle: true,
//     splitTest: 10,
//     dataColumns: ['lat', 'long'],
//     labelColumns: ['price']
// });

// v1
// function knn(features, labels, predictionPoint, k) {
//     return features
//         .sub(predictionPoint)
//         .pow(2)
//         .sum(1)
//         .pow(0.5)
//         .expandDims(1)
//         .concat(labels, 1)
//         .unstack()
//         .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
//         .slice(0, k)
//         .reduce((acc, pair) => acc + pair.get(1), 0) / k;
// }

// V2
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot'],
    labelColumns: ['price']
});

// V2
function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

    return features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.get(1), 0) / k;
}



features = tf.tensor(features);
labels = tf.tensor(labels);
// testFeatures = tf.tensor(testFeatures);
// testLabels = tf.tensor(testLabels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    
    // console.log('Guess', result, testLabels[0][0]);
    console.log('Error', err * 100);
})

