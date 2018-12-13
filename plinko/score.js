const outputs = [];
const k = 5;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel])
}

function runAnalysis() {
  // Write code here to analyze stuff
  const testSetSize = 50;
  const [testSet, trainingSet] = splitDataset(outputs, testSetSize);

  _.range(1, 15).forEach(k => {
    const accuracy = _.chain(testSet)
    .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3])
    .size()
    .divide(testSetSize)
    .value();

    console.log('For k of', k, 'Accuracy:' + accuracy);
  });
}

// one dimension
// function knn(data, point, k) {
//   return _.chain(data)
//     .map(row => [distance(row[0], point), row[3]])
//     .sortBy(row => row[0])
//     .slice(0, k)
//     .countBy(row => row[1])
//     .toPairs()
//     .sortBy(row => row[1])
//     .last()
//     .first()
//     .parseInt()
//     .value();
// }

function knn(data, point, k) {
  // point has 3 values
  return _.chain(data)
    .map(row => {
      return [
        distance(_.initial(row), point),
        _.last(row)
      ]
    })
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function distance(pointA, pointB) {
  // pointA = 300, pointB = 350
  // return Math.abs(pointA - pointB);

  // pointA = [300, .5, 16], pointB = [350, .55, 16]
  return _.chain(pointA)
    .zip(pointB)
    .map(([a, b]) => (a - b) ** 2)
    .sum()
    .value() ** 0.5;
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}