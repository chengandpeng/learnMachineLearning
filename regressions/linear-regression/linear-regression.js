const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];
        this.bHistory = [];

        this.options = Object.assign({ 
            learningRate: 0.01,
            iterations: 1000,
        }, options);

        this.weights = tf.zeros([this.features.shape[1], 1]);
    } 

    // 梯度下降
    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights);
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    // 开始训练
    train() {
        const batchQuantity = Math.floor(
            this.features.shape[0] / this.options.batchSize
        );

        for (let i = 0; i < this.options.iterations; i++) {
            // console.log(this.options.learningRate);
            for(let j = 0; j < batchQuantity; j++) {
                const { batchSize } = this.options;
                const startIndex = j * batchSize;

                const featrueSlice = this.features.slice([ startIndex, 0 ], [ batchSize, -1 ]);
                const labelSlice = this.labels.slice([ startIndex, 0 ], [ batchSize, -1 ]);

                this.gradientDescent(featrueSlice, labelSlice);                
            }
            this.bHistory.push(this.weights.get(0, 0));
            this.recordMSE();
            // this.updateLearningRate();
        }
    }

    predict(observations) {
        return this.processFeatures(observations).matMul(this.weights);
    }

    // 测试准确性 R2
    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);    

        const predictions = testFeatures.matMul(this.weights);
    
        const res = testLabels
          .sub(predictions)
          .pow(2)
          .sum()
          .get();
        const tot = testLabels
          .sub(testLabels.mean())
          .pow(2)
          .sum()
          .get();
    
        return 1 - res / tot;
    }
    
    // standardize初始化
    processFeatures(features) {
        features = tf.tensor(features);

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }

        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    // 标准化数据
    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }

    // 记录MSE
    recordMSE() {
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get();

        this.mseHistory.unshift(mse);
    }

    updateLearningRate() {
        if (this.mseHistory.length < 2) {
            return;
        }

        // mse变大了则除以2，变小了乘以1.05
        if (this.mseHistory[0] > this.mseHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LinearRegression;