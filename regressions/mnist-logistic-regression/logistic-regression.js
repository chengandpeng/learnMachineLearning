const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.costHistory = [];
        this.bHistory = [];

        this.options = Object.assign({ 
            learningRate: 0.1,
            iterations: 1000,
            decisionBoundary: 0.5
        }, options);

        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    } 

    // 梯度下降
    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).softmax();
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])

        return this.weights.sub(slopes.mul(this.options.learningRate));
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

                this.weights = tf.tidy(() => {
                    const featrueSlice = this.features.slice([ startIndex, 0 ], [ batchSize, -1 ]);
                    const labelSlice = this.labels.slice([ startIndex, 0 ], [ batchSize, -1 ]);
    
                    return this.gradientDescent(featrueSlice, labelSlice);           
                });
            }
            this.bHistory.push(this.weights.get(0, 0));
            this.recordCost();
            this.updateLearningRate();
        }
    }

    predict(observations) {
        return this.processFeatures(observations)
            .matMul(this.weights)
            .softmax()
            .argMax(1);
    }

    // 测试准确性
    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        testLabels = tf.tensor(testLabels).argMax(1);

        const incorrect = predictions
            .notEqual(testLabels)
            .sum()
            .get();

        return (predictions.shape[0] - incorrect) / predictions.shape[0];
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

        const filler = variance.cast('bool').logicalNot().cast('float32');

        this.mean = mean;
        this.variance = variance.add(filler);

        return features.sub(mean).div(this.variance.pow(0.5));
    }

    // 记录MSE
    recordCost() {
        const cost = tf.tidy(() => {
            const guesses = this.features.matMul(this.weights).softmax();

            const termOne = this.labels
                .transpose()
                .matMul(guesses.log())
    
            const termTwo = this.labels
                .mul(-1)
                .add(1)
                .transpose()
                .matMul(
                    guesses.mul(-1).add(1).log()
                );
    
            return termOne
                .add(termTwo)
                .div(this.features.shape[0])
                .mul(-1)
                .get(0, 0);
            
        });
        
        this.costHistory.unshift(cost);
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) {
            return;
        }

        // mse变大了则除以2，变小了乘以1.05
        if (this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LogisticRegression;