const tf = require('@tensorflow/tfjs');
const fetch = require('node-fetch');

// Use node-fetch in tfjs
global.fetch = fetch;

const testModel = async () => {
    // Set the model URL
    const modelUrl = 'http://localhost:8080/tfjs_model/model.json';

    // Load the model
    const model = await tf.loadLayersModel(modelUrl);

    // Create a dummy input
    const input = tf.zeros([1, 224, 224, 3]);

    // Make a prediction
    const output = model.predict(input);

    // Log the output
    output.print();
};

testModel();