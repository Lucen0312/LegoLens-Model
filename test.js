const tf = require('@tensorflow/tfjs-node');
const assert = require('assert');
const fs = require('fs');
const parseString = require('xml2js').parseString;
const jpeg = require('jpeg-js');
global.fetch = require('node-fetch');
fs.existsSync('tfjs_model/model.json') // should return true
describe('YOLO Model', function() {
  let model;
  let trainData;

  before(async function() {
    // Load the model
    model = await tf.loadLayersModel('tfjs_model/model.json');

    // Load the training data
    const imageFiles = fs.readdirSync('data/train');
    const xmlFiles = fs.readdirSync('data/annotations');

    trainData = await Promise.all(imageFiles.map(async (file, i) => {
      const imageData = fs.readFileSync(`data/train/${file}`);
      const xmlData = fs.readFileSync(`data/annotations/${xmlFiles[i]}`, 'utf8');

      const image = jpeg.decode(imageData, true);
      const tensor = tf.browser.fromPixels(image).expandDims(0);

      let result;
      parseString(xmlData, function (err, res) {
          result = res;
      });

      // Extract bounding box information
      const bbox = result.annotation.object[0].bndbox[0];
      const xmin = parseInt(bbox.xmin[0]);
      const ymin = parseInt(bbox.ymin[0]);
      const xmax = parseInt(bbox.xmax[0]);
      const ymax = parseInt(bbox.ymax[0]);

      return { image: tensor, annotations: { xmin, ymin, xmax, ymax } };
    }));
  });

  it('should output the correct shape', async function() {
    // Convert the training data to a tensor
    const input = tf.stack(trainData.map(data => data.image));

    // Make a prediction
    const output = model.predict(input);

    // Check the output shape
    assert.deepStrictEqual(output.shape, [1, 13, 13, 255]);
  });
});