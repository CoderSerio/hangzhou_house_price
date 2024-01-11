const fs = require("fs");
const csvParser = require("csv-parser");
const iconv = require("iconv-lite");
const tf = require("@tensorflow/tfjs-node");

// 需要关注的特征和目标变量
const featuresColumns = ["经度", "纬度"];
const targetColumn = "单价";

// 映射到某个区间
const handleFeature = (value) => {
  return parseFloat(value) / 100;
};

const preprocessData = (data) => {
  // 具体到这个问题上，主要是经度维度存在小数点后位数特别多的情况，直接用会导致精度溢出
  //! 这里进行【归一化】操作，一般就是将所有值映射到 [0, 1] 这个区间内、
  //! 不过这里的我们就随便映射到一个区间了

  return data
    .filter((item) => item[targetColumn])
    .map((item) => {
      const processedItem = {
        features: [handleFeature(item["经度"]), handleFeature(item["纬度"])],
        unitPrice: parseFloat(item[targetColumn]),
      };
      return processedItem;
    });
};

const createModel = (inputShape) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      inputShape: [inputShape],
      activation: "linear",
    })
  );
  const learningRate = 0.0001;
  model.compile({
    optimizer: tf.train.sgd(learningRate),
    loss: "meanSquaredError",
  });
  return model;
};

const trainModel = async (processedData, epochs = 100) => {
  console.log("开始训练");
  const xs = tf.tensor2d(
    processedData.map((item) => item.features),
    [processedData.length, processedData[0].features.length]
  );
  const ys = tf.tensor1d(processedData.map((item) => item.unitPrice));

  const model = createModel(processedData[0].features.length);

  await model.fit(xs, ys, {
    epochs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      },
    },
  });
  return model;
};

// 主函数
async function main() {
  let data = [];
  //! 第一步：读取数据
  fs.createReadStream("my_data/linan杭州二手房.csv")
    .pipe(iconv.decodeStream("gbk")) // 将这里的'gbk'替换为你的实际编码格式
    .pipe(csvParser())
    .on("data", (row) => {
      data.push(row);
    })
    .on("end", () => {
      //! 第二步：预处理数据
      const processedData = preprocessData(data);
      //! 第三步：训练模型
      trainModel(processedData)
        .then((model) => {
          //! 第四步：测试模型(由于这里只做示意，所以这里就随便 mock 了一个数据意思一下就行了)
          const mockSample = [
            handleFeature(119.730838),
            handleFeature(30.255086),
          ];
          const xs = tf.tensor2d(mockSample, [1, mockSample.length]);
          const predictionTensor = model.predict(xs);
          console.log("结果：", predictionTensor.arraySync());

          // 保存到本地
          model.save("file://./my-model");
        })
        .catch((err) => console.error(err));
    });
}

main();
