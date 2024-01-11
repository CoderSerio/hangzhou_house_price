const fs = require("fs");
const csvParser = require("csv-parser");
const iconv = require("iconv-lite");
const tf = require("@tensorflow/tfjs-node");

// 需要关注的特征和目标变量
const featuresColumns = ["经度", "纬度"];
const targetColumn = "单价";

// 预处理函数，将原始数据转换为包含特征与目标值的对象数组
const preprocessData = (data) => {
  return data
    .filter((item) => {
      const hasAllFeatures = featuresColumns.every((feature) => item[feature]);
      return hasAllFeatures && item[targetColumn];
    })
    .map((item) => {
      const processedItem = {
        features: featuresColumns.map((feature) => parseFloat(item[feature])),
        unitPrice: parseFloat(item[targetColumn]),
      };

      // 对日期进行处理（假设挂牌时间是yyyy-MM-dd格式）
      // if (!isNaN(processedItem.features[2])) {
      //   // 如果挂牌时间能被解析为数字，则转化为天数差值
      //   const dateStr = new Date(item["挂牌时间"]).toISOString().split("T")[0];
      //   const baseDate = new Date("2020-01-01").getTime(); // 基准日期
      //   const listingDate = new Date(dateStr).getTime();
      //   processedItem.features[2] =
      //     (listingDate - baseDate) / (1000 * 60 * 60 * 24);
      // }
      console.log(processedItem);
      return processedItem;
    });
};

// 创建模型函数
const createModel = (inputShape) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      inputShape: [inputShape],
      activation: "linear",
    })
  );
  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });
  return model;
};

// 训练模型函数
const trainModel = async (processedData, epochs = 100) => {
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
  fs.createReadStream("my_data/linan杭州二手房.csv")
    .pipe(iconv.decodeStream("gbk")) // 将这里的'gbk'替换为你的实际编码格式
    .pipe(csvParser())
    .on("data", (row) => {
      data.push(row);
    })
    .on("end", () => {
      const processedData = preprocessData(data);

      trainModel(processedData)
        .then((model) => {
          // model.save("./house_price_model");
          // console.log(model);
          model.save("file://./my-model");
          // 进行评估或保存模型等操作...
        })
        .catch((err) => console.error(err));
    });
}

main();
