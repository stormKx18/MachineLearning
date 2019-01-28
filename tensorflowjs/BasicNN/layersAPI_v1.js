
console.log('layersAPI_v1')
//Create model
const model=tf.sequential();

//Configure layers
const config_hidden={
  inputShape:[2],
  units:4,
  activation: 'sigmoid'
};
const config_output={
  units:3,
  activation:'sigmoid'
};

//Create layers
const hidden= tf.layers.dense(config_hidden);
const output= tf.layers.dense(config_output);

//Add layers to model
model.add(hidden);
model.add(output);

//Define optimizer
learningRate=0.1;
const config_optimizer={
  optimizer: tf.train.sgd(learningRate),
  loss:tf.losses.meanSquaredError
}
//Compile model
model.compile(config_optimizer);

const x_train=tf.tensor([
  [0.1,0.5],
  [0.9,0.3],
  [0.4,0.5],
  [0.7,0.1]
]);

const y_train=tf.tensor([
  [0.2,0.1,0.7],
  [0.9,0.05,0.05],
  [0.4,0.5,0.1],
  [0.5,0.3,0.2]
]);

const x_test=tf.tensor([
  [0.9,0.1]
]);

//Configure config_training
const config_training={
  verbose:true,
  epochs:10,
//  shuffle:true
  shuffle:true,
  batch_size:10
}

async function train(epochs){
  for(let i=0;i<epochs;i++){
    const response=await model.fit(x_train,y_train,config_training);
    console.log(response.history.loss[0]);
  }//End for
}//End train

//Train model
let epochs=100;
train(epochs).then(()=>{
  console.log('Training is complete');
  //Predictions
  let y_tmp=model.predict(x_train);
  y_tmp.print();
});
