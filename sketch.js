let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "o"

let state = 'waiting';
let targetLabel;

let lastPlayed = ""; // Keeps track of the last played sound


function keyPressed() {
  if (key == 's') {
    brain.saveData();
  } else {
  targetLabel = key;
  console.log(targetLabel);
  setTimeout(function() {
    console.log('collecting');
    state = 'collecting';
    setTimeout(function() {
      console.log('not collecting');
      state = 'waiting';
    }, 10000);
  }, 2000);
  }
}

function setup() {
  leftSound = loadSound('sounds/left.mp3');
  rightSound = loadSound('sounds/right.mp3');
  outSound = loadSound('sounds/out.mp3');
  leftUpSound = loadSound('sounds/leftup.mp3');
  rightUpSound = loadSound('sounds/rightup.mp3');
  bothUpSound = loadSound('sounds/bothup.mp3');

  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  
  let options = {
    inputs: 34,
    outputs: 6,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
const modelInfo = {
  model: 'model/model.json',
  metadata: 'model/model_meta.json',
  weights: 'model/model.weights.bin',
};
brain.load(modelInfo, brainLoaded);
  //brain.loadData('arms.json', dataReady);
}

function brainLoaded() {
  console.log("pose classification ready!");
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
        for(let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) {
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label;
  }
//  console.log(results);
 // console.log(results[0].confidence);
  classifyPose();
}

function dataReady() {
  brain.normalizeData();
  brain.train({epochs: 50}, finished);
}
function finished() {
  console.log('model trained');
  brain.save();
}

function gotPoses(poses) {
  //console.log(poses);
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    
    if (state == 'collecting') {
let inputs = [];
        for(let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    let target = [targetLabel];
    brain.addData(inputs, target);
  }
  }
}

function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  image(video, 0, 0);
  
  if (pose) {
    
    let eyeR = pose.rightEye;
    let eyeL = pose.leftEye;
    let d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y);
    
    for(let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0,255,0);
      ellipse(x,y,16,16);
    }
    
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(255);
      line(a.position.x, a.position.y,b.position.x,b.position.y);
  }
}
  fill(255,0,255);
  noStroke();
  textSize(256);
  textAlign(CENTER, CENTER);
  text(poseLabel,width/2,height/2);
  
  if (poseLabel !== lastPlayed) {
    switch (poseLabel) {
      case "r":
        rightSound.play();
        break;
      case "l":
        leftSound.play();
        break;
      case "o":
        outSound.play();
        break;
      case "p":
        rightUpSound.play();
        break;
      case "t":
        leftUpSound.play();
        break;
      case "u":
        bothUpSound.play();
        break;
    }
    lastPlayed = poseLabel; // Update the last played sound
  }
}