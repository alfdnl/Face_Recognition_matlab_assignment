% This code is use to train the model.

% n is the number of subjects
n = 5;
% You can press stop button manually on tranining plot(on top right corner besides number of iterations) once accuracy reaches upto desired level

% Data Preprocessing
 im = imageDatastore('croppedfaces','IncludeSubfolders',true,'LabelSource','foldernames');
 % Resize the images to the input size of the net
 im.ReadFcn = @(loc)imresize(imread(loc),[299,299]);
 [Train ,Test] = splitEachLabel(im,0.8,'randomized');
 %fc = fullyConnectedLayer(n);
 net = googlenet;
 
 inputSize = net.Layers(1).InputSize;
 
 if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
 else
  lgraph = layerGraph(net);
 end
 
 [learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = n;

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),Train);
augimdsTest = augmentedImageDatastore(inputSize(1:2),Test);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsTest, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsTest);

 
 % Saving Model
 model_google_1 = net;
 save model_google_1
 
 names = Test.Labels;
 pred = (YPred==names);
 s = size(pred);
 acc = sum(pred)/s(1);
 fprintf('The accuracy of the test set is %f %% \n',acc*100);

