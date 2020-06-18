% n is the number of subjects
n = 5;
% You can press stop button manually on tranining plot(on top right corner besides number of iterations) once accuracy reaches upto desired level

% looping through all subjects and cropping faces if found
% extract the subject photo and crop faces and saving it in to respective
% folders
%for i =1:n
%   str = ['s0',int2str(i)];
%  ds1 = imageDatastore(['photos\',str],'IncludeSubfolders',true,'LabelSource','foldernames');
%   cropandsave(ds1,str);
%end
 im = imageDatastore('croppedfaces','IncludeSubfolders',true,'LabelSource','foldernames');
 % Resize the images to the input size of the net
 im.ReadFcn = @(loc)imresize(imread(loc),[299,299]);
 [Train ,Test] = splitEachLabel(im,0.8,'randomized');
 %fc = fullyConnectedLayer(n);
 net = inceptionv3;
 inputSize = net.Layers(1).InputSize;
 if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
 else
  lgraph = layerGraph(net);
 end 
 
 %[learnableLayer,classLayer] = findLayersToReplace(lgraph);
learnableLayer = net.Layers(313);
classLayer = net.Layers(315);
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

layers(1:17) = freezeWeights(layers(1:17));
lgraph = createLgraphUsingConnections(layers,connections);
 %ly = net.Layers;
 %ly(313) = fc;
 %cl = classificationLayer;
 %ly(315) = cl; 
 % options for training the net if your newnet performance is low decrease
 % the learning_rate
 learning_rate = 0.00001;
 opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,'MaxEpochs',10,'Shuffle','every-epoch','MiniBatchSize',64,'Plots','training-progress');
 [newnet,info] = trainNetwork(Train, lgraph, opts);
 [predict,scores] = classify(newnet,Test);
 disp(scores);
 
 % Saving Model
 model_1 = newnet;
 save model_1
 
 names = Test.Labels;
 pred = (predict==names);
 s = size(pred);
 acc = sum(pred)/s(1);
 fprintf('The accuracy of the test set is %f %% \n',acc*100);

