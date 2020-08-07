% This code is used to test the model

% Load trained model
load model_google_1.mat;
newnet = model_google_1;

% Test a new Image
% use code below with giving path to your new image
 img = imread('.\Image_to_test\test_zarif.jpg');
 [img,face] = cropface(img);
% face value is 1 when it detects face in image or 0
 if face == 1
   img = imresize(img,[224 224]);
   [predict,scores] = classify(newnet,img);
 end
 nameofs01 = 'Aliff';
 nameofs02 = 'Sofea';
% nameofs03 = 'name of subject 3';
if predict=='s01'
   name='Aliff'; 
   disp(name)
   disp(scores)
elseif  predict=='s02'
   name='Amin';
   disp(name)
   disp(scores)
elseif  predict=='s03'
   name='Aqiff';
   disp(name)
   disp(scores)
elseif  predict=='s04'
   name='Akif';
   disp(name)
   disp(scores)
elseif  predict=='s05'
   name='Zarif';
   disp(name)
   disp(scores)
else
    disp("Nothing detected")
end
