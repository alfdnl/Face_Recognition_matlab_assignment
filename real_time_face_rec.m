% Create the face detector object.
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',[150,150]);

% Load trained model
load model_google_1.mat;
newnet = model_google_1;

% Create the webcam object.
cam = webcam();

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
runLoop = true;
numPts = 0;
frameCount = 0;
i=1;

while runLoop
    
    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    % Detect Face
    bbox = faceDetector.step(videoFrameGray);
    if ~isempty(bbox)
        
        % Crop face image to predict classification
        img=imcrop(videoFrame,bbox(i,:));
        img = imresize(img,[224 224]);
        
        [predict,scores] = classify(newnet,img);
        disp(scores);
    
        % Convert the rectangle represented as [x, y, w, h] into an
        % M-by-2 matrix of [x,y] coordinates of the four corners. This
        % is needed to be able to transform the bounding box to display
        % the orientation of the face.
        bboxPoints = bbox2points(bbox(1, :));

        % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
        % format required by insertShape.
        bboxPolygon = reshape(bboxPoints', 1, []);
        
        % Display a bounding box around the detected face.
        if predict=='s01'            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            position = [(bboxPolygon(1,7)+bboxPolygon(1,5))/2 (bboxPolygon(1,8)+bboxPolygon(1,6))/2];
            disp(position);
            videoFrame = insertText(videoFrame,position ,['Aliff'], 'AnchorPoint', 'LeftBottom');
        elseif predict=='s02'            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            position = [(bboxPolygon(1,7)+bboxPolygon(1,5))/2 (bboxPolygon(1,8)+bboxPolygon(1,6))/2];
            disp(position);
            videoFrame = insertText(videoFrame,position ,['Amin'], 'AnchorPoint', 'LeftBottom');
        elseif predict=='s03'            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            position = [(bboxPolygon(1,7)+bboxPolygon(1,5))/2 (bboxPolygon(1,8)+bboxPolygon(1,6))/2];
            disp(position);
            videoFrame = insertText(videoFrame,position ,['Aqiff'], 'AnchorPoint', 'LeftBottom');
        elseif predict=='s04'            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            position = [(bboxPolygon(1,7)+bboxPolygon(1,5))/2 (bboxPolygon(1,8)+bboxPolygon(1,6))/2];
            disp(position);
            videoFrame = insertText(videoFrame,position ,['Akif'], 'AnchorPoint', 'LeftBottom');
        elseif predict=='s05'            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            position = [(bboxPolygon(1,7)+bboxPolygon(1,5))/2 (bboxPolygon(1,8)+bboxPolygon(1,6))/2];
            disp(position);
            videoFrame = insertText(videoFrame,position ,['Zarif'], 'AnchorPoint', 'LeftBottom');
        else
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            position = [(bboxPolygon(1,7)+bboxPolygon(1,5))/2 (bboxPolygon(1,8)+bboxPolygon(1,6))/2];
            disp(position);
            videoFrame = insertText(videoFrame,position ,['unknown'], 'AnchorPoint', 'LeftBottom');
        end
    end

    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);