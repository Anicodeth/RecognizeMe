function Test  

    %Cleaned 80 percent data, 90% accuracy
    load('E:\RecognizeMe\Dataset\Model\trained_networkCleanedTrain.mat', 'net');

    %60 percent data is used, 85% accuracy
    %load('E:\RecognizeMe\Dataset\Model\trained_network60Data.mat', 'net');

    %80 percent data uncleaned, 70-80%  accuracy
    %load('E:\RecognizeMe\Dataset\Model\trained_network75-80.mat', 'net');

    %80 percent data, 90-100%  accuracy (OVER FITTED)
    %load('E:\RecognizeMe\Dataset\Model\trained_network90-100.mat', 'net');

    %70 - 80 acuraccy, classifier between ananya, hass, ephi
    %Overfitteed confidence
    %load('E:\RecognizeMe\Dataset\Model\trained_networkHassAnanyaEphiClassier.mat', 'net');

    testImagePath = imageDatastore('E:\RecognizeMe\Dataset\Test');
   
    for i = 1:length(testImagePath.Files)
            testImage = testImagePath.readimage(i);
            faceDetector = vision.CascadeObjectDetector;
            face = step(faceDetector, testImage);
            fprintf('No face or multiple faces detected in jthe test image.\n');
            figure;
            imshow(testImage);
            hold on;
            for j = 1:size(face, 1)
                singleFace = face(j, :);
                rectangle('Position',singleFace,'EdgeColor','r','LineWidth', 2);
                faceImage = imcrop(testImage, singleFace);
                faceGray = im2gray(faceImage);
                resizedFace = imresize(faceGray, [ 150, 150 ]);
                [ predictedFaceLabel, scores ]  = classify(net, resizedFace);
                confidence = max(scores);
        
                text(singleFace(1), singleFace(2) - 40, sprintf('Predicted Label: %s \nConfidence: %.2f%%', char(predictedFaceLabel), (confidence - 0.025) * 100), 'Color','r','FontSize',12);
                pause(0.5)
            end
            hold off;
    end


