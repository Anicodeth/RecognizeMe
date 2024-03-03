function PreProcess()
    inputPath = "E:\RecognizeMe\Dataset\Train2\";
    outputFolder = "E:\RecognizeMe\Dataset\Processed4";
    images = imageDatastore(inputPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    faceDetector = vision.CascadeObjectDetector;
    
    for i = 1 : length(images.Files)
        image = images.readimage(i);
        imageGray = rgb2gray(image);
        %imageGray = image;%
        faces = step(faceDetector, imageGray);
        
        for j = 1:size(faces, 1)
            face = faces(j, :);
            cropped = imcrop(imageGray, face);
            resized = imresize(cropped, [150, 150]);
            
            label = images.Labels(i);
            
            labelFolder = fullfile(outputFolder, char(label));
            if ~exist(labelFolder, 'dir')
                mkdir(labelFolder);
            end
            imshow(resized);   
            pause(0.2);
            outputFileName = fullfile(labelFolder, sprintf('processed_image_%d_%d.png', i, j));
            imwrite(resized, outputFileName);
            

        end
    end
end

