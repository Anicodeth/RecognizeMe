function Train

    imds = imageDatastore('E:\RecognizeMe\Dataset\Processed2', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    if ~license('test', 'Deep_Learning_HDL_Toolbox')
        error('Deep Learning Toolbox is not installed. Please install the toolbox to proceed.');
    end

    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
    

    layers = [
        imageInputLayer([150 150 1])
        
        convolution2dLayer(3, 16, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer        
        maxPooling2dLayer(2, 'Stride', 2)
    
        convolution2dLayer(3, 64, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(128)
        reluLayer
        
        fullyConnectedLayer(numel(categories(imdsTrain.Labels)))
        softmaxLayer
        classificationLayer
    ];

    
    options = trainingOptions('adam', ...
        'MaxEpochs', 25, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', imdsValidation, ...
        'ValidationFrequency', 10, ...
        'ExecutionEnvironment', 'auto', ...
        'Plots', 'training-progress');



    net = trainNetwork(imdsTrain, layers, options);
    %plotconfusion(net(imdsTrain));%
    

    save('E:\RecognizeMe\Dataset\Model\trained_networktrail.mat', 'net');
end
