% Conditional Variational Autoencoder (Conditional-VAE) 
% Augmented data from Paper1 for train-validate-test 

clc; clear;

filePathTrainValid = "Data\31merOesDoubleInterpolatedP3Abnormal.mat";
load(filePathTrainValid);


rng('default')

XTrain = segsIP31OesDoubleP3Abnormal;
YTrain = categorical(labelsIP31OesDoubleP3Abnormal');
XTrainReshaped = reshape(XTrain,[300 1 1 size(XTrain,2)]);
 
dsTrainVae = arrayDatastore(XTrainReshaped,"IterationDimension",4);
% readall(dsTrainVae)

% Define Encoder Network Architecture
numLatentChannels = 32;
embeddingDimension = 32;
inputSize = [300 1 1];
numClasses = 6;
scale = 0.1;
prob = 0.25;
numFilters = 16;

layersEncoder = [
    imageInputLayer(inputSize,Normalization="none")    
    fullyConnectedLayer(300)
    functionLayer(@(X) feature2image(X,[300 1 1]),Formattable=true)    
    convolution2dLayer([3 1],numFilters,'Padding',[1 0],'Stride',2,'Name', "conv1")       
    leakyReluLayer(scale,'Name','relu1')
    dropoutLayer(prob)
    convolution2dLayer([3 1],numFilters,'Padding',[1 0],'Stride',2,'Name', "conv2")         
    leakyReluLayer(scale,'Name','relu2')
    dropoutLayer(prob)
    convolution2dLayer([3 1],numFilters,'Padding',[1 0],'Stride',2,'Name', "conv3")   
    additionLayer(2,'Name', 'add2')    
    leakyReluLayer(scale,'Name','relu3')
    dropoutLayer(prob)
    fullyConnectedLayer(2*numLatentChannels)
    samplingLayer];

lgraphEncoder = layerGraph(layersEncoder);
skipConv2 = convolution2dLayer([3 1],numFilters,'Padding',[1 0],'Stride',2,'Name','skipConv2');
lgraphEncoder = addLayers(lgraphEncoder,skipConv2);
lgraphEncoder = connectLayers(lgraphEncoder,'relu2','skipConv2');
lgraphEncoder = connectLayers(lgraphEncoder,'skipConv2','add2/in2'); 
lgraphE = layerGraph(layersEncoder);
netEncodeSigd2IPAbnormal = dlnetwork(lgraphEncoder);
% analyzeNetwork(netEncodeSigd2IPAbnormal);
% plot(lgraphEncoder);

% Define Decoder Network Architecture
projectionSize = [32 1 64];
inputSize = [300 1 1];
numInputChannels = size(inputSize,1);
numLatentInputs = numLatentChannels;

layersDecoder = [
    featureInputLayer(numLatentInputs,Name = 'Noise',Normalization="none")
    fullyConnectedLayer(prod(projectionSize),Name ='FC_prepN')
    functionLayer(@(X) feature2image(X,projectionSize),Formattable=true,Name ='ProjectNoise')    
    transposedConv2dLayer([7 1],numFilters,'Stride',1,'Cropping',[0 0],'Name','tconv1')          
    leakyReluLayer(scale,'Name','relu1')
    dropoutLayer(prob,'Name','drop1')
    transposedConv2dLayer([3 1],numFilters,'Stride',2,'Cropping',[1 0],'Name','tconv2')    
    leakyReluLayer(scale,'Name','relu2')
    dropoutLayer(prob,'Name','drop2')
    transposedConv2dLayer([4 1],numFilters,'Stride',2,'Cropping',[1 0],'Name','tconv3')
    additionLayer(2,'Name', 'add2')    
    leakyReluLayer(scale,'Name','relu3')
    dropoutLayer(prob,'Name','drop3')
    transposedConv2dLayer([4 1],numInputChannels,'Stride',2,'Cropping',[1 0],'Name','tconv4')    
    functionLayer(@(X) 1.5*tanh(X),'Name','scalingTanh')
];
lgraphDecoder = layerGraph(layersDecoder);
skiptConv2 = transposedConv2dLayer([4 1],numFilters,'Cropping',[1 0],'Stride',2,'Name','skiptConv2');
lgraphDecoder = addLayers(lgraphDecoder,skiptConv2); 
lgraphDecoder = connectLayers(lgraphDecoder,'relu2','skiptConv2');
lgraphDecoder = connectLayers(lgraphDecoder,'skiptConv2','add2/in2');
netDecodeSigd2IPAbnormal = dlnetwork(lgraphDecoder);
% analyzeNetwork(netDecodeSigd2IPAbnormal)
% plot(lgraphDecoder);

% Define Model Loss Function

% Specify Training Options
numEpochs =6000;
miniBatchSize = 36;
initialLearnRate = 0.0002;
decay = 1;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
executionEnvironment = "CPU";
learnRate = initialLearnRate;


% Train Model
mbqTrain = minibatchqueue(dsTrainVae, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    PartialMiniBatch="return",...
    OutputEnvironment=executionEnvironment);

 % itialize the parameters for the Adam solver.
trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];

numObservationsTrain = size(XTrain,2)
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize)
numIterations = numEpochs * numIterationsPerEpoch

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Epoch","ExecutionEnvironment","Iteration","LearningRate"], ...
    XLabel="Iteration");

groupSubPlot(monitor,"Loss","TrainingLoss")

epoch = 0;
iteration = 0;
lossValue =[];

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbqTrain);

    % Loop over mini-batches.
    while hasdata(mbqTrain) && ~monitor.Stop
        iteration = iteration + 1;

        % Read mini-batch of data.
        X = next(mbqTrain);
        % Evaluate loss and gradients.
        [loss,gradientsE,gradientsD,lastOutputTrain] = dlfeval(@modelLoss,netEncodeSigd2IPAbnormal,netDecodeSigd2IPAbnormal,X);
        lossValue = [lossValue loss];
        % Determine learning rate for time-based decay learning rate schedule.
%         learnRate = initialLearnRate/(1 + decay*iteration);
%         learnRate = initialLearnRate;
        
        % Update learnable parameters.
        [netEncodeSigd2IPAbnormal,trailingAvgE,trailingAvgSqE] = adamupdate(netEncodeSigd2IPAbnormal, ...
            gradientsE,trailingAvgE,trailingAvgSqE,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        [netDecodeSigd2IPAbnormal, trailingAvgD, trailingAvgSqD] = adamupdate(netDecodeSigd2IPAbnormal, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the training progress monitor. 
        recordMetrics(monitor,iteration,TrainingLoss=loss);
        updateInfo(monitor,ExecutionEnvironment = executionEnvironment,LearningRate=learnRate,...
            Epoch=epoch + " of " + numEpochs, Iteration = iteration + " of " + numIterations);



        monitor.Progress = 100*iteration/numIterations;
    end
end

% Save Networks
% 
% filePathOutput = "Results\Paper3Result\";
% 
% 
% netEncoderPath = strcat(filePathOutput,"netEncodeSigd2IPAbnormal.mat");
% netDecoderPath = strcat(filePathOutput,"netDecodeSigd2IPAbnormal.mat");
% 
% save(netEncoderPath , "netEncodeSigd2IPAbnormal");
% save(netDecoderPath , "netDecodeSigd2IPAbnormal");

% Generate new data
numObservationsNew =200-36;

ZNew = randn(numLatentInputs,numObservationsNew,"single");
ZNew = dlarray(ZNew,"CB");
XGenNew = predict(netDecodeSigd2IPAbnormal,ZNew);
XGenNewMat = extractdata(XGenNew);
XGenArray = reshape(XGenNewMat,[300,  size(XGenNew,4)]);
XGenTable = array2table(XGenArray);



allAnomalyTable = [array2table(XTrain) XGenTable];

cutTable = allAnomalyTable(101:300,:);
Abnormal31merOes = [allAnomalyTable;cutTable];

% outputPath = "Results\Paper3DB\";
% outputName = strcat(outputPath,"Abnormal31merOes.mat");
% 
% save(outputName, "Abnormal31merOes");

% Helper Functions
% Model Loss Function

function [loss,gradientsE,gradientsD,Y] = modelLoss(netE,netD,X)

% Forward through encoder.
[Z,mu,logSigmaSq] = forward(netE,X);

% Forward through decoder.
Y = forward(netD,Z);

% Calculate loss and gradients.
loss = elboLoss(Y,X,mu,logSigmaSq);
[gradientsE,gradientsD] = dlgradient(loss,netE.Learnables,netD.Learnables);

end
% ELBO Loss Function

function loss = elboLoss(Y,Target,mu,logSigmaSq)

% Reconstruction loss.
reconstructionLoss = mse(Y,Target);

% KL divergence.
KL = -0.5 * sum(1 + logSigmaSq - mu.^2 - exp(logSigmaSq),1);
KL = mean(KL);

% Combined loss.
loss = reconstructionLoss + KL;

end
% Model Predictions Function
function [XReal, XhatSingle,XhatDlArray]= modelPredictions(netE,netD,mbq)

XReal =[];
XhatSingle = [];             % a single array
Label=[];
XhatDlArray = [];
XGeneratedAll= []; % a deep learning array

% Reset mini-batch queue.
reset(mbq);
% Loop over mini-batches.
while hasdata(mbq)
    Xtest = next(mbq);    
    XRealLoop = Xtest;   

    % Forward through encoder.
     
    ZGen = predict(netE,Xtest);

    % Forward through dencoder.
    XGenerated = predict(netD,ZGen);
    XGeneratedAll= cat(4,XGeneratedAll,XGenerated);

    % Extract and concatenate predictions.
    XReal = cat(4,XReal,extractdata(XRealLoop));
    XhatSingle =  cat(4,XhatSingle,extractdata(XGenerated));
end
XhatDlArray = XGeneratedAll;
end
% Mini Batch Preprocessing Function

function X = preprocessMiniBatch(XCell)
    % Extract image data from the cell array and concatenate over fourth
    % dimension to add a third singleton dimension, as the channel
    % dimension.
    X = cat(4,XCell{:});

end
% Mini-Batch Predictors Preprocessing Function

function X = preprocessMiniBatchPredictors(XCell)
X = cat(4,XCell{:});
end

