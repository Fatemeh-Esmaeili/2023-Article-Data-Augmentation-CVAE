% Conditional Variational Autoencoder (Conditional-VAE) 
% Generating new data

% clc; clear;

filePathTrainValid = "Data\35merAdeDoubleP3.mat";
load(filePathTrainValid);


filePathTest = "Data\35merAdeDoubleP3DA.mat";
load(filePathTest);

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

rng('default')

XTrain = segs35AdeDoubleP3;
YTrain = categorical(labels35AdeDoubleP3');

% Test set is Augmented data from Data Scaling Method Paper1
XTest = segs35AdeDoubleP3DA;
YTest = categorical(labels35AdeDoubleP3DA');


XTrainReshaped = reshape(XTrain,[500 1 1 size(XTrain,2)]);
XTestReshaped = reshape(XTest,[500 1 1 size(XTest,2)]);


dsXTrain = arrayDatastore(XTrainReshaped,"IterationDimension",4);
dsYTrain= arrayDatastore(YTrain);
dsTrainVae = combine(dsXTrain,dsYTrain);


dsXTest = arrayDatastore(XTestReshaped,"IterationDimension",4);
dsYTest = arrayDatastore(YTest);
dsTestVae = combine(dsXTest,dsYTest);
% XTrainReshaped(:,:,1,3)
% readall(dsTrainVae)
% readall(dsTestVae)


% Define Encoder Network Architecture
numLatentChannels = 32;
embeddingDimension = 32;
inputSize = [500 1 1];
numClasses = 6;
scale = 0.1;
prob = 0.25;


layersEncoder = [
    imageInputLayer(inputSize,Normalization="none")    
    fullyConnectedLayer(500)
    functionLayer(@(X) feature2image(X,[500 1 1]),Formattable=true)
    concatenationLayer(3,2,'Name',"cat")
    convolution2dLayer([3 1],16,'Padding',[1 0],'Stride',2,'Name', "conv1")
    
    
    leakyReluLayer(scale,'Name','relu1')
    dropoutLayer(prob)
    convolution2dLayer([3 1],16,'Padding',[1 0],'Stride',2,'Name', "conv2")
    
    additionLayer(2,'Name','add1')
    leakyReluLayer(scale,'Name','relu2')
    dropoutLayer(prob)
    convolution2dLayer([3 1],16,'Padding',[1 0],'Stride',2,'Name', "conv3")
   
    additionLayer(2,'Name', 'add2')
    leakyReluLayer(scale,'Name','relu3')
    dropoutLayer(prob)
    fullyConnectedLayer(2*numLatentChannels)
    samplingLayer];

lgraphEncoder = layerGraph(layersEncoder);


skipConv1 = convolution2dLayer([3 1],16,'Padding',[1 0],'Stride',4,'Name','skipConv1');
skipConv2 = convolution2dLayer([3 1],16,'Padding',[1 0],'Stride',2,'Name','skipConv2');


lgraphEncoder = addLayers(lgraphEncoder,skipConv1);
lgraphEncoder = addLayers(lgraphEncoder,skipConv2);

lgraphEncoder = connectLayers(lgraphEncoder,'cat','skipConv1');
lgraphEncoder = connectLayers(lgraphEncoder,'skipConv1','add1/in2');


lgraphEncoder = connectLayers(lgraphEncoder,'relu2','skipConv2');
lgraphEncoder = connectLayers(lgraphEncoder,'skipConv2','add2/in2');



layers = [
    featureInputLayer(1)
    embeddingLayer(embeddingDimension,numClasses)
    fullyConnectedLayer(prod(inputSize(1:2)))
    functionLayer(@(X) feature2image(X,[inputSize(1:2) 1]),Formattable=true,Name="emb_reshape")];
 
lgraphE = layerGraph(layersEncoder);
lgraphEncoder = addLayers(lgraphEncoder,layers);
lgraphEncoder = connectLayers(lgraphEncoder,"emb_reshape","cat/in2");
netEncodeSigd1Final = dlnetwork(lgraphEncoder);
analyzeNetwork(netEncodeSigd1Final)
% plot(lgraphEncoder);

% Define Decoder Network Architecture
projectionSize = [64 1 64];
inputSize = [500 1 1];
numInputChannels = size(inputSize,1);
numLatentInputs = numLatentChannels;

layersDecoder = [
    featureInputLayer(numLatentInputs,Name = 'Noise',Normalization="none")
    fullyConnectedLayer(prod(projectionSize),Name ='FC_prepN')
    functionLayer(@(X) feature2image(X,projectionSize),Formattable=true,Name ='ProjectNoise')
    concatenationLayer(3,2,Name="cat")
    transposedConv2dLayer([2 1],16,'Stride',1,'Cropping',[1 0],'Name','tconv1')           
    leakyReluLayer(scale,'Name','relu1')
    dropoutLayer(prob,'Name','drop1')
    transposedConv2dLayer([3 1],16,'Stride',2,'Cropping',[1 0],'Name','tconv2')   
    additionLayer(2,'Name', 'add1')  
    leakyReluLayer(scale,'Name','relu2')
    dropoutLayer(prob,'Name','drop2')
    transposedConv2dLayer([3 1],16,'Stride',2,'Cropping',[1 0],'Name','tconv3')   
    additionLayer(2,'Name', 'add2') 
    leakyReluLayer(scale,'Name','relu3')
    dropoutLayer(prob,'Name','drop3')
    transposedConv2dLayer([4 1],numInputChannels,'Stride',2,'Cropping',[0 0],'Name','tconv4')    
    functionLayer(@(X) 1.5*tanh(X),'Name','scalingTanh')
];

lgraphDecoder = layerGraph(layersDecoder);
 
skiptConv1 = transposedConv2dLayer([3 1],16,'Cropping',[2 0],'Stride',2,'Name','skiptConv1');
skiptConv2 = transposedConv2dLayer([3 1],16,'Cropping',[1 0],'Stride',2,'Name','skiptConv2');

lgraphDecoder = addLayers(lgraphDecoder,skiptConv1);
lgraphDecoder = addLayers(lgraphDecoder,skiptConv2);

lgraphDecoder = connectLayers(lgraphDecoder,'cat','skiptConv1');
lgraphDecoder = connectLayers(lgraphDecoder,'skiptConv1','add1/in2');

lgraphDecoder = connectLayers(lgraphDecoder,'relu2','skiptConv2');
lgraphDecoder = connectLayers(lgraphDecoder,'skiptConv2','add2/in2');

layers = [
    featureInputLayer(1,Name ='LabelsDecoder')
    embeddingLayer(embeddingDimension,numClasses,Name ='embedLabel')
    fullyConnectedLayer(prod(projectionSize(1:2)),Name ='FC_prepL')
    functionLayer(@(X) feature2image(X,[projectionSize(1:2) 1]),Formattable=true,Name="ProjectLabel")];

lgraphDecoder = addLayers(lgraphDecoder,layers);
lgraphDecoder = connectLayers(lgraphDecoder,"ProjectLabel","cat/in2");
netDecodeSigd1Final = dlnetwork(lgraphDecoder);

% plot(lgraphDecoder);
analyzeNetwork(netDecodeSigd1Final)

% Define Model Loss Function

% Specify Training Options
numEpochs = 4000;
miniBatchSize = 38;
initialLearnRate = 0.0002;
decay = 1;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
validationFrequency =10;
executionEnvironment = "CPU";
learnRate = initialLearnRate;


% Train Model
mbqTrain = minibatchqueue(dsTrainVae, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@(X,T) preprocessMiniBatch(X,T), ...
    MiniBatchFormat=["SSCB" "BC"], ...
    PartialMiniBatch="return",...
    OutputEnvironment=executionEnvironment);



mbqTest = minibatchqueue(dsTestVae, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@(X,T) preprocessMiniBatch(X,T), ...
    MiniBatchFormat=["SSCB" "BC"], ...
    PartialMiniBatch="return",...
    OutputEnvironment=executionEnvironment);

% Initialize the parameters for the Adam solver.
trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];

numObservationsTrain = size(XTrain,2);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

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
        [X,T] = next(mbqTrain);
        % Evaluate loss and gradients.
        [loss,gradientsE,gradientsD,lastOutputTrain] = dlfeval(@modelLoss,netEncodeSigd1Final,netDecodeSigd1Final,X,T);
        lossValue = [lossValue loss];
        % Determine learning rate for time-based decay learning rate schedule.
%         learnRate = initialLearnRate/(1 + decay*iteration);
%         learnRate = initialLearnRate;
        
        % Update learnable parameters.
        [netEncodeSigd1Final,trailingAvgE,trailingAvgSqE] = adamupdate(netEncodeSigd1Final, ...
            gradientsE,trailingAvgE,trailingAvgSqE,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        [netDecodeSigd1Final, trailingAvgD, trailingAvgSqD] = adamupdate(netDecodeSigd1Final, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the training progress monitor. 
        recordMetrics(monitor,iteration,TrainingLoss=loss);
        updateInfo(monitor,ExecutionEnvironment = executionEnvironment,LearningRate=learnRate,...
            Epoch=epoch + " of " + numEpochs, Iteration = iteration + " of " + numIterations);

        % Validate network.
%         if iteration == 1 || ~hasdata(mbqTrain)
%         if mod(iteration,validationFrequency) == 0 || iteration == 1
%              [ValReal,ValPred,~,ValGen]= modelPredictions(netEncodeSigd1OrigWithoutValid,netDecodeSigd1OrigWithoutValid,mbqValid);
%              lossValidation = mse(ValGen,ValReal) + KL;
% 
%             % Update plot.
%              recordMetrics(monitor,iteration,ValidationLoss=lossValidation);
%         end

        monitor.Progress = 100*iteration/numIterations;
    end
end

% Plot Loss function in a Normal plot

lossValueDouble = extractdata(lossValue);
figure;
plot(lossValueDouble, 'DisplayName','Training Loss','LineWidth',1);
xlabel("Iterations")
ylabel("Loss Function ")
legend;

figName= "lossDA35Ade.jpg";
figPath = "Results\Paper3Images\";
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',1200);

% Test Model
[XRealTest, XhatTest,TestLabel,XGenTest]= modelPredictions(netEncodeSigd1Final,netDecodeSigd1Final,mbqTest);
size(XhatTest);


errTest = mean((XRealTest-XhatTest).^2,[1 2 3]);
errTestRMSE = sqrt(errTest);


% MSE - Recon Err Test
figure;
histogram(errTest,60, 'FaceColor',"#A2142F","EdgeColor","#A2142F");
xlabel("Reconstruction Error (MSE)");
ylabel("Frequency");
yticklabels(strrep(yticklabels,'-','$-$'));

% To print this histogram plot go to Script ss0p10d2PlotReconErrorDataAugmentation

% RMSE - Might be Recon Err Test
% figure
% histogram(errTestRMSE,60,'FaceColor',"#A2142F","EdgeColor","#A2142F")
% xlabel("Reconstruction Error (RMSE)");
% ylabel("Frequency");
% yticklabels(strrep(yticklabels,'-','$-$'));
% 
% 
% figName = strcat(figPath,figName);
% exportgraphics(gcf,figName,'Resolution',1200);



% Plot a reconstructed signal from Test Set
% n = randi(length(TestLabel),1);
n= 730
ClassReconSeg = TestLabel(n)

Xreal = XRealTest(:,:,:,n);
Xrecon = XhatTest(:,:,1,n);

figure;
plot(Xreal,"DisplayName","Original Segment","LineWidth",1.2);
hold on;
plot(Xrecon,"DisplayName","Reconstructed Segment","LineWidth",1.2);
hold off;
legend

yticklabels(strrep(yticklabels,'-','$-$'));
xlabel("Time (s)");
ylabel("Normalized Drain Current")


% To print the Reconstructed Segments go to script ss0p11d1PlotReconSeg


% Save Networks
filePathOutput = "Results\Paper3Result\";
netEncoderPath = strcat(filePathOutput,"netEncodeSigd1Final.mat");
netDecoderPath = strcat(filePathOutput,"netDecodeSigd1Final.mat");
save(netEncoderPath , "netEncodeSigd1Final");
save(netDecoderPath , "netDecodeSigd1Final");

% Generate new data
structNormalPath = "C:\Users\fesm704\OneDrive - The University of Auckland\DataBase\Paper3DB\sNormalP3Ade35.mat";
load(structNormalPath)


for i = 1:size(s,2)  
    numObservationsNew =200- size(s(i).zscore,2) ;
    idxClass = i;
    ZNew = randn(numLatentInputs,numObservationsNew,"single");
    TNew = repmat(single(idxClass),[1 numObservationsNew]);
    ZNew = dlarray(ZNew,"CB");
    TNew = dlarray(TNew,"CB");
    XGenNew = predict(netDecodeSigd1Final,ZNew,TNew);    
    XGenNewMat = extractdata(XGenNew);
    XGenArray = reshape(XGenNewMat,[500,  size(XGenNew,4)]);
    XGenTable = array2table(XGenArray);
    s(i).CVAESigWFinal =[s(i).zscore XGenTable];
end

save("Results\sNormalP3Ade35.mat","s");

% Helper Functions
% Model Loss Function

function [loss,gradientsE,gradientsD,Y] = modelLoss(netE,netD,X,Label)

% Forward through encoder.
[Z,mu,logSigmaSq] = forward(netE,X,Label);

% Forward through decoder.
Y = forward(netD,Z,Label);

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

function [XReal, XhatSingle,Label,XhatDlArray]= modelPredictions(netE,netD,mbq)

XReal =[];
XhatSingle = [];             % a single array
Label=[];
XhatDlArray = [];
XGeneratedAll= []; % a deep learning array

% Reset mini-batch queue.
reset(mbq);
% Loop over mini-batches.
while hasdata(mbq)
    [Xtest,Ttest] = next(mbq);
    
    XRealLoop = Xtest;

    Label = [Label Ttest];

    % Forward through encoder.
     
    ZGen = predict(netE,Xtest,Ttest);

    % Forward through dencoder.
    XGenerated = predict(netD,ZGen,Ttest);
    XGeneratedAll= cat(4,XGeneratedAll,XGenerated);

    % Extract and concatenate predictions.
    XReal = cat(4,XReal,extractdata(XRealLoop));
    XhatSingle =  cat(4,XhatSingle,extractdata(XGenerated));
end
XhatDlArray = XGeneratedAll;
end
% Mini Batch Preprocessing Function

function [X,T] = preprocessMiniBatch(XCell,TCell)
    % Extract image data from the cell array and concatenate over fourth
    % dimension to add a third singleton dimension, as the channel
    % dimension.
    X = cat(4,XCell{:});

    % Extract label data from cell and concatenate.
    T = cat(2,TCell{:})'; 

end
% Mini-Batch Predictors Preprocessing Function

function X = preprocessMiniBatchPredictors(XCell)
X = cat(4,XCell{:});
end

