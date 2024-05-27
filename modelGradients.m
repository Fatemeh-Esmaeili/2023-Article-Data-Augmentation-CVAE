function [infGrad, genGrad] = modelGradients(encoderNet, decoderNet, x,y)
[z, zMean, zLogvar] = sampling(encoderNet, x,y);
xPred = sigmoid(predict(decoderNet, z,y));
loss = ELBOloss(x, xPred, zMean, zLogvar);
[genGrad, infGrad] = dlgradient(loss, decoderNet.Learnables, ...
    encoderNet.Learnables);
end