function pa = psychAdapt(cmd, varargin)
%{ 
    cmd is a STRING input and should be one of the following:

    'setup'             (prepare the adaptive track with initial values
    'train'             (train the adaptive model around a threshold estimate
    'test'              (use this for upating the model after each trial with the most
                        recent information: threshold used and trial accuracy)
    'plotTraining       (plot the logistic psychometric function with the
                        threshold estimate "starred" on the plot)
    'outputEstimate'    (display the threshold at the target accuracy level)
  
    variable inputs should be in pairs('input', val), and can be any of the following:

    'model' :structure                ----  the model structure that saves all adaptive track information (needs to be passed in each time during training and updating)
    'targetAcc': number (> 0 and < 1) ----  the target accuracy that the model will adapt performance to
    'threshGuess': number             ----  the guess for the threshold at the targetAcc (supply this if training, if updating this will be computed for you) 
    'min': number                     ----  minimum stimulus value
    'max': number                     ----  maximum stimulus value
    'acc': number 0 or 1              ----  accuracy for the trial that was shown (0 or 1)
    'name': string                    ----  optional name for this adaptive track (ex. audio, visual, haptic, etc.)
    'probeLength': number             ----  number of trials for training
    'stimulusValue': number           ----  value used for the most recent stimulus
%}

switch cmd
    
    case 'setup'
        pa.train.rt = [];
        pa.train.trialIdx = 0;
        pa.test.trialIdx = 0;
        pa.train.targetAcc   = cell2mat(varargin(find(strcmp(varargin, 'targetAcc'))+1));
        pa.test.targetAcc    = cell2mat(varargin(find(strcmp(varargin, 'targetAcc'))+1));
        pa.train.threshGuess = cell2mat(varargin(find(strcmp(varargin, 'threshGuess'))+1));
        pa.train.min         = cell2mat(varargin(find(strcmp(varargin, 'min'))+1));
        pa.train.max         = cell2mat(varargin(find(strcmp(varargin, 'max'))+1));
        pa.train.probeLength = cell2mat(varargin(find(strcmp(varargin, 'probeLength'))+1));
        %{
        pa.train.probeVals   = normrnd(pa.train.threshGuess, pa.train.sdGuess, [1 pa.train.probeLength]);
        pa.train.probeVals(pa.train.probeVals > pa.train.max) = pa.train.max;
        pa.train.probeVals(pa.train.probeVals < pa.train.min) = pa.train.min;
        pa.train.probeVals(pa.train.probeVals > pa.train.threshGuess+pa.train.sdGuess) = pa.train.threshGuess+pa.train.sdGuess;
        pa.train.probeVals(pa.train.probeVals < pa.train.threshGuess-pa.train.sdGuess) = pa.train.threshGuess-pa.train.sdGuess;
        %}
        pa.train.probeVals = Shuffle(linspace(pa.train.min,pa.train.max,pa.train.probeLength));
        pa.train.stimVal = pa.train.threshGuess;
        pa.stimVal = pa.train.stimVal;
        pa.test.betaVals = [];
        pa.train.name = 'psychAdapt';
        pa.test.name = 'psychAdapt';
        pa.train.name = cell2mat(varargin(find(strcmp(varargin, 'name'))+1));
        pa.test.name = cell2mat(varargin(find(strcmp(varargin, 'name'))+1));
    case 'train'
        pa                   = cell2mat(varargin(find(strcmp(varargin, 'model'))+1));
        pa.train.trialIdx    = pa.train.trialIdx + 1;
        i                    = pa.train.trialIdx;
        if ~isempty(cell2mat(varargin(find(strcmp(varargin, 'rt')))))
            pa.train.rt(i) = cell2mat(varargin(find(strcmp(varargin, 'rt'))+1));
        end 
        pa.train.acc(i)      = cell2mat(varargin(find(strcmp(varargin, 'acc'))+1));
        pa.train.trainAcc    = mean(pa.train.acc);
        pa.train.stimVal     = pa.train.probeVals(i);
        pa.train.stimulusVals(i) = cell2mat(varargin(find(strcmp(varargin, 'stimulusValue'))+1));
        pa = computeTrainingThreshold(pa); 
        pa.stimVal = pa.train.stimVal;
    case 'computeThreshold' % call this after training to set starting point for testing
        pa = cell2mat(varargin(find(strcmp(varargin, 'model'))+1));
        pa.stimVal = pa.test.stimVal;
    case 'test'
        pa = cell2mat(varargin(find(strcmp(varargin, 'model'))+1));
        pa.test.min = pa.train.min;
        pa.test.max = pa.train.max;
        pa.test.trialIdx = pa.test.trialIdx + 1;
        i = pa.test.trialIdx;
        if ~isempty(cell2mat(varargin(find(strcmp(varargin, 'rt')))))
            pa.test.rt(i) = cell2mat(varargin(find(strcmp(varargin, 'rt'))+1));
        end 
        pa.test.acc(i) = cell2mat(varargin(find(strcmp(varargin, 'acc'))+1));
        pa.test.stimulusVals(i) = cell2mat(varargin(find(strcmp(varargin, 'stimulusValue'))+1));
        pa.test.testAcc    = mean(pa.test.acc);
        pa = updateTestingModel(pa);
        pa.stimVal = pa.test.stimVal;
    case 'plotTraining'
        pa = cell2mat(varargin(find(strcmp(varargin, 'model'))+1));
        [acc, aI] = sort(pa.train.acc);
        acc = acc';
        stimVals = pa.train.stimulusVals(aI)';
        n = ones(size(acc));
        [b,~,stats] = glmfit(stimVals,[acc n],'binomial','link','logit');
        %[b,~,stats] = glmfit(stimVals,[acc n],'binomial','link','logit','weights',w(aI));
        yfit = glmval(b,stimVals,'logit');
        [yfits, I] = sort(yfit);
        if b(2) < 0
            threshGuess = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + b(1)) / abs(b(2));
        else
            threshGuess = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + abs(b(1))) / b(2);
        end
        pa.train.threshGuessAtTargetAcc = threshGuess;
        figure;
        %plot(stimVals(I), acc(I)./n,'o',stimVals(I),yfits./n,'-','LineWidth',2, threshGuess,pa.train.targetAcc,'*');
        plot(stimVals(I), acc(I)./n,'o',stimVals(I),yfits./n,'-','LineWidth',2);
        title(pa.train.name);
        hold on;
        plot(threshGuess,pa.train.targetAcc,'*k');
        x = [threshGuess threshGuess];
        y = [0 pa.train.targetAcc];
        line(x,y,'Color','red','LineStyle','--')
        x = [0 threshGuess];
        y = [pa.train.targetAcc pa.train.targetAcc];
        line(x,y,'Color','red','LineStyle','--')
        hold off;
        
    case 'plotTesting' %includes data from training plot too!
        pa = cell2mat(varargin(find(strcmp(varargin, 'model'))+1));
        targetAcc = pa.test.targetAcc;
        %missW = targetAcc; %taret acc can't be less than .50
        %hitW = 1-targetAcc;
        missW = 1; %taret acc can't be less than .50
        hitW = 1;
        trainW = 1;
        trainAcc = pa.train.acc;
        testAcc = pa.test.acc;
        allAcc = [trainAcc testAcc]; 
        %allAcc = [testAcc];
        [acc, aI] = sort(allAcc);
        trainVals = pa.train.stimulusVals;
        testVals = pa.test.stimulusVals;
        allVals = [trainVals testVals];
        testMiss = find(testAcc == 0);
        testHit = find(testAcc == 1);
        %testW = sqrt(1:length(testAcc))+testingFactor;
        testW = zeros(size(testAcc));
        testW(testMiss) = missW;
        testW(testHit) = hitW;
        w = [zeros(size(trainVals))+trainW testW];
        w = w';
        %w = ones(size(allAcc));
        %allVals = [testVals];
        acc = acc';
        stimVals = allVals(aI)';
        n = ones(size(acc));
        %[b,~,stats] = glmfit(stimVals,[acc n],'binomial','link','logit');
        [b,~,stats] = glmfit(stimVals,[acc n],'binomial','link','logit','weights',w(aI)); % with weights
        %[b,~,stats] = glmfit(stimVals,[acc n],'binomial','link','logit'); % no weights 
        yfit = glmval(b,stimVals,'logit'); %only needed if plotting
        [yfits, I] = sort(yfit); %only needed if plotting
        if b(2) < 0
            threshGuess = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + b(1)) / abs(b(2));
        else
            threshGuess = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + abs(b(1))) / b(2);
        end
        if threshGuess > pa.test.max
            threshGuess = pa.test.max;
        end
        if threshGuess < pa.test.min
            threshGuess = pa.test.min;
        end
        figure;
        plot(stimVals(I), acc(I)./n,'o',stimVals(I),yfits./n,'-','LineWidth',2);
        title(pa.test.name);
        hold on;
        plot(threshGuess,pa.test.targetAcc,'*k');
        x = [threshGuess threshGuess];
        y = [0 pa.train.targetAcc];
        line(x,y,'Color','red','LineStyle','--')
        x = [0 threshGuess];
        y = [pa.train.targetAcc pa.train.targetAcc];
        line(x,y,'Color','red','LineStyle','--')
        hold off;
        
    case 'outputEstimate'
        
    otherwise
        error('command: %s is not recognized by the psychAdapt function', cmd);       
end

end %end psychAdapt


function pa = updateTestingModel(pa)
%hitWeightFactor = 10;
%missWeightFactor = hitWeightFactor - (hitWeightFactor*pa.test.targetAcc);
targetAcc = pa.test.targetAcc;
missW = 1; %taret acc can't be less than .50
hitW = 1;
%missW = targetAcc; %taret acc can't be less than .50
%hitW = 1-targetAcc;
trainW = 1;
testingFactor = 0;
trainAcc = pa.train.acc;
testAcc = pa.test.acc;
allAcc = [trainAcc testAcc];
[acc, aI] = sort(allAcc);
trainVals = pa.train.stimulusVals;
testVals = pa.test.stimulusVals;
allVals = [trainVals testVals];
testMiss = find(testAcc == 0);
testHit = find(testAcc == 1);
%testW = sqrt(1:length(testAcc))+testingFactor;
testW = zeros(size(testAcc));
testW(testMiss) = missW;
testW(testHit) = hitW;
w = [zeros(size(trainVals))+trainW testW];
w = w';
acc = acc';
stimVals = allVals(aI)';
n = ones(size(acc));
[b,~,stats] = glmfit(stimVals,[acc n],'binomial','link','logit','weights',w(aI));
pa.test.betaVals = [pa.test.betaVals b];
if b(2) < 0
    pa.test.threshGuess = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + b(1)) / abs(b(2));
    midGuess = (log(0.5/(1-0.5)) + b(1)) / abs(b(2));
else
    pa.test.threshGuess = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + abs(b(1))) / b(2);
    midGuess = (log(0.5/(1-0.5)) + abs(b(1))) / b(2);
end
if pa.test.threshGuess < pa.test.min % if weird values from logistic fit, make a sensible random value
    pa.test.threshGuess = pa.test.min + (pa.test.max-pa.test.min).*rand(1,1);
elseif pa.test.threshGuess > pa.test.max
    pa.test.threshGuess = pa.test.min + (pa.test.max-pa.test.min).*rand(1,1);
end
threshGuess = pa.test.threshGuess;
if midGuess > threshGuess
    midGuess = pa.test.max/2;
end
if midGuess < pa.test.min
    midGuess = pa.test.min;
end
if pa.test.testAcc < pa.test.targetAcc
    pa.test.stimVal = threshGuess + (pa.test.max-threshGuess).*rand(1,1); % make easier
else
    pa.test.stimVal = threshGuess - (threshGuess-midGuess).*rand(1,1); % make harder
end
if pa.test.stimVal > pa.test.max
    pa.test.stimVal = pa.test.max;
end
if pa.test.stimVal < pa.test.min
    pa.test.stimVal = pa.test.min;
end
end %end updateModel


function pa = computeTrainingThreshold(pa)
[acc, aI] = sort(pa.train.acc);
acc = acc';
stimVals = pa.train.stimulusVals(aI)';
n = ones(size(acc));
b = glmfit(stimVals,[acc n],'binomial','link','logit');
yfit = glmval(b,stimVals,'logit');
if b(2) < 0
    pa.test.stimVal = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + b(1)) / abs(b(2));%(log(pa.test.targetAcc/(1-pa.test.targetAcc)) + abs(b(1))) / b(2);
else
    pa.test.stimVal = (log(pa.train.targetAcc/(1-pa.train.targetAcc)) + abs(b(1))) / b(2);
end
end %end computeTrainingThreshold


