%-----------------------Part 1----------------------------------
%------Input training and testing data--------------------------
trainData = csvread('train.txt');
trainingDataSize = size(trainData,1);

testData = csvread('test.txt');
testDataSize  = size(testData,1);

%------Training the gaussian naive bayes classifier--------------
fprintf('Naive Bayes:\n');
%Separating all the training samples into separate classes
class1 = [];
class2 = [];
class3 = [];
class4 = [];
class5 = [];
class6 = [];
class7 = [];

for i =1:trainingDataSize
    if trainData(i,11) == 1
        class1 = cat(1,class1,trainData(i,2:10));
    elseif trainData(i,11) == 2
        class2 = cat(1,class2,trainData(i,2:10));
    elseif trainData(i,11) == 3
        class3 = cat(1,class3,trainData(i,2:10));
    elseif trainData(i,11) == 4
        class4 = cat(1,class4,trainData(i,2:10));    
    elseif trainData(i,11) == 5
        class5 = cat(1,class5,trainData(i,2:10));
    elseif trainData(i,11) == 6
        class6 = cat(1,class6,trainData(i,2:10));
    elseif trainData(i,11) == 7
        class7 = cat(1,class7,trainData(i,2:10));
    end
end

%Generating the summary map from all the classes 
classMap = containers.Map([1 2 3 4 5 6 7],{class1 class2 class3 class4 class5 class6 class7});

%Calculating the std deviation and mean of all the features for all the
%classes
stdMap = containers.Map('KeyType','int32','ValueType','any');
meanMap = containers.Map('KeyType','int32','ValueType','any');

for i=1:7
    if size(classMap(i),1) ~= 0
        stdMap(i) = std(classMap(i),1,1);
        meanMap(i) = mean(classMap(i),1);
    end
end

%-------------Finding the accuracy of training and testing data------

%Training
calculatedY = [];
for trainIndex=1:trainingDataSize
    prob = [];
    for i=1:7
        if size(classMap(i),1) ~= 0 
            condProb = normpdf(trainData(trainIndex,2:10), meanMap(i),stdMap(i));
            condProb = prod(condProb);
            priorProb = size(classMap(i),1)/trainingDataSize;
            prob = cat(2,prob,condProb*priorProb);
        else
            prob = cat(2,prob,0);
        end
    end
    [maxval,argmax] = max(prob);
    calculatedY = [calculatedY;argmax];
end

accuratePredictions = (calculatedY == trainData(:,11));
trainingAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
fprintf('Training Accuracy = %4.2f\n',trainingAccuracy);

    
%Testing
calculatedY = [];
for testIndex=1:testDataSize
    prob = [];
    for i=1:7
        if size(classMap(i),1) ~= 0 
            condProb = normpdf(testData(testIndex,2:10), meanMap(i),stdMap(i));
            condProb = prod(condProb);
            priorProb = size(classMap(i),1)/trainingDataSize;
            prob = cat(2,prob,condProb*priorProb);
        else
            prob = cat(2,prob,0);
        end
    end
    [maxval,argmax] = max(prob);
    calculatedY = [calculatedY;argmax];
end

accuratePredictions = (calculatedY == testData(:,11));
testAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
fprintf('Test Accuracy = %4.2f\n\n',testAccuracy);


%---------------------Part 2---------------------------------
fprintf('K Nearest Neighbours:\n');
kValues = [1 3 5 7];
x = trainData(:,2:10);
y = trainData(:,11);

xmean = mean(x);
xstd = std(x);

x = zscore(x);

xtest = testData(:,2:10);
ytest = testData(:,11);
for i=1:testDataSize
    xtest(i,:) = (xtest(i,:)- xmean)./ xstd;
end


%KNN Training Accuracy
fprintf('KNN Training results:\n');
for k = kValues
    fprintf('For k = %d :\n',k);
    l1Accuracy = zeros(trainingDataSize,1);
    l2Accuracy = zeros(trainingDataSize,1);

    for i=1:trainingDataSize
        subtractedRows = zeros(size(x));
        for j=1:trainingDataSize
            subtractedRows(j,:) = x(j,:) - x(i,:);
        end

        norm1 = sum(abs(subtractedRows),2);
        norm1(i) = inf;
        
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm1Index] = min(norm1);
            kNearestNeighbours(krep) = y(minNorm1Index);
            norm1(minNorm1Index) = inf;
        end
        
        l1Accuracy(i) = (mode(kNearestNeighbours) == y(i)); 

        norm2 = sqrt(sum(subtractedRows.^2,2));
        norm2(i) = inf;
        
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm2Index] = min(norm2);
            kNearestNeighbours(krep) = y(minNorm2Index);
            norm2(minNorm2Index) = inf;
        end
        l2Accuracy(i) = (mode(kNearestNeighbours) == y(i));
    end

    l1Accuracy = sum(l1Accuracy)*100/trainingDataSize;
    l2Accuracy = sum(l2Accuracy)*100/trainingDataSize;

    fprintf('L1 training accuracy = %4.2f\n', l1Accuracy);
    fprintf('L2 training accuracy = %4.2f\n\n', l2Accuracy);

end


%KNN Testing accuracy
fprintf('KNN Testing results:\n');
for k = kValues
    fprintf('For k = %d :\n',k);
    l1Accuracy = zeros(testDataSize,1);
    l2Accuracy = zeros(testDataSize,1);

    for i=1:testDataSize
        subtractedRows = zeros(size(x));
        for j=1:trainingDataSize
            subtractedRows(j,:) = x(j,:) - xtest(i,:);
        end

        norm1 = sum(abs(subtractedRows),2);
        norm1(i) = inf;
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm1Index] = min(norm1);
            kNearestNeighbours(krep) = y(minNorm1Index);
            norm1(minNorm1Index) = inf;
        end
        l1Accuracy(i) = (mode(kNearestNeighbours) == ytest(i)); 

        
        norm2 = sqrt(sum(subtractedRows.^2,2));
        norm2(i) = inf;
        
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm2Index] = min(norm2);
            kNearestNeighbours(krep) = y(minNorm2Index);
            norm2(minNorm2Index) = inf;
        end
        l2Accuracy(i) = (mode(kNearestNeighbours) == ytest(i));
    end

    l1Accuracy = sum(l1Accuracy)*100/testDataSize;
    l2Accuracy = sum(l2Accuracy)*100/testDataSize;

    fprintf('L1 testing accuracy = %4.2f\n', l1Accuracy);
    fprintf('L2 testing accuracy = %4.2f\n\n', l2Accuracy);
end
