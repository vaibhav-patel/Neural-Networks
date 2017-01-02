clear all
close all
clc

% Load the training data.....

addpath('kMeans');

data = load('VC.tra'); %Specify input file

n = size(data,2);
inp = n-1; % Initialize to No of features

X = data(:,1:n-1); % Specify range of features
Y = data(:,n); % Specify range of output class


%[~,Y] = max(Ydata,[],2);
m = size(X,1);

numUnique = size(unique(Y),1);
out = numUnique;
% Learning rates
lam = 0.001;
lcm = 0.001;
lbm = 0.001;
Centers = [];
betas = [];
centresPerClass = 35; % Initialize to number of neurons you require per class

Ydata = ones(m,out) * 0;
for i = 1:m
    Ydata(i,Y(i)) = 1;
end

%randidx = randperm(size(X, 1));
%This part here is for splitting the dataset into crossvalidation and training
% Ignore if you are not splitting. But don't comment it. Let it be with no changes.
split_point = round(1*m);
xtrain = X(1:split_point,:);
ytrain = Y(1:split_point,:);
ydata_train = Ydata(1:split_point,:);
mtrain = size(xtrain,1);

%{
Xtest = X(split_point+1:m,:);
Ytest = Y(split_point+1:m,:);
mtest = size(Xtest,1);
%}

for c = 1:numUnique
    Xc = xtrain((ytrain==c),:);

    init_centroids = Xc(1:centresPerClass,:);

    % number of iterations for kMeans = 100
    [Centroids_c, memberships_c] = kMeans(Xc, init_centroids, 100);

    toRemove = [];

    for (i = 1 : size(Centroids_c, 1))

        if (sum(memberships_c == i) == 0)
            toRemove = [toRemove; i];
        end
    end

    if (~isempty(toRemove))

        Centroids_c(toRemove, :) = [];

        memberships_c = findClosestCentroids(Xc, Centroids_c);
    end

    numRBFNeurons = size(Centroids_c,1);

    sigmas = zeros(numRBFNeurons,1);
    for i = 1 : numRBFNeurons

        center = Centroids_c(i, :);

        members = X((memberships_c == i), :);

        differences = bsxfun(@minus, members, center);

        sqrdDiffs = sum(differences .^ 2, 2);
        distances = sqrt(sqrdDiffs);
        sigmas(i, :) = mean(distances);
    end

    betas_c = 1 ./ (2 .* sigmas .^ 2);

    Centers = [Centers;Centroids_c];
    betas = [betas; betas_c];
end

%size(betas)
%size(Centers)
numRBFNeurons = size(Centers, 1);

X_activ = zeros(mtrain, numRBFNeurons);

% For each training example...
for (i = 1 : mtrain)

    input = xtrain(i, :);

    diffs = bsxfun(@minus, Centers, input);

    sqrdDists = sum(diffs .^ 2, 2);


    z = exp(-betas .* sqrdDists);

    X_activ(i, :) = z';

end
X_activ = [ones(mtrain, 1), X_activ];

outWeights = zeros(numRBFNeurons+1, numUnique);

for (c = 1 : numUnique)

    y_c = (ytrain == c);

    outWeights(:, c) = pinv(X_activ' * X_activ) * X_activ' * y_c;
end


conftra = zeros(out,out);
oldbetas = zeros(size(betas));
%{
epo = 1500; % Number of epochs. Initialize to your required value
for j = 1:epo
    sumerr = 0;
    miscla = 0;
    for i = 1:mtrain
        input = xtrain(i, :);
        tt = ydata_train(i,:);

        diffs = bsxfun(@minus, Centers, input);

        sqrdDists = sum(diffs .^ 2, 2);


        z = exp(-betas .* sqrdDists);

        z = [1;z];
        Yo = z' * outWeights;
        err = tt - Yo;
        if(Yo'*tt>1)
            err = zeros(size(tt));
        end
        outWeights = outWeights + lam * (z * err);
        %diffs = bsxfun(@times,diffs,betas);
        %keyboard
        outW = outWeights(2:end,:);

        zupdate = z(2:end,:);
        %keyboard
        Centers = Centers + lcm * repmat((outW*err').*(zupdate.*betas),1,inp).*(repmat(input,numRBFNeurons,1)-Centers);
        oldBetas = betas;
        betas = betas - lbm * (outW*err').*(zupdate.*betas).*(sum((repmat(input,numRBFNeurons,1)-Centers) .^ 2, 2)) ;
        cond = betas <= 0;
        betas(cond == 1) = oldbetas(cond==1);
        ca = ytrain(i);
        [~,cp] = max(Yo);

        if(ca~=cp)
            miscla = miscla + 1;
        end
    end
    %disp(1-(miscla/m))
    %miscla

end
%}
train_out = [];

for i = 1:mtrain
    input = xtrain(i, :);

    diffs = bsxfun(@minus, Centers, input);

    sqrdDists = sum(diffs .^ 2, 2);


    z = exp(-betas .* sqrdDists);

    z = [1;z];
    Yo = z' * outWeights;

    ca = ytrain(i);

    [~,cp] = max(Yo);
    train_out = [train_out;cp];
    conftra(ca,cp) = conftra(ca,cp) + 1;
end
save ae_train.txt train_out
acc = 0;
for i = 1:out
    acc = acc + conftra(i,i);
end
disp((acc/mtrain)*100)

acc = 0;
ni = 0;
for i = 1:out
    acc = acc + conftra(i,i)/sum(conftra,2)(i);
    
end
disp((acc/out)*100)

acc = 1;
ni = 0;
for i = 1:out
    acc = acc * 100 * conftra(i,i)/sum(conftra,2)(i);
    
end
disp(power(acc,1/out))

conftra


conftes = zeros(out,out);


Ntest = load('VC.tes'); % Testing set input
Xtest = Ntest(:,1:n-1); % Range of features in testing set
Ytest = load('VC.cla'); % Range of output class in testing set
mtest = size(Xtest,1);
out = [];


for i = 1:mtest
    input = Xtest(i, :);

    diffs = bsxfun(@minus, Centers, input);

    sqrdDists = sum(diffs .^ 2, 2);


    z = exp(-betas .* sqrdDists);
    z = [1;z];

    Yo = z' * outWeights;
    ca = Ytest(i);
    [~,cp] = max(Yo);
    conftes(ca,cp) = conftes(ca,cp) + 1;
    out = [out;cp];
end

save ae_test.txt out; % Saving output in a txt file
acc = 0;
for i = 1:size(conftes,1)
    acc = acc + conftes(i,i);
end
disp(acc*100/mtest)


acc = 0;
for i = 1:size(conftes,1)
    acc = acc + conftes(i,i)/sum(conftes,2)(i);
    
end
disp((acc/size(conftes,1))*100)

acc = 1;
for i = 1:size(conftes,1)
    acc = acc * 100 * conftes(i,i)/sum(conftes,2)(i);
    
end
disp(power(acc,1/size(conftes,1)))

conftes
