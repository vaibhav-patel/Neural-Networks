% Program for RBF classification using pseudo inverse method
% Modification: The centres are initialized by using k-means clustering

% Author: Rajdeep Pinge

clear all
close all
clc

% Load the training data..................................................
Ntrain=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Set 5\Wine.tra');
[TD,in] = size(Ntrain);

%Load testing data
NFeature=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Set 5\Wine.tes');
[NTestD,~]=size(NFeature);

NAns=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Results\Group 5\Wine.cla');


% Initialize the Algorithm Parameters.....................................
inp = in-1;          % No. of input neurons
K = 20;             % No. of hidden neurons

% output of the training data
trueOut = Ntrain(:, inp+1:end);

%find out the number of classes
Nclasses = size(unique(trueOut, 'rows'), 1);

out = Nclasses;            % No. of Output Neurons

% create the output vectors for the true/actual outputs for training data
Ytrue = zeros(TD, Nclasses);
for i = 1 : TD
   Ytrue(i, :) = -1;
   % can take another loop to traverse column-wise
   Ytrue(i, trueOut(i, 1)) = 1;
end

% create the output vectors for the true/actual outputs for testing data
YAns = zeros(NTestD, Nclasses);
for i = 1 : NTestD
   YAns(i, :) = -1;
   % can take another loop to traverse column-wise
   YAns(i, NAns(i, 1)) = 1;
end


% CROSS VALIDATION INITIALIZATION

% cross validation factor
CVFactor = 0.9;

NTD = floor(TD * CVFactor);     %training data after cross validation

NCV = TD - NTD;     %cross validation testing sample

% matrices to store the data to optimize the parameters to be used
%resultcheck_tra = zeros(min(TD,200)+1-(inp+out+1), 2);
%resultcheck_tes = zeros(min(TD,200)+1-(inp+out+1), 2);


% loop to get optimized number of hidden neurons initially
% once the optimum neurons are fixed, the loop is disabled

%for K = (inp + out + 1) : min(TD, 200)

    % training data features
    xinp = Ntrain(:, 1:inp);
    
    % k-Means code by ChrisMcCormick has been used
    % initialize centres using k-means clustering
    
    % first initialize centres randomly
    u = kMeansInitCentroids(xinp, K);
    
    % apply kMeans to alter the centres
    kM_max_iter = 1000;
    [u, u_member_cluster] = kMeans(xinp, u, kM_max_iter);

   
    %%% setting spread sigma
   
    % find the maximum distace among centres
    dist = zeros(K, K);
    for i = 1 : K
       for j = i+1 : K
          dist(i,j) = sqrt( sum((u(i, :) - u(j, :)).^2) );
          dist(j,i) = dist(i, j);
       end
    end

    dmax = max(dist(:));

    % set spread
    sigma = dmax / sqrt(K);

    % intermediate matrix storing function values of all the gaussians for
    % all the inputs and centres
    G = zeros(TD, K);

    % build G matrix
    for i = 1 : TD
       G(i,:) = exp( - sum ( (( repmat(xinp(i,:),K,1) ) - u(:,:)).^2, 2 ) / (2*sigma*sigma) ); 
    end

    % find weights using pseudo inverse of G matrix
    Weight = pinv(G) * Ytrue;

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % test over training data
   
    % build intermediate matrix for the data
    G = zeros(TD, K);
    for i = 1 : TD
       G(i,:) = exp( - sum ( (( repmat(xinp(i,:),K,1) ) - u(:,:)).^2, 2 ) / (2*sigma*sigma) ); 
    end

    % output of the testing over training data
    output_tra = G * Weight;

    % confusion matrix for training data 
    conf_mat_tra = zeros(out,out);

    % predicted label
    [~,pred_label_tra]= max(output_tra,[],2); % the class is taken as max prob in the coded vec

    % build the confusion matrix
    for sa=1:size(pred_label_tra)
        conf_mat_tra(trueOut(sa),pred_label_tra(sa))=conf_mat_tra(trueOut(sa),pred_label_tra(sa))+1;
    end

    %disp(conf_mat_tra) % we have obtained the confusion matrix
    
    % store actual data and predicted data in one matrix
    predict_tra = [trueOut pred_label_tra];

    %correct classifications
    correct_tra = sum(diag(conf_mat_tra));

    %overall accuracy
    overall_acc_tra = 100*correct_tra/TD;

    
    %%% to get average accuracy
    % get all the unique class labels
    classLabel = unique(trueOut, 'rows');

    % using histogram to get frequency of labels
    [labelCount_tra, classLabel] = hist(Ntrain(:,end), unique(classLabel));

    %average accuracy
    avg_acc_tra = 100/out * sum(diag(conf_mat_tra)./labelCount_tra');

    %geometric-mean accuracy
    geo_mean_acc_tra = nthroot(prod(100*diag(conf_mat_tra)./labelCount_tra'),out);

    
    % resultcheck_tra(K-(inp+out), :) = [K overall_acc_tra];

    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test the network for the given testing data........................................................

    % G matrix for testing data
    G_tes = zeros(NTestD, K);

    % build G matrix
    for i = 1 : NTestD
       G_tes(i,:) = exp( - sum ( (( repmat(NFeature(i,:),K,1) ) - u(:,:)).^2, 2 ) / (2*sigma*sigma) ); 
    end

    % store output of the testing data
    output_tes = G_tes * Weight;

    % confusion matrix for training data 
    conf_mat_tes = zeros(out, out);

    % predicted label
    [~,pred_label_tes]= max(output_tes,[],2); % the class is taken as max prob in the coded vec


    % build the confusion matrix for testing data
    for sa=1:size(pred_label_tes)
     conf_mat_tes(NAns(sa),pred_label_tes(sa))=conf_mat_tes(NAns(sa),pred_label_tes(sa))+1;
    end

    %disp(conf_mat_tes) % we have obtained the confusion matrix

    % store actual data and predicted data in one matrix
    predict_tes = [NAns pred_label_tes];
    
    %correct classifications
    correct_tes = sum(diag(conf_mat_tes));

    %overall accuracy
    overall_acc_tes = 100*correct_tes/NTestD;


    %%% to get average accuracy
    % using histogram to get frequency of labels
    [labelCount_tes,classLabel] = hist(NAns, unique(classLabel));

    %average accuracy
    avg_acc_tes = 100/out * sum(diag(conf_mat_tes)./labelCount_tes');

    %geometric-mean accuracy
    geo_mean_acc_tes = nthroot(prod(100*diag(conf_mat_tes)./labelCount_tes'),out);

    
    %resultcheck_tes(K-(inp+out), :) = [K overall_acc_tes];

       
%end


% plotting data obtained from loop to get the optimum number of hidden
% neurons

%plot(resultcheck_tra(:, 1), resultcheck_tra(:, 2))
%hold on
%plot(resultcheck_tes(:, 1), resultcheck_tes(:, 2), '-r')
%xlabel('No. of Hidden Neurons')
%ylabel('Overall Accuracy')
%Title('Graph to find optimum neurons based on results')
%legend('Training Data','Testing Data','Location','northwest')
%axis([10,70,85,105])