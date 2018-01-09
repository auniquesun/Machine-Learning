function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
alternatvie_parameters = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
num_alternatvie_parameters = length(alternatvie_parameters);    % 求一个向量的长度用length，而不是len
error = 100000000;  %将error置为一个大整数
C_tag = 0;
sigma_tag = 0;
for i=1:num_alternatvie_parameters
    C = alternatvie_parameters(i,1);
    for j=1:num_alternatvie_parameters
        sigma = alternatvie_parameters(j,1);
        model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));
        predictions = svmPredict(model,Xval);
        predict_error = mean(double(predictions ~= yval));
        if predict_error < error
            error = predict_error;
            C_tag = i;
            sigma_tag = j;
        end    
    end    
end

C = alternatvie_parameters(C_tag);
sigma = alternatvie_parameters(sigma_tag);
% =========================================================================

end
