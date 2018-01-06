function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% cross_val数据集的大小
num_cross_val = size(yval,1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

% theta = trainLinearReg(X, y, lambda);

% difference_h_and_y = X * theta - y;
% cross_val_difference_h_and_y = Xval * theta - yval;
% cross_val_sum_rows = sum(cross_val_difference_h_and_y.^2);
for i=1:m
    theta = trainLinearReg(X(1:i,:), y(1:i), lambda);
    difference_h_and_y = X(1:i,:) * theta - y(1:i);
    rows = difference_h_and_y(1:i,:);
    sum_rows = sum(rows.^2);
    error_train(i,1) = 1 / 2 / i  * sum_rows;
    
    cross_val_difference_h_and_y = Xval * theta - yval;
    cross_val_sum_rows = sum(cross_val_difference_h_and_y.^2);
    %特别注意，这里cross_val数据集的大小不同于train数据集的大小，所以要除以的是 num_cross_val，而不是 i
    error_val(i,1) = 1 / 2 / num_cross_val * cross_val_sum_rows;
end    

% 特别注意，用多项式回归，当 lambda = 0时，模型过拟合，
% 学习曲线表现为 cross_val 随着先高后低，train_val 几乎为0，与横轴“重合”，如果不仔细观察，几乎看不到 train_val 的曲线



% -------------------------------------------------------------

% =========================================================================

end
