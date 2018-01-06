function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1); %行
n = size(X, 2); %列

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);   %共有 num_labels 个分类器，特征数 n + 1

% Add ones to the X data matrix
X = [ones(m, 1) X]; 

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% Set Initial theta
K = num_labels; %共分 K 类
initial_theta = zeros(n + 1, K);    % initial_theta的转置 与 all_theta的形状相同
% c = [10 1 2 3 4 5 6 7 8 9]; % 把0映射到10,而1-9与自身一一对应


for i=1:K
    % i等于几，就表示在训练第几个分类器，训练时不必管这一类对应的数字是多少，逻辑上对应就行
    % 如果想知道第i类对应哪个数字，很容易知道
    c = i;
    initial_theta_I = initial_theta(:,i);   % 第i类
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [trained_theta_I] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                initial_theta_I, options);
    initial_theta(:,i) = trained_theta_I;
end

all_theta = initial_theta';


% =========================================================================


end
