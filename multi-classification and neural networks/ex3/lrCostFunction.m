function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));  % theta 的形状为 400 x 1
grad_row_num = size(grad,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% 求costFunction，向量化的实现
Z = X * theta;
hypo = sigmoid(Z);
first_item = y' * log(hypo);

second_item = (1 - y)' * log(1 - hypo);

len_theta = length(theta);
regularized_theta = theta(2:len_theta);
regularized_item = sum(regularized_theta.^2);

J = -1 / m * (first_item + second_item) + lambda / 2 / m * regularized_item;

% 求gradient，向量化的实现
no_regularized_item = X' * (hypo - y);
theta_tmp = theta;
theta_tmp(1,1) = 0;

grad = 1 / m * no_regularized_item + lambda / m * theta_tmp;
% 非向量化的实现
% for j=1:grad_row_num
%     grad(j,1) = 1 / m * (hypo - y)' * X(:,j);
% end
% =============================================================

grad = grad(:);

end
