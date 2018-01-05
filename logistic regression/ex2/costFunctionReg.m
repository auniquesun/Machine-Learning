function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = X * theta;
hypothesis = sigmf(z,[1,0]);
y_ = 1 - y;
hypothesis_ = 1 - hypothesis;
item1 = y' * log(hypothesis) + y_' * log(hypothesis_);

all_theta = theta' * theta;
all_substract_first_theta = sum(all_theta) - theta(1,1) * theta(1,1);
item2 = all_substract_first_theta;

J = -1 / m * item1 + lambda / 2 / m * item2;

product1 = hypothesis - y;
X_theta0 = X(:,1);
grad_first_row = 1 / m * product1' * X_theta0;
grad(1,1) = grad_first_row;

X_size = size(X);
%X_column_num = X_size(1,2); %矩阵X的列数
num_theta = X_size(1,2);    %矩阵theta的函数，与 矩阵X的列数 相等
for j=2:num_theta
    item2_1 = product1' * X(:,j);
    item2_2 = theta(j,1);
    grad(j,1) = 1 / m * item2_1 + lambda / m * item2_2; 
end
% =============================================================

end
