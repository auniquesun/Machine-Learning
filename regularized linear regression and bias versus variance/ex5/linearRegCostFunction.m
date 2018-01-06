function [J, grad] = linearRegCostFunction(X, y, theta, lambda) %ע�⣺X��һ�е�ֵȫΪ1
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% һ������ɱ�����J
h_theta = X * theta;
difference_h_and_y = h_theta - y;
item1 = sum(difference_h_and_y.^2);

% ����������ʱ��Ҫȥ�� ��0
num_X_column = size(X,2);
theta_except_index_0 = theta(2:num_X_column,:);
item2 = sum(theta_except_index_0.^2);
J = 1 / 2 / m * item1 + lambda / 2 / m * item2;

% ���������ݶ�grad
theta0_1 = difference_h_and_y;
theta0_2 = X(:,1)';
theta0 = 1 / m * theta0_2 * theta0_1;

theta(1,1) = theta0;

% X������
column = size(X,2);
for j=2:column
    thetaJ_1 = X(:,j)' * difference_h_and_y;
    thetaJ_2 = theta(j,1);
    thetaJ = 1 / m * thetaJ_1 + lambda / m * thetaJ_2;
    theta(j,1) = thetaJ;
end    

% grad = grad(:);
grad = theta;
% =========================================================================

end
