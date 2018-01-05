function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); %theta的形状为 3x1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%   计算 J
z = X * theta;
sigmod_z = sigmf(z,[1,0]);
item1 = y' * log(sigmod_z);
trans_y = 1-y;
item2 = trans_y' * log(1-sigmod_z);
item = item1 + item2;
sum_item = sum(item);
J = - 1 / m * sum_item;

X_theta0 = X(:,1);
diifference = sigmod_z - y;
item_theta0 = diifference' * X_theta0;
theta0 = 1 / m * item_theta0;
%   计算 grad
X_theta1 = X(:,2);
item_theta1 = diifference' * X_theta1;
theta1 = 1 / m * item_theta1;

X_theta2 = X(:,3);
item_theta2 = diifference' * X_theta2;
theta2 = 1 / m * item_theta2;

theta = [theta0;theta1;theta2];

grad = theta;

% =============================================================

end
