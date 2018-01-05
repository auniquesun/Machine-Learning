function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

sum_h_theta = X * theta;
item_loss = sum_h_theta - y;    %注意，这里是小写y
sum_item_loss = item_loss.^2;
sum_loss = sum(sum_item_loss);  %列求和
J = sum_loss / m / 2;




% =========================================================================

end
