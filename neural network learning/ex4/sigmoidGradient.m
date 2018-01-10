function g = sigmoidGradient(z)

% ######## 之所以把数字 0 映射到数字 10，是因为matlab数组下标从 1 开始，######## 
% ######## 第一类 数字1，第二类 数字2，。。。第九类 数字9，第十类数字0；这样除了第10类，其他类就是数字对应的值   ######## 
% ######## 当然也可以把 第一类对应成数字0，第二类对应成数字1，。。。，第10类对应成数字9 ########

%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

% it's so easy, whatever z is a number, vector or matrix 
g = sigmoid(z) .* (1 - sigmoid(z));












% =============================================================




end
