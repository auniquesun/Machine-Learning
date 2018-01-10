function g = sigmoidGradient(z)

% ######## ֮���԰����� 0 ӳ�䵽���� 10������Ϊmatlab�����±�� 1 ��ʼ��######## 
% ######## ��һ�� ����1���ڶ��� ����2���������ھ��� ����9����ʮ������0���������˵�10�࣬������������ֶ�Ӧ��ֵ   ######## 
% ######## ��ȻҲ���԰� ��һ���Ӧ������0���ڶ����Ӧ������1������������10���Ӧ������9 ########

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
