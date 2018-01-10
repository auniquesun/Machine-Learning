function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
% ######## 之所以把数字 0 映射到数字 10，是因为matlab数组下标从 1 开始，######## 
% ######## 第一类 数字1，第二类 数字2，。。。第九类 数字9，第十类数字0；这样除了第10类，其他类就是数字对应的值   ######## 
% ######## 当然也可以把 第一类对应成数字0，第二类对应成数字1，。。。，第10类对应成数字9 ########                               
                               
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));    % 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  % 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));  % 25 x 401
Theta2_grad = zeros(size(Theta2));  % 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
  
a1 = X; % 5000 x 400
a1_scale = [ones(m,1) a1];  % 扩展a1, a1的形状变成 5000 x 401

Z2 = Theta1 * a1_scale';  % 25 x 5000
a2 = sigmoid(Z2);   % 25 x 5000
a2_scale = [ones(1,m) ; a2];    % 扩展a2，a2的形状变成 26 x 5000

Z3 = Theta2 * a2_scale; % 10 x 5000
a3 = sigmoid(Z3);   % 10 x 5000 （5000个样本，对每个样本从0到9的预测值）
hypo = a3;  % 10 x 5000

y_scale = zeros(m,num_labels);    % 5000 x 10；有多少个样本,神经网络有几个输出值
for row=1:m
    y_scale(row,y(row,1)) = 1;    % y是样本的标签，它是列向量（5000 x 1），数字0已结被映射到数字10，
                                        % y(row,1)表示y向量第row行的元素值，元素值从1到10取值
end
y_scale_trans = y_scale';   % 10 x 5000

log_of_hypo = log(hypo); % 10 x 5000
log_of_one_minus_hypo = log(1 - hypo); % 10 x 5000
cost_matrix = (y_scale_trans .* log_of_hypo) + ((1 - y_scale_trans) .* log_of_one_minus_hypo);
first_item = sum(sum(cost_matrix));
% J = -1 / m / first_item;    % 这么低级的错误，应该是 J = -1 / m * first_item;
                            % 要看仔细了，引以为鉴

Theta1_col_num = size(Theta1,2);
Theta1_without_bias = Theta1(:,2:Theta1_col_num);

Theta2_col_num = size(Theta2,2);
Theta2_without_bias = Theta2(:,2:Theta2_col_num);

square_of_Theta1_without_bias = Theta1_without_bias .^ 2;
square_of_Theta2_without_bias = Theta2_without_bias .^ 2;

regularized_item = sum(sum(square_of_Theta1_without_bias)) + sum(sum(square_of_Theta2_without_bias));
J = -1 / m * first_item + lambda / 2 / m * regularized_item;

tmp_Theta_I = Theta1; % 25 x 401
tmp_Theta_I(:,1) = 0;   % 把第一列置为0，统一正则化项
tmp_Theta_II = Theta2; % 10 x 26
tmp_Theta_II(:,1) = 0;  % 把第一列置为0，统一正则化项

regularized_Item_I = zeros(size(Theta1));
regularized_Item_II = zeros(size(Theta2));

triangle_Delta_I = zeros(size(Theta1)); % 25 x 401
% 验证维度是正确的 fprintf('dimension of triangle_Delta_I : %f %f\n',size(triangle_Delta_I));
triangle_Delta_II = zeros(size(Theta2));  % 10 x 26
% 验证维度是正确的 fprintf('dimension of triangle_Delta_II : %f %f\n',size(triangle_Delta_II));
for i=1:m
    a_I = X(i,:)'; % a_I的形状 400 x 1
    a_I_scale = [1 ; a_I]; % a_I_scale的形状 401 x 1
    
    z_II = Theta1 * a_I_scale; % 25 x 1
    a_II = sigmoid(z_II); % 25 x 1
    a_II_scale = [1 ; a_II];  % 26 x 1
    
    z_III = Theta2 * a_II_scale; % 10 x 1
    a_III = sigmoid(z_III); % 10 x 1
    hypo_I = a_III; % 10 x 1
    
    y_I = y_scale(i,:)'; % y_scale的形状5000 x 10，取其第i行（第i个样本），
                         % 再求转置，形状为10 x 1 
    delta_III = hypo_I - y_I;   % 10 x 1
    delta_II = (Theta2_without_bias' * delta_III) .* sigmoidGradient(z_II); % (25 x 10, 10 x 1), 25 x 1
    
    triangle_Delta_I = triangle_Delta_I + (delta_II * a_I_scale');  % 25 x401,(25 x 1,1 x 401)
    
    triangle_Delta_II = triangle_Delta_II + (delta_III * a_II_scale');  % 10 x 26,(10 x 1,1 x 26)
    
    regularized_Item_I = regularized_Item_I + tmp_Theta_I;
    regularized_Item_II = regularized_Item_II + tmp_Theta_II;
end

triangle_Delta_I = 1 / m * triangle_Delta_I + lambda / m * tmp_Theta_I;
triangle_Delta_II = 1 / m * triangle_Delta_II + lambda / m * tmp_Theta_II;

Theta1_grad = triangle_Delta_I;
Theta2_grad = triangle_Delta_II;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
