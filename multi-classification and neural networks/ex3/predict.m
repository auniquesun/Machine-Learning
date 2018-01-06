function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X_scale = [ones(m,1) X]; % 扩展X
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = X_scale;% a1的形状 5000 x 401
Z2 = Theta1 * a1'; % Z2的形状 25 x 401

a2_transpose = sigmoid(Z2); % a2_transpose的形状25 x 5000
a2_transpose_scale = [ones(1,m);a2_transpose];  % a2_transpose_scale的形状26 x 5000
Z3 = Theta2 * a2_transpose_scale; % Z3的形状 10 x 5000

[hypo_transpose,hypo_transpose_index] = max(sigmoid(Z3)); % 默认计算出每列的最大值
p = hypo_transpose_index';
% =========================================================================


end
