function W = randInitializeWeights(L_in, L_out)

% ######## 之所以把数字 0 映射到数字 10，是因为matlab数组下标从 1 开始，######## 
% ######## 第一类 数字1，第二类 数字2，。。。第九类 数字9，第十类数字0；这样除了第10类，其他类就是数字对应的值   ######## 
% ######## 当然也可以把 第一类对应成数字0，第二类对应成数字1，。。。，第10类对应成数字9 ########


%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%









% =========================================================================

end
