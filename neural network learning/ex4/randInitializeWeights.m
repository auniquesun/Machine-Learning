function W = randInitializeWeights(L_in, L_out)

% ######## ֮���԰����� 0 ӳ�䵽���� 10������Ϊmatlab�����±�� 1 ��ʼ��######## 
% ######## ��һ�� ����1���ڶ��� ����2���������ھ��� ����9����ʮ������0���������˵�10�࣬������������ֶ�Ӧ��ֵ   ######## 
% ######## ��ȻҲ���԰� ��һ���Ӧ������0���ڶ����Ӧ������1������������10���Ӧ������9 ########


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
