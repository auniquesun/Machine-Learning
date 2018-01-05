function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    
    sum_h_theta = X * theta;
    item_loss = sum_h_theta - y;    %ע�⣬������Сдy
    %sum_item_loss = sum(item_loss); %�ڼ���theta(1,1)ʱ��Ҫ�õ�sum_item_loss
    X_theta0 = X(:,1);              % X �ĵ�һ��ȫΪ1
    sum_X_theta0 = item_loss' * X_theta0;
    X_theta1 = X(:,2);                    %��ȡ����X�ĵڶ��У�����Ҫע��X�����У�֮ǰ��X������ ones(m,1)
                                          % X �ĵڶ��в��� population ��ʵ������  
    sum_X_theta1 = item_loss' * X_theta1;    
    
    %ע�⣬theta(1,1)��theta(2,1)��****���㷽����ͬ****
    theta(1,1) = theta(1,1) - (alpha / m * sum_X_theta0);
    theta(2,1) = theta(2,1) - (alpha / m * sum_X_theta1);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('iter:%f  J:%f\n',iter,J_history(iter,1));
end

end
