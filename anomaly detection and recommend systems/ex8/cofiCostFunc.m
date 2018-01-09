function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params

   %num_movies x num_features的矩阵，元素值从1到num_movies*num_features
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));    % X_grad的大小 num_movies x num_features，初值全为0
Theta_grad = zeros(size(Theta));    % Theta_grad的大小 num_users x num_features，初值全为0

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% ######## compute J ########
sum_loss = 0;
first_block_loss = 0; 
% first block of J
hypo = X * Theta';
partial_difference_hypo_and_actual = (hypo - Y).^2;
all_difference_hypo_and_actual = sum(sum(partial_difference_hypo_and_actual.*R));
first_block_loss = 1 / 2 * all_difference_hypo_and_actual;

% for i=1:num_movies
%     for j=1:num_users
%         if(R(i,j) == 1)
%             difference_hypo_and_actual = Theta(j,:) * X(i,:)' - Y(i,j);
%             item_loss = sum(difference_hypo_and_actual.^2);
%             first_block_loss = first_block_loss + item_loss;
%         end
%     end
% end
% first_block_loss = 1 / 2 * first_block_loss;

% second block of J
second_block_loss = 0;
X_back = X;
X_back_itemSquare = X_back.^2;
second_block_loss = lambda / 2 * sum(sum(X_back_itemSquare));

% third block of J
third_block_loss = 0;
Theta_back = Theta;
Theta_back_itemSquare = Theta_back.^2;
third_block_loss = lambda / 2 * sum(sum(Theta_back_itemSquare));

sum_loss = first_block_loss + second_block_loss + third_block_loss;
J = sum_loss;

% ######## compute grad ########
for i=1:num_movies
    for k=1:num_features
        for j=1:num_users
            if(R(i,j) == 1)
                difference_hypo_and_actual = Theta(j,:) * X(i,:)' - Y(i,j);
                product = difference_hypo_and_actual * Theta(j,k);
                X_grad(i,k) = X_grad(i,k) + product;
            end
        end
        X_grad(i,k) = X_grad(i,k) + lambda * X(i,k);
    end
end

for j=1:num_users
    for k=1:num_features
        for i=1:num_movies
            if(R(i,j) == 1)
                difference_hypo_and_actual = Theta(j,:) * X(i,:)' - Y(i,j);
                product = difference_hypo_and_actual * X(i,k);
                Theta_grad(j,k) = Theta_grad(j,k) + product;    %千万注意，这里是用 Theta_grad 计算，而不是用 Theta，之前就犯了这样的错误
            end
        end
        Theta_grad(j,k) = Theta_grad(j,k) + lambda * Theta(j,k);    %千万注意，这里是用 Theta_grad 计算，而不是用 Theta，之前就犯了这样的错误
    end     
end    
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
