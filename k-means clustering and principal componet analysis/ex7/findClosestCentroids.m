function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); % 矩阵centroids形状的第一个参数，即矩阵centroids有多少行，即聚类中心的个数

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

m = size(idx,1);    %样本个数
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
distance_example_I_centroid_J = zeros(3,1);
for i=1:m
    for j=1:K
        example_I = X(i,:);
        example_I_trans = example_I';
        centroid_J = centroids(j,:);
        centroid_J_trans = centroid_J';
        difference = example_I_trans - centroid_J_trans;
        distance_example_I_centroid_J(j,1) = sum(sum(difference.^2));   %求出example_I距各中心点距离的平方和
    end
    min = distance_example_I_centroid_J(1,1);
    k = 1;  %记录 example_I 距离哪个中心最近
    for j=2:K
        if min > distance_example_I_centroid_J(j,1);
            k = j;
            min = distance_example_I_centroid_J(j,1);
        end
    end
    idx(i,1) = k;
end






% =============================================================

end

