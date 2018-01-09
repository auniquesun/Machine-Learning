function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);    %X的大小

% You need to return the following variables correctly.
centroids = zeros(K, n);    %centroids的大小


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
num_example_Cluster_I = zeros(K,1);
for i=1:m
    for j=1:K
        if idx(i,1) == j
            num_example_Cluster_I(j,1) = num_example_Cluster_I(j,1) + 1;
            centroids(j,:) = centroids(j,:) + X(i,:);
        end
    end
%     if idx(i,1) == 1
%         num_example_Cluster_I(1,1) = num_example_Cluster_I(1,1) + 1;
%         centroids(1,:) = centroids(1,:) + X(i,:);
%     elseif idx(i,1) == 2
%         num_example_Cluster_I(2,1) = num_example_Cluster_I(2,1) + 1;
%         centroids(2,:) = centroids(2,:) + X(i,:);
%     else
%         num_example_Cluster_I(3,1) = num_example_Cluster_I(3,1) + 1;
%         centroids(3,:) = centroids(3,:) + X(i,:);
%     end
end

for i=1:K
    centroids(i,:) = centroids(i,:) / num_example_Cluster_I(i,1);
end



% =============================================================


end

