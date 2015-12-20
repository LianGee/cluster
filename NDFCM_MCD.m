function [ centroids, clusterAssment, original_label, cluster_label,intersection, X] = NDFCM_MCD( k )
%NDFCM Summary of this function goes here
%   Detailed explanation goes here
X = dataSet();
[shape1, shape2] = size(X);
numSamples = shape1;
%第一列存放样本应该属于哪一类，第二列存放该样本与簇中心的误差
clusterAssment = zeros(numSamples, 2);
clusterChanged = true;

%第一步：初始化簇中心
[centroids] = initCentroids(X, k);
%disp(centroids);
while clusterChanged
    clusterChanged = false;
    %对于每一个样本
    for i = 1:numSamples
        minDist = 1000000000000000.0;
        minIndex = 0;
        %第二步：
        %找到离每一个簇中心最近的样本
        for j = 1:k
            distance = euclDistance(centroids(j,:), X(i, :));
            if distance < minDist
                minDist = distance;
                minIndex = j;
            end
        end
        %第三步：
        %更新簇
        if clusterAssment(i,1)~= minIndex
            clusterChanged = true;
            clusterAssment(i, :) = [minIndex, minDist^2];
        end  
        %disp(clusterAssment);
    end
    %第四步，更新簇中心。每个簇的所有元素的各维度的平均值    
    mean_cluster = cell(k,shape2);
    mean_cluster = init_mean_cluster(mean_cluster, X);
    for n = 1:k
        count = 0;
        for m = 1:numSamples
            if clusterAssment(m,1) == n
                %因为使用平均值，簇中心的最后一个维度是标记类别的，因此这个类别不准确
                mean_cluster{n,1} = X{m,1} + mean_cluster{n,1};
                mean_cluster{n,2} = X{m,2} + mean_cluster{n,2};
                count = count + 1;
            end            
        end
        mean_cluster{n,1} = mean_cluster{n,1}./count;
        mean_cluster{n,2} = mean_cluster{n,2}./count;
    end     
    centroids = mean_cluster;
end


%%正确率计算
%a = find(X(:,9) == 1)得到第一类所有下标

% ch_label = cell(1,k);
% cluster_label = cell(1,k);
%ch_label存储原始数据中1-12类的，每一类的下标集合。
%cluster_label存储聚类之后数据中1-12类的，每一类的下标集合。
% for i = 1:k
%     ch_label{i} = find(X(:,5) == i);
%     cluster_label{i} = find(clusterAssment(:,1) == i);
% end
%求出聚类之后，每一类和原始类别中相似度最高的哪一类
%max_similarity 存放和原始类别中交集元素的个数的最大值
%accurates 存放每一类聚类的准确率
% max_similarity_index = zeros(1,k);
% accurates = zeros(1,k);
% for i = 1:k
%     max_similarity = 0;
%     for j = 1:k
%         intersection = intersect(ch_label{i}, cluster_label{j});
%         if(max_similarity < length(intersection))
%             max_similarity = length(intersection);
%             max_similarity_index(1,i) = j;
%         end
%     end
%     accurates(1,i) = max_similarity/length(ch_label{i});
% end
% bar(accurates,0.2);

%计算聚类之后，每一个类别中各个通道的含量
%每个类别的样本集合
original_label = cell(1,k);
cluster_label = cell(1,k);
for i = 1:k
    original_label{i} = find([X{:,3}] == i);
    cluster_label{i} = find(clusterAssment(:,1) == i);
end
intersection = zeros(k,k);
for i = 1:k
   for j = 1:k
       intersection(i, j) = length(intersect(cluster_label{i}, original_label{j}));
   end
end
end

function [mean_cluster] = init_mean_cluster(mean_cluster, X)
    [m, n] = size(X{1,1});
    a = zeros(m,n);
    [m, n] = size(mean_cluster);
    for i = 1:m
        for j = 1:n
            mean_cluster{i,j} = a;
        end
    end
end

function [X] = dataSet()%生成数据
%使用经过特征值提取处理之后的数据源
load('E:\matlab\data\Multi_Point.mat')
[m, n] = size(Multi_Point);
%暂时只使用AR系数
X = Multi_Point;
end
%MU is an N-by-D matrix, and sigma is a D-by-D matrix
%mvnrnd returns an N-by-D matrix.

function [euclDist] = euclDistance(vector1, vector2) %计算欧式距离
%vecDistX = sqrt(sum((vector1{1,1} - vector2{1,1}).^2));
%vecDistY = sqrt(sum((vector1{1,2} - vector2{1,2}).^2));
%euclDist = sqrt(vecDistX.^2 + vecDistY.^2);
euclDist = sum(abs(vector1{1,1} - vector2{1,1})) + sum(abs(vector1{1,2} - vector2{1,2}));
end

function [centroids] = initCentroids(X, k)%随机生成k个初始簇中心
[numSamples, dim] = size(X);
centroids = cell(k, dim);
for i = 1:k
    index = round(numSamples*rand());
    centroids(i,:) = X(index, :);
end
end

