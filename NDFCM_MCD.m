function [ centroids, clusterAssment, original_label, cluster_label,intersection, X] = NDFCM_MCD( k )
%NDFCM Summary of this function goes here
%   Detailed explanation goes here
X = dataSet();
[shape1, shape2] = size(X);
numSamples = shape1;
%��һ�д������Ӧ��������һ�࣬�ڶ��д�Ÿ�����������ĵ����
clusterAssment = zeros(numSamples, 2);
clusterChanged = true;

%��һ������ʼ��������
[centroids] = initCentroids(X, k);
%disp(centroids);
while clusterChanged
    clusterChanged = false;
    %����ÿһ������
    for i = 1:numSamples
        minDist = 1000000000000000.0;
        minIndex = 0;
        %�ڶ�����
        %�ҵ���ÿһ�����������������
        for j = 1:k
            distance = euclDistance(centroids(j,:), X(i, :));
            if distance < minDist
                minDist = distance;
                minIndex = j;
            end
        end
        %��������
        %���´�
        if clusterAssment(i,1)~= minIndex
            clusterChanged = true;
            clusterAssment(i, :) = [minIndex, minDist^2];
        end  
        %disp(clusterAssment);
    end
    %���Ĳ������´����ġ�ÿ���ص�����Ԫ�صĸ�ά�ȵ�ƽ��ֵ    
    mean_cluster = cell(k,shape2);
    mean_cluster = init_mean_cluster(mean_cluster, X);
    for n = 1:k
        count = 0;
        for m = 1:numSamples
            if clusterAssment(m,1) == n
                %��Ϊʹ��ƽ��ֵ�������ĵ����һ��ά���Ǳ�����ģ����������׼ȷ
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


%%��ȷ�ʼ���
%a = find(X(:,9) == 1)�õ���һ�������±�

% ch_label = cell(1,k);
% cluster_label = cell(1,k);
%ch_label�洢ԭʼ������1-12��ģ�ÿһ����±꼯�ϡ�
%cluster_label�洢����֮��������1-12��ģ�ÿһ����±꼯�ϡ�
% for i = 1:k
%     ch_label{i} = find(X(:,5) == i);
%     cluster_label{i} = find(clusterAssment(:,1) == i);
% end
%�������֮��ÿһ���ԭʼ��������ƶ���ߵ���һ��
%max_similarity ��ź�ԭʼ����н���Ԫ�صĸ��������ֵ
%accurates ���ÿһ������׼ȷ��
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

%�������֮��ÿһ������и���ͨ���ĺ���
%ÿ��������������
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

function [X] = dataSet()%��������
%ʹ�þ�������ֵ��ȡ����֮�������Դ
load('E:\matlab\data\Multi_Point.mat')
[m, n] = size(Multi_Point);
%��ʱֻʹ��ARϵ��
X = Multi_Point;
end
%MU is an N-by-D matrix, and sigma is a D-by-D matrix
%mvnrnd returns an N-by-D matrix.

function [euclDist] = euclDistance(vector1, vector2) %����ŷʽ����
%vecDistX = sqrt(sum((vector1{1,1} - vector2{1,1}).^2));
%vecDistY = sqrt(sum((vector1{1,2} - vector2{1,2}).^2));
%euclDist = sqrt(vecDistX.^2 + vecDistY.^2);
euclDist = sum(abs(vector1{1,1} - vector2{1,1})) + sum(abs(vector1{1,2} - vector2{1,2}));
end

function [centroids] = initCentroids(X, k)%�������k����ʼ������
[numSamples, dim] = size(X);
centroids = cell(k, dim);
for i = 1:k
    index = round(numSamples*rand());
    centroids(i,:) = X(index, :);
end
end

