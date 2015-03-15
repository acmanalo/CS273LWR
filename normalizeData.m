function normalizedData = normalizeData( X )
%NORMALIZEDATA Summary of this function goes here
%   Detailed explanation goes here

normalizedData = zeros(size(X));
minX = min(X);
maxX = max(X);
for i = 1:size(X, 1)
    normalizedData(i, :) = (X(i, :) - minX) / (maxX - minX);
end

end

