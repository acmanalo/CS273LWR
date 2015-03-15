function error = mse(actual, predicted)
%MSE Summary of this function goes here
%   Detailed explanation goes here

error = sum((actual - predicted).^2) / size(actual, 1);
end

