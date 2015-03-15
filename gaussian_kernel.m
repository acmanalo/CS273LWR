function d = gaussian_kernel(x, x0, c)
% x = nearby datapoints
% x0 = datapoint
% c = kernel parameter

    diff = x - x0;
    dot_product = diff * diff';
    
    a = 1;
    
    d = a * exp(dot_product / (-2 * c^2));
end