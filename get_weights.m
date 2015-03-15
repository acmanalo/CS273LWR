function weights = get_weights(training_inputs, datapoint, c)
    n = size(training_inputs, 1);
    weights = sparse(n, n);
    
    for i = 1:n
        weights(i, i) = gaussian_kernel(training_inputs(i, :), datapoint, c);
    end
end