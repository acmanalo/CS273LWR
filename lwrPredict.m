function prediction = lwrPredict(training_inputs, training_outputs, datapoint, c)
    weights = get_weights(training_inputs, datapoint, c);
    
    x = training_inputs;
    y = training_outputs;
    
    ep = .01;
    xt = x' * (weights * x) + ep * eye(size(x, 2));
    theta = xt^-1 * (x' * (weights * y));
    
    prediction = datapoint * theta;
end