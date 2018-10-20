function result = mean_squared_error(targets, outputs)

    % Output error at the output units (1 row per pattern, 1 column per output unit)
    errors_at_output_layer = targets - outputs;
    result = mean(0.5 * sum(errors_at_output_layer.^2, 2));
    
end