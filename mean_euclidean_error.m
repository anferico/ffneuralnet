function result = mean_euclidean_error(targets, outputs)

    % Output error at the output units (1 row per pattern, 1 column per output unit)
    errors_at_output_layer = targets - outputs;
    result = mean(sqrt(sum(errors_at_output_layer.^2, 2)));

end