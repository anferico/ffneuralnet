function [partition, n_pieces] = train_validation_split(dataset, validation_fraction, do_cross_validation, shuffle, seed)
    n_patterns = length(dataset);
    
    if shuffle
        % Saving the current state of the random numbers generator
        s = rng;
        % Shuffling the dataset
        rng(seed);
        dataset = dataset(randperm(size(dataset, 1)), :);
        % Restoring the old state of the random numbers generator
        rng(s);
    end
    
    if not(do_cross_validation)        
        n_patterns_valid = floor(n_patterns * validation_fraction);
        n_patterns_train = n_patterns - n_patterns_valid;
        
        tr = dataset(1 : n_patterns_train, :);       % Training set
        va = dataset(n_patterns_train+1 : end, :);   % Validation set
        partition = {tr; va};
        n_pieces = 2;
    else
        chunks = ceil(1 / validation_fraction);
        partition = cell(chunks, 1);
        n_patterns_per_chunk = floor(n_patterns * validation_fraction);
        last_index = 1;
        for i = 1:chunks-1
            partition{i, 1} = dataset(last_index : last_index + n_patterns_per_chunk - 1, :);
            last_index = last_index + n_patterns_per_chunk;
        end
        partition{chunks, 1} = dataset(last_index : end, :);
        n_pieces = chunks;
    end

end