function best_model = neural_network(train_data, train_targets,...
                        hidden_layers_sizes, activation_functions, regularization,...
                        learning_rate, adaptive_learning_rate, max_iterations,...
                        momentum, nesterov, validation_frac, cross_validate,...                        
                        shuffle_dataset, error_evaluation_func, error_func_name,... 
                        xavier, weights_interval_semiwidth, seed, plots_path, filename_prefix)
    
    % ASSUMPTION: Input parameters are provided properly and in the correct format
    
    loss_threshold = 3.5;
    adaptive_learning_rate_threshold = 0.0001;
    counter = 0;
    
    final_train_errors_list = [];
    final_valid_errors_list = [];    

    % Splitting the dataset into training set and validation set
    [partition_data, pieces] = train_validation_split(train_data, validation_frac, cross_validate, shuffle_dataset, 33);     
    [partition_targets, ~] = train_validation_split(train_targets, validation_frac, cross_validate, shuffle_dataset, 33);     

    % At each iteration, build the training and validation set putting
    % together the chunks in partition_data and partition_targets.
    % As a result, for each one of the k-fold of the cross validation,
    % we have a different model (because these models are trained using
    % different training sets each time).
    for p=1:pieces
        
        % This is needed for static train/validation split
        if (not(cross_validate) && p ~= 2)
            continue;
        end
        
        % 1:end ~= p means that you want to skip the p-th row
        tr_data = partition_data{1:end ~= p, 1};       % Training set's features
        tr_targets = partition_targets{1:end ~= p, 1}; % Training set's targets
        
        va_data = partition_data{p, 1};                % Validation set's features
        va_targets = partition_targets{p, 1};          % Validation set's targets
        
        
        nPatterns_train = size(tr_data, 1); % # of input patterns

        nInput = size(tr_data, 2);          % # of input units
        nOutput = size(tr_targets, 2);      % # of output units
        % The number of hidden units is an hyperparameter

        % Minimum and maximum values for the initial weights
        min_weight = -weights_interval_semiwidth;
        max_weight = weights_interval_semiwidth;

        % Variables holding the different losses for each epoch. 
        % They will be used to plot the results at the end.
        epochs = (1:max_iterations)';            % Simply holds the epoch numbers
        losses_train = zeros(max_iterations, 1); % Holds the values of the training error at each epoch
        losses_valid = zeros(max_iterations, 1); % Holds the values of the validation error at each epoch

        % Initialize the random number generator using the seed passed as
        % parameter. This way, experiments can be made reproducible
        rng(seed, 'v5uniform');

        
        % -----------------------------------------------------------------
        % Initializing the weights
        % -----------------------------------------------------------------

        % Cell object containing matrices representing the weights from
        % each layer to the following one. If the total number of layers
        % (including input and output layers) is N, the number of matrices
        % to store is N-1.
        weights_matrices = cell(2 + length(hidden_layers_sizes) - 1, 1);
        
        % Loop over the sizes of each hidden layer
        nUnits_curr_layer = nInput;
        for j = 1 : length(hidden_layers_sizes)     
            
            nUnits_next_layer = hidden_layers_sizes(j);
            
            % W = Weights matrix holding weights of edges from layer (j-1) to
            % layer j (layer 0 being the input layer). Includes biases' weights
            if xavier % The user requested Xavier's initialization
                max_weight = 1 / (sqrt(nUnits_curr_layer));
                min_weight = -max_weight;
            end
            W = min_weight + (max_weight - min_weight) * rand(nUnits_curr_layer + 1, nUnits_next_layer);              
            W(end, :) = 0; % Initialize biases' weights to 0
            weights_matrices{j, 1} = W;
            
            nUnits_curr_layer = nUnits_next_layer;               
        end
        
        % W = Weights matrix holding weights of edges from the last hidden layer to
        % the output layer. Includes biases' weights
        W = min_weight + (max_weight - min_weight) * rand(nUnits_curr_layer + 1, nOutput);        
        W(end, :) = 0; % Initialize biases' weights to 0
        weights_matrices{end, 1} = W;        
        
        % Fixing the remaining hyperparameters for the algorithm
        eta = learning_rate;   
        alpha = momentum;
        lambda = regularization;
        original_eta = learning_rate;
        
        % Weight changes at last iteration. Needed for momentum
        weights_matrices_change_old = cell(size(weights_matrices));
        for j = 1 : length(weights_matrices_change_old)             
            weights_matrices_change_old{j, 1} = zeros(size(weights_matrices{j, 1}));
        end
        
        
        %------------------------------------------------------------------
        % Beginning of the training process
        %------------------------------------------------------------------
        for i = 1 : max_iterations
                        
            % -------------------------------------------------------------
            % Computing outputs and derivatives for each layer 
            % (after the presentation of TRAINING patterns)
            % -------------------------------------------------------------
            
            layers_outputs_tr = cell(length(weights_matrices), 1);
            layers_derivatives_tr = cell(length(weights_matrices), 1);
            O_prev_layer_tr = tr_data;
            for j = 1 : size(weights_matrices, 1)                            
            
                O_prev_layer_tr_with_biases = [O_prev_layer_tr, ones(size(O_prev_layer_tr, 1), 1)];
                W = weights_matrices{j, 1};
                act_fun = activation_functions{j};
                % O_curr_layer_tr = Outputs of units in the current layer
                % d_curr_layer_tr = Derivative of such units' activation functions                    
                [O_curr_layer_tr, d_curr_layer_tr] = act_fun(O_prev_layer_tr_with_biases * W);
                
                layers_outputs_tr{j, 1} = O_curr_layer_tr;
                layers_derivatives_tr{j, 1} = d_curr_layer_tr;                
                O_prev_layer_tr = O_curr_layer_tr;
            end
            
            
            % -------------------------------------------------------------
            % Evaluating the training error
            % -------------------------------------------------------------
            
            O_Outputs_tr = layers_outputs_tr{end, 1};               
            losses_train(i) = error_evaluation_func(tr_targets, O_Outputs_tr);
            
            
            % -------------------------------------------------------------
            % Computing outputs for each layer 
            % (after the presentation of VALIDATION patterns)
            % -------------------------------------------------------------
            
            O_prev_layer_va = va_data;
            for j = 1 : size(weights_matrices, 1)                            
            
                O_prev_layer_va_with_biases = [O_prev_layer_va, ones(size(O_prev_layer_va, 1), 1)];
                W = weights_matrices{j, 1};
                act_fun = activation_functions{j};
                
                O_curr_layer_va = act_fun(O_prev_layer_va_with_biases * W);                           
                O_prev_layer_va = O_curr_layer_va;
            end
            
            
            % -------------------------------------------------------------
            % Evaluating the validation error
            % -------------------------------------------------------------               
                       
            losses_valid(i) = error_evaluation_func(va_targets, O_prev_layer_va);
            
            
            % -----------------------------------------------------
            % Gradient descent with backpropagation
            % -----------------------------------------------------            
            
            if (nesterov)                                        
                % I have to evaluate the outputs of the network again, but
                % this time I need to apply the momentum first
                
                O_prev_layer_tr = tr_data;
                for j = 1 : length(weights_matrices)           
                    
                    O_prev_layer_tr_with_biases = [O_prev_layer_tr, ones(size(O_prev_layer_tr, 1), 1)];
                    % Computing the interim points on the weights space
                    W = weights_matrices{j, 1};
                    W_change_old = weights_matrices_change_old{j, 1};
                    interim = W + W_change_old;
                    act_fun = activation_functions{j};
                    
                    [O_curr_layer_tr, d_curr_layer_tr] = act_fun(O_prev_layer_tr_with_biases * interim);

                    layers_outputs_tr{j, 1} = O_curr_layer_tr;
                    layers_derivatives_tr{j, 1} = d_curr_layer_tr;                
                    O_prev_layer_tr = O_curr_layer_tr; 
                end
            end
            
            
            % -------------------------------------------------------------
            % Computing deltas for each layer
            % -------------------------------------------------------------            
            
            layers_deltas = cell(length(weights_matrices), 1);
            
            % Loop backwards over the outputs of each layer
            for j = length(layers_outputs_tr) : -1 : 1
                
                O_curr_layer_tr = layers_outputs_tr{j, 1};
                d_curr_layer_tr = layers_derivatives_tr{j, 1};
                                
                if (j == length(layers_outputs_tr))
                    % I'm at the output layer, so the error contribution of
                    % each unit is simply the difference between the
                    % desired output and the predicted output
                    err_contrib = tr_targets - O_curr_layer_tr;
                else
                    % I'm at a hidden layer, so the error contribution of
                    % each unit is computed by backpropagating the deltas
                    % from the layer following this one
                    W = weights_matrices{j+1, 1};
                    err_contrib = layers_deltas{j+1, 1} * W(1:end-1, :)';
                end
                
                delta = err_contrib .* d_curr_layer_tr;
                layers_deltas{j, 1} = delta;
            end
            
            
            % -------------------------------------------------------------
            % Updating the weights
            % -------------------------------------------------------------
            
            for j = 1 : length(weights_matrices)                                
            
                W_change_old = weights_matrices_change_old{j, 1};
                W = weights_matrices{j, 1};
                
                momentum_term = alpha * W_change_old;  
                regulariz_term = lambda * W;    
                regulariz_term(end, :) = 0;  % Don't regularize biases' weights
                
                if (j == 1)
                    O_curr_layer_tr = tr_data;
                else
                    O_curr_layer_tr = layers_outputs_tr{j-1, 1};
                end
                O_curr_layer_tr_with_biases = [O_curr_layer_tr, ones(size(O_curr_layer_tr, 1), 1)];
                delta_curr_layer = layers_deltas{j, 1};
                
                W_change_new = (O_curr_layer_tr_with_biases' * delta_curr_layer * eta) / nPatterns_train;
                weights_matrices{j, 1} = W + W_change_new + momentum_term - regulariz_term;    
                weights_matrices_change_old{j, 1} = W_change_new;            
            end
            
            % Implementation of an adaptive learning rate: if the loss 
            % is not decreasing by a certain threshold between two 
            % consecutive epochs, reduce the learning rate by 80%.
            if (adaptive_learning_rate && i > 1)
                gain = abs(losses_train(i) - losses_train(i-1));
                if (gain < adaptive_learning_rate_threshold)
                    eta = eta / 5;
                end
            end

        end


        % ---------------------------------------------------------
        % Plotting out the results
        % ---------------------------------------------------------
        
        loss_train = losses_train(end);
        loss_valid = losses_valid(end);
        
        if (loss_train <= loss_threshold)

            counter = counter + 1;

            f = figure('visible', 'off');
            plot(epochs, losses_train, 'r--');
            hold on;
            plot(epochs, losses_valid, 'b');
            
            ylabel(error_func_name);
            xlabel('Epochs');
            
            legend('Training', 'Validation');
            
            nHiddenStr = sprintf('Hidden layers sizes: %s', mat2str(hidden_layers_sizes));
            etaStr = sprintf('Learning rate: %0.4f', original_eta);
            alphaStr = sprintf('Momentum: %0.4f', alpha);
            lambdaStr = sprintf('L2 Regularization: %0.6f', lambda);
            
            strTr = sprintf('Final %s (Training): %0.5f', error_func_name, loss_train);
            strVa = sprintf('Final %s (Validation): %0.5f', error_func_name, loss_valid);
            
            myStr = [nHiddenStr newline etaStr newline alphaStr newline lambdaStr newline strTr newline strVa];
           
            % Top right
            annotation('textbox', [.46 .38 .4 .4], 'String', myStr, 'FitBoxToText', 'on');
            % Bottom right
            %annotation('textbox', [.40 .01 .4 .4], 'String', myStr, 'FitBoxToText', 'on');
            % Bottom left
            %annotation('textbox', [.17 .01 .4 .4], 'String', myStr, 'FitBoxToText', 'on');
            
            filename = sprintf('%s%d[%d]', plots_path, filename_prefix, counter);
            saveas(f, filename, 'png');                                       
        end
        
        % Here is where a single configuration of train/validation
        % (obtained by assembling the pieces returned by the cross
        % validation) ends.
        final_train_errors_list = [final_train_errors_list, loss_train];
        final_valid_errors_list = [final_valid_errors_list, loss_valid];
    end
%     fprintf('CROSS VALIDATION RESULTS:\n');
%     fprintf('\tAverage training error: %0.5f\n', mean(final_train_errors_list));
%     fprintf('\tAverage validation error: %0.5f\n', mean(final_valid_errors_list));
    
    best_model = Model(hidden_layers_sizes, learning_rate, momentum, regularization,... 
                        weights_matrices, activation_functions,... 
                        mean(final_train_errors_list), mean(final_valid_errors_list));
end
