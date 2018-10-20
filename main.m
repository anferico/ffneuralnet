
% Get the dataset

% ML Cup dataset
tr = dlmread('Datasets/ML_Cup/ML-CUP17-TR.csv', ',');
dataset_data = tr(:, 2:end-2);
dataset_targets = tr(:, end-1:end);

% Z-score standardization
scaled_dataset_data = zeros(size(dataset_data));
for i = 1 : size(dataset_data, 2)
    scaled_dataset_data(:, i) = (dataset_data(:, i) - mean(dataset_data(:, i))) ./ std(dataset_data(:, i));
end
% scaled_dataset_targets = zeros(size(dataset_targets));
% for i = 1 : size(dataset_targets, 2)
%     scaled_dataset_targets(:, i) = (dataset_targets(:, i) - mean(dataset_targets(:, i))) ./ std(dataset_targets(:, i));
% end


% Monk dataset (we'll just append the test set to the training set)
% tr = dlmread('Datasets/Monk/monk_train_1_encoded.csv', ' ');
% ts_monk = dlmread('Datasets/Monk/monk_test_1_encoded.csv', ' ');
% tr = [tr; ts_monk];
% dataset_data = tr(:, 2:end);
% dataset_targets = tr(:, 1);

% Path where plots will be saved
plots_path = 'Plots/ML_Cup/';

% Here, the name of the function that operates the split
% may be misleading. Despite the name, the function 
% train_validation_split simply performs a split 
% among the data. Indeed, in this case we are performing
% a train/test split rather than a train/validation split.
test_fraction = 0.2;
[partition_data, ~] = train_validation_split(scaled_dataset_data, test_fraction, false, true, 64);
[partition_targets, ~] = train_validation_split(dataset_targets, test_fraction, false, true, 64);

train_data = partition_data{1,1};
train_targets = partition_targets{1,1};
test_data = partition_data{2,1};
test_targets = partition_targets{2,1};

temp_model = Model;
best_model = Model;
best_model.Final_valid_error = realmax;
min_error = realmax;


% -------------------------------------------------------------------------
% Grid search
% -------------------------------------------------------------------------

% Number of hidden units (first layer)
%hiddens_L1 = [5, 7, 10];
hiddens_L1 = [8];     

% Number of hidden units (second layer)
%hiddens_L2 = [5, 7, 10];
hiddens_L2 = [10]; 

% Learning rate eta
etas = [0.001];
%etas = [0.0005, 0.001];      

% Momentum parameter alpha
alphas = [0.9];
%alphas = [0.8, 0.9];

% L2 Regularization parameter lambda
lambdas = [0.000001];
%lambdas = [0.00001, 0.000001];;

counter = -1;

for i = 1 : length(hiddens_L1)    
    
    nHidden_L1 = hiddens_L1(i);
    
    for j = 1 : length(hiddens_L2)
       
        nHidden_L2 = hiddens_L2(j);
        
        for k = 1 : length(etas)
            
            eta = etas(k);
            
            for u = 1 : length(alphas)
                
                alpha = alphas(u);
                
                for v = 1 : length(lambdas)
                    
                    lambda = lambdas(v);
                    counter = counter + 1;
                    
                    temp_model = neural_network(train_data,... % Training features
                                   train_targets,...         % Training targets
                                   [nHidden_L1, nHidden_L2],...          % Hidden layers sizes
                                   {@hyperbolic_tangent, @hyperbolic_tangent, @identity},... % Activation functions
                                   lambda,...                % L2 regularization parameter
                                   eta,...                   % Learning rate (initial value)
                                   false,...                 % Whether or not to use an adaptive learning rate
                                   3000,...                  % Max iterations
                                   alpha,...                 % Momentum parameter
                                   false,...                 % Whether or not to use Nesterov's Momentum
                                   0.2,...                   % Validation fraction
                                   true,...                 % Perform a cross validation
                                   true,...                  % Shuffle the dataset before splitting
                                   @mean_euclidean_error,... % Error evaluation function
                                   'MEE',...                 % Name of error evaluation function 
                                   true,...                  % Use Xavier initialization for the weights
                                   0.3,...                   % Radius of interval of initial weights (meaningful iff xavier=false)
                                   0,...                     % Seed to initialize weights, using rand()
                                   plots_path,...
                                   counter);
                    
                    if (temp_model.Final_valid_error < min_error)
                        best_model = temp_model;
                        min_error = temp_model.Final_valid_error;
                    end
                end
                
            end
            
        end
        
    end %
    
end

best_model







