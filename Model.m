classdef Model
   properties
      Hidden_layer_sizes
      Learning_rate
      Momentum
      Regularization
      Weights
      Activation_functions
      Final_train_error
      Final_valid_error
   end
   methods
      % Constructor method
      function obj = Model(hiddens_sizes, eta, alpha, lambda, w, act_functions, train_err, valid_err)
          if  nargin > 0
              obj.Hidden_layer_sizes = hiddens_sizes;
              obj.Learning_rate = eta;
              obj.Momentum = alpha;
              obj.Regularization = lambda;
              obj.Weights = w;
              obj.Activation_functions = act_functions;
              obj.Final_train_error = train_err;
              obj.Final_valid_error = valid_err;
          end
      end
       
      % Use the model to predict the outputs, given the inputs
      function output = Predict(this, data)
          O_prev_layer = data;
          for i = 1 : length(this.Weights)
              
              W = this.Weights{i, 1};
              act_fun = this.Activation_functions{i};
              
              O_prev_layer_with_biases = [O_prev_layer, ones(size(O_prev_layer, 1), 1)];
              [O_curr_layer, ~] = act_fun(O_prev_layer_with_biases * W);              
              O_prev_layer = O_curr_layer; 
              
          end
          output = O_curr_layer;
      end
   end
end
