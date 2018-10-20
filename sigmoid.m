function [res, deriv] = sigmoid(x)

    res = 1 ./ (1 + exp(-1 * x));
    deriv = 1 * (res .* (1 - res));
end