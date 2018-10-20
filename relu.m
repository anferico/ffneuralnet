function [res, deriv] = relu(x)

    res = max(0, x);
    deriv = max(0, sign(x));
end