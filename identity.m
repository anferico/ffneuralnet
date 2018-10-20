
function [res, deriv] = identity(x)

    res = x;
    deriv = ones(size(x));   
end