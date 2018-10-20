function [res, deriv] = hyperbolic_tangent(x)

    a = 1.7159;
    b = 2/3;
    res = a .* tanh(b.*x);
    deriv = a*b .* (sech(b.*x) .^ 2);
end