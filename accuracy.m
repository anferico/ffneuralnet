function acc = accuracy(actual, predicted)
    absDiff = sum(abs(actual - round(predicted)), 2);
    acc = sum(absDiff(:)==0) / size(absDiff, 1);
end