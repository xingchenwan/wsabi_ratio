M = csvread('posterior_samples_6Feb.csv');
%bounds = [0, 0.5; 0, 0.015; 0, 8; 0, 4];
%bounds = bounds';
cornerplot(M)%, bounds);