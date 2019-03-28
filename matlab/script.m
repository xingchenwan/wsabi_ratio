M = csvread('mc_sotonmet_samples.csv');
M = log(M);
%bounds = [0, 5; 0, 200; 0,10];
%bounds = bounds';
cornerplot(M)%, bounds);