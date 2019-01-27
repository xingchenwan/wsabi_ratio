M = csvread('PosteriorSampling_24Jan_edited.csv');
bounds = [5, 11; 0, 0.1; 1, 8; 0, 25; 0, 5; 0.06, 0.12];
bounds = bounds';
cornerplot(M,bounds);