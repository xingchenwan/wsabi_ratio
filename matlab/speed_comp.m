sample_number = [15, 20, 30, 50, 75, 100, 120, 150, 200];
quad_time_wsabi = [0.44, 0.54, 0.69, 1.01, 1.50, 2.34, 3.12, 4.79, 6.35];
quad_time_bq = [0.63, 0.78, 1.44, 1.64, 2.60, 4.03, 5.62, 8.55, 12.5];
hold on
plot(sample_number, quad_time_wsabi, 'b-o')
plot(sample_number, quad_time_bq, 'r-o')
xlabel('Number of samples')
ylabel('Time taken to obtain samples (s)')
legend('Uncertainty Sampling', 'Active Sampling')
set(gca,'FontSize',20)