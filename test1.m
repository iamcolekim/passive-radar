%fid_x = fopen('C:\Users\david\ece496\passive-radar\9f.cs8',  'r');
fid_y = fopen('C:\Users\david\ece496\passive-radar\63.cs8',  'r');
len = 2^15;

x = fread(fid_y, 2*len, 'int8');
x_complex = double(x(1:2:end)) + 1i*double(x(2:2:end));
x_complex = x_complex - mean(x_complex);
ts_x = timeseries(x_complex, (0:len-1)');

y = fread(fid_y, 2*len, 'int8');
y_complex = double(y(1:2:end)) + 1i*double(y(2:2:end));
y_complex = y_complex - mean(y_complex);
ts_y = timeseries(y_complex, (0:len-1)');

plot(ts_x.Time, ts_x.Data, 'b', 'LineWidth', 1.5);
hold on;
plot(ts_y.Time, ts_y.Data, 'r', 'LineWidth', 1.5)
legend('x = 63', 'y = 63');
grid on;
%fclose(fid_x);
fclose(fid_y);