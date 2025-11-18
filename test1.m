fid_x = fopen('C:\Users\david\ece496\passive-radar\9f_500M_01_11_car_3.cs8',  'r');
fid_y = fopen('C:\Users\david\ece496\passive-radar\63_500M_01_11_car_3.cs8',  'r');
len = 2^15;
Fs = 20e6;
x = fread(fid_x, 2*len, 'int8');
x_complex = double(x(1:2:end)) + 1i*double(x(2:2:end));
x_complex = x_complex - mean(x_complex);
%ts_x = timeseries(x_complex, (0:len-1)');

y = fread(fid_y, 2*len, 'int8');
y_complex = double(y(1:2:end)) + 1i*double(y(2:2:end));
y_complex = y_complex - mean(y_complex);
%ts_y = timeseries(y_complex, (0:len-1)');

[R, lags] = xcorr(x_complex);
R_normalized = abs(R)/max(abs(R));

figure(1)
plot(lags/Fs*1e6, 10*log10(R_normalized));
title('Auto-Correlation of Signal x');
xlabel('Lag (µs)');
ylabel('Normalized Cross-Correlation (dB)');
ylim([-60, 10])
xlim([-100, 100])
hold on


lms = dsp.LMSFilter('Length', 1024, 'StepSize', 0.1, 'Method', 'Normalized LMS');

[x_complex_filt, err, wts] = lms(x_complex,y_complex);


[R_before, lags_before] = xcorr(x_complex, y_complex);
R_normalized_before = abs(R_before)/max(abs(R_before));

[R_after, lags_after] = xcorr(err, y_complex);
R_normalized_after = abs(R_after)/max(abs(R_after));


figure(2)
plot(lags/Fs*1e6, 10*log10(R_normalized_after), 'b', 'LineWidth', 1.5);
title('Cross-Correlation of Signals x and y');
xlabel('Lag (µs)');
ylabel('Normalized Cross-Correlation (dB)');
hold on;
plot(lags/Fs*1e6, 10*log10(R_normalized_before), 'r', 'LineWidth', 1.5);
legend('After', 'Before')
ylim([-60, 10])
xlim([-100, 100])

%figure(3)
%plot(real(err(1:1000)), 'b', 'LineWidth', 1.5);
%title('Error signal')
%hold on;
%figure(4)
%plot(real(y_complex(1:100)), 'r', 'LineWidth', 1.5)
%title('time domain signal y')
%legend('x = 63', 'y = 63');
%grid on;
fclose(fid_x);
fclose(fid_y);