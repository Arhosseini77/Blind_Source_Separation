%% Q2
clc;clear;close all;
% Load the data from hw7.mat
data = load('hw7.mat');
x2 = data.x2;

K = 5;                 % Number of events
L = 100;               % Length of waveform
T = length(x2);        % Length of input signal
%% Plot the input signal
plot(x2);
title('Input Signal (x2)');
xlabel('Time');
ylabel('Amplitude');

%% Define starting points and initialize parameters
t_w = [201, 608 , 951 , 1308 , 1608];         % Starting points of events
t_w_last = t_w + L;                         % Last points of events
alpha = [1, 1, 1, 1, 2];                    % Initial alpha values
e = 10;                                    % Initialization for error
iteration = 1;

while true
    % Update waveform estimation (s_hat) given fixed events
    for k = 1:K
        y(:, k) = x2(t_w(k):t_w(k) + L - 1);
    end
    s_hat = y * pinv(alpha);
    s_hat = s_hat / norm(s_hat);
    
    % Update events (t_w) given fixed waveform estimation
    for j = 1:T - L + 1
        Z(:, j) = x2(j:j + L - 1);
    end
    Z_reduced = Z;
    for k = 1:K
        projections = s_hat' * Z_reduced;
        [~, I] = max(abs(projections));
        alpha_pro = projections(I);
        t_w(k) = I;
        alpha(k) = alpha_pro;
        Z_size = Z_reduced;
        Z_reduced(:, max(1, I - L + 1):min(size(Z_size, 2), I + L - 1)) = 0;
    end
    
    % Calculate the reconstructed signal and error
    si = zeros(1, T);
    for q = 1:length(t_w)
        si(t_w(q) + L/2) = alpha(q);
    end
    error = norm(x2 - conv(si, s_hat, 'same'));
    
    % Check convergence
    if t_w == t_w_last
        break;
    end
    t_w_last = t_w;
    iteration = iteration + 1;
end

%% Plot the reconstructed signal
figure;
plot(x2);
hold on;
plot(conv(si, s_hat, 'same'));
title('Reconstructed Signal');
xlabel('Time');
ylabel('Amplitude');

%% Plot the estimated waveform
figure;
plot(s_hat);
title('Estimated Waveform');
xlabel('Time');
ylabel('Amplitude');


