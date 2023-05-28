clc;clear;close all;

%% Q4

% Load the data from hw7.mat
data = load('hw7.mat');
x1 = data.x1 ; 

% Define the starting points of the signals in time series
starting_points = [102, 299, 803, 1201, 1402];

% Convert x1 to the frequency domain using Fourier Transform
X1 = fft(x1);

% Perform single-channel sparse blind deconvolution in the frequency domain
K = 5; % Number of sources/components
L = 100; % Length of the segments
T = length(x1); % Total number of samples

taw = starting_points;
taw_last = taw;
alpha = ones(1, K);

i = 1;
while i < 100
    % W is fixed
    Y = zeros(L, K);
    for k = 1:K
        Y(:, k) = X1(taw(k):taw(k) + L - 1);
    end
    
    S_hat = Y * pinv(alpha);
    S_hat = S_hat / norm(S_hat);
    
    % S is fixed
    Z = zeros(L, T - L + 1);
    for j = 1:T - L + 1
        Z(:, j) = X1(j:j + L - 1);
    end
    
    Z_reduced = Z;
    for k = 1:K
        projections = S_hat' * Z_reduced;
        [~, index] = max(abs(projections));
        alpha_pro = projections(index);
        taw(k) = index;
        alpha(k) = alpha_pro;
        Z_size = Z_reduced;
        Z_reduced(:, max(1, index - L + 1):min(size(Z_size, 2), index + L - 1)) = 0;
    end
    
    if isequal(taw, taw_last)
        break;
    end
    
    taw_last = taw;
    i = i + 1;
end

% Reconstruct the signal in the time domain
Si = zeros(1, T);
for q = 1:length(taw)
    Si(taw(q) + L / 2) = alpha(q);
end

S1_hat = S_hat;
Si1 = Si;

% Apply inverse Fourier Transform to obtain the reconstructed signal in the time domain
x1_reconstructed = ifft(S1_hat);

% Plotting the results
subplot(2, 1, 1);
plot(real(x1_reconstructed));
hold on;
plot(x1);
legend('Reconstructed Signal', 'Original Signal');
title('Single-Channel Sparse Blind Deconvolution - x1');
xlabel('Sample');
ylabel('Amplitude');

subplot(2, 1, 2);
plot(abs(S1_hat));
title('Sparse Representation - x1');
xlabel('Frequency Bin');
ylabel('Magnitude');
