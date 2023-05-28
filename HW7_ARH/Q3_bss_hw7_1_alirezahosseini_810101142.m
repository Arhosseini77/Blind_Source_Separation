clc;clear;close all;

%% Q3
% Load the data from hw7.mat
data = load('hw7.mat');
X = data.X;

K = 5;
L = 100;
T = size(X, 2);


%% Initialization
A = rand(2);
for i = 1:2
    A(:, i) = A(:, i) / norm(A(:, i));
end

last_e = 100;
last_2e = 101;
i = 1;

while i < 100
    % A is fixed
    S_hat = A \ X;
    
    % for Channel 1
    x1 = X(1, :);
    taw1 = [200 300 800 1200 1400];
    taw1_last = [210 300 800 1200 1400];
    alpha1 = [1 1 1 1 2];
    j = 1;
    while j < 100
        % W is fixed
        y1 = zeros(L, K);
        for k = 1:K
            y1(:, k) = x1(taw1(k) : taw1(k) + L - 1);
        end
        
        s1_hat = y1 * pinv(alpha1);
        s1_hat = s1_hat / norm(s1_hat);
        
        % S is fixed
        Z1 = zeros(L, length(x1) - L + 1);
        for k = 1:length(x1) - L + 1
            Z1(:, k) = x1(k : k + L - 1);
        end
        
        Z1_reduced = Z1;
        for k = 1:K
            projections = s1_hat' * Z1_reduced;
            [~, I] = max(abs(projections));
            alpha_pro = projections(I);
            taw1(k) = I;
            alpha1(k) = alpha_pro;
            Z_size = Z1_reduced;
            Z1_reduced(:, max(1, I - L + 1) : min(size(Z_size, 2), I + L - 1)) = 0;
        end
        
        if isequal(taw1, taw1_last)
            break;
        end
        
        taw1_last = taw1;
        j = j + 1;
    end
    
    si1 = zeros(1, length(x1));
    for q = 1:length(taw1)
        si1(taw1(q) + L / 2) = alpha1(q);
    end
    
    S1_hat = s1_hat;
    Si1 = si1;
    
    % for Channel 2
    x2 = X(2, :);
    taw2 = [200 300 800 1200 1400];
    taw2_last = [210 300 800 1200 1400];
    alpha2 = [1 1 1 1 2];
    j = 1;
    while j < 100
        % W is fixed
        y2 = zeros(L, K);
        for k = 1:K
            y2(:, k) = x2(taw2(k) : taw2(k) + L - 1);
        end
        
        s2_hat = y2 * pinv(alpha2);
        s2_hat = s2_hat / norm(s2_hat);
        
        % S is fixed
        Z2 = zeros(L, length(x2) - L + 1);
        for k = 1:length(x2) - L + 1
            Z2(:, k) = x2(k : k + L - 1);
        end
        
        Z2_reduced = Z2;
        for k = 1:K
            projections = s2_hat' * Z2_reduced;
            [~, I] = max(abs(projections));
            alpha_pro = projections(I);
            taw2(k) = I;
            alpha2(k) = alpha_pro;
            Z_size = Z2_reduced;
            Z2_reduced(:, max(1, I - L + 1) : min(size(Z_size, 2), I + L - 1)) = 0;
        end
        
        if isequal(taw2, taw2_last)
            break;
        end
        
        taw2_last = taw2;
        j = j + 1;
    end
    
    si2 = zeros(1, length(x2));
    for q = 1:length(taw2)
        si2(taw2(q) + L / 2) = alpha2(q);
    end
    
    S2_hat = s2_hat;
    Si2 = si2;
    
    S_hat = [conv(Si1, S1_hat, 'same'); conv(Si2, S2_hat, 'same')];
    
    % S is fixed
    A = X * pinv(S_hat);
    for kk = 1:2
        A(:, kk) = A(:, kk) / norm(A(:, kk));
    end
    
    i = i + 1;
    e = norm(X - A * S_hat) / norm(X);
    
    if abs(e - last_e) < 0.0001 || abs(e - last_2e) < 0.0001
        break;
    end
    
    last_2e = last_e;
    last_e = e;
end

%% Reconstructed Signals
X_new = A * S_hat;

%% Plotting Results
subplot(2, 2, 1);
plot(X_new(1, :));
legend('New Data');
title('Channel 1 - Reconstructed Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(2, 2, 2);
plot(X(1, :));
legend('Real Data');
title('Channel 1 - Original Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(2, 2, 3);
plot(X_new(2, :));
legend('New Data');
title('Channel 2 - Reconstructed Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(2, 2, 4);
plot(X(2, :));
legend('Real Data');
title('Channel 2 - Original Signal');
xlabel('Sample');
ylabel('Amplitude');
