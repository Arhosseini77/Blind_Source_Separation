clc;clear;close all;

%% Section 1 :
load('hw6-part1.mat');
% Access the variables:
% D: Dictionary matrix
% x: Noiseless observation

%% BP
N = size(D, 2);
f = ones(2 * N, 1);
Aeq = [D, -D];
beq = x;
lb = zeros(2 * N, 1);
options = optimoptions('linprog', 'Display', 'none');
yhat = linprog(f, [], [], Aeq, beq, lb, [], [], options);
splus = yhat(1:N);
sminus = yhat(N+1:end);
sBP = splus - sminus;
%% MP
N0 = 2; % Desired sparsity level
N = size(D, 2);
residual = x; % Set the initial residual as the observation x
sMP = zeros(N, 1); % Initialize the sparse vector sMP
posMP = zeros(1, N0); % Initialize the positions of selected atoms

for i = 1:N0
    correlation = D' * residual; % Calculate the correlations between dictionary atoms and residual
    [~, pos] = max(abs(correlation)); % Find the atom with the maximum correlation
    posMP(i) = pos; % Store the selected atom position
    atom = D(:, pos); % Retrieve the selected atom
    projection = atom' * residual; % Project the residual onto the selected atom
    sMP(pos) = sMP(pos) + projection; % Update the sparse vector
    residual = residual - projection * atom; % Update the residual
end

%% BP X_noisy :
% Set the parameters
N = size(D, 2);  % Number of columns in the dictionary
M = size(D, 1);  % Number of rows in the dictionary
eps = 0.01;  % Threshold for non-zero elements in 𝒔BP_noisy

% Solve the Basis Pursuit problem using linear programming
f = ones(2*N, 1);
Aeq = [D, -D];
beq = x_noisy;
lb = zeros(2*N, 1);
yhat = linprog(f, [], [], Aeq, beq, lb, []);

% Extract the positive and negative parts of the solution
splus = yhat(1:N);
sminus = yhat(N+1:end);

% Calculate 𝒔BP_noisy
sBP_noisy = splus - sminus;

% Find the positions with non-zero elements in 𝒔BP_noisy
posBP_noisy = find(abs(sBP_noisy) > eps)';

% Display 𝒔BP_noisy
disp('SBP_noisy:')
[posBP_noisy; sBP_noisy(posBP_noisy)']

% Calculate the loss (error) between 𝒔BP_noisy and the true 𝒔
loss_BP_noisy = norm(sBP_noisy - sBP);

% Display the loss
fprintf('Loss (BP_noisy): %.2f\n', loss_BP_noisy);

%% Lasso with different Landa
% Set the range of lambda values
lambda_values = [0.001, 0.01, 0.1, 1, 10, 100];

% Initialize the solution matrix
s_lasso = zeros(N, numel(lambda_values));

% Solve the LASSO problem for each lambda value
for i = 1:numel(lambda_values)
    lambda = lambda_values(i);
    [s_lasso(:, i), ~] = lasso(D, x_noisy, 'Lambda', lambda, 'Standardize', false);
end

% Compare the estimated sparse vectors with the true sparse vector
for i = 1:numel(lambda_values)
    pos_lasso = find(abs(s_lasso(:, i)) > 0);
    fprintf('Lambda = %.3f:\n', lambda_values(i));
    fprintf('Non-zero elements of S:\n');
    fprintf('%d ', pos_lasso);
    fprintf('\n');
    fprintf('Values of non-zero elements of S:\n');
    fprintf('%.2f ', s_lasso(pos_lasso, i));
    fprintf('\n');
    fprintf('Loss (LASSO): %.2f\n', norm(D*s_lasso(:, i) - x_noisy));
    fprintf('------------------------\n');
end

