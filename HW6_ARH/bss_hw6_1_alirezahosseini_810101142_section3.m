clc;clear;close all;

%% Section 3:

%% A  :
load('hw6-part3.mat');
C = abs(D.' * D);
max_inner_product = max(max(triu(C, 1)));
mutual_coherence = max_inner_product;
disp(['Mutual Coherence Measure: ' num2str(mutual_coherence)]);

%% MOD:
% Load the observation matrix X from the file
load('hw6-part3.mat', 'X');

% Set the parameters
No = 2;  % Sparsity level
MaxIter = 100;  % Maximum number of iterations

% Initialize the dictionary matrix D using random values
D = randn(10, 40);

% Perform MOD algorithm
tic;
for iter = 1:MaxIter
    % Sparse recovery using OMP
    S_hat = zeros(size(D, 2), size(X, 2));
    for i = 1:size(X, 2)
        r = X(:, i);
        Omega = [];
        res_norm = norm(r);
        j = 0;
        
        while j < No
            j = j + 1;
            
            % Find the index of the atom that best represents the residual
            proj = abs(D' * r);
            [~, idx] = max(proj);
            
            % Add the index to the support set
            Omega = union(Omega, idx);
            
            % Compute the least squares solution on the support set
            S_hat(Omega, i) = pinv(D(:, Omega)) * X(:, i);
            
            % Update the residual
            r = X(:, i) - D(:, Omega) * S_hat(Omega, i);
            
            % Calculate the norm of the residual
            res_norm = norm(r);
            
            % If the residual is small enough, terminate
            if res_norm < 1e-6
                break;
            end
        end
    end
    
    % Update the dictionary
    D = X * pinv(S_hat);
    
    % Calculate the objective function (representation error)
    error = norm(X - D * S_hat, 'fro')^2;
    
    % Store the objective function value for the convergence graph
    objective(iter) = error;
end
mod_time = toc;

% Plot the convergence graph
figure;
plot(1:MaxIter, objective);
title('Convergence Graph');
xlabel('Iteration');
ylabel('Objective Function');

% Report the convergence time
disp("Convergence time (MOD): " + mod_time + " seconds");


%% MOD SSR 
% Load the observation matrix X from the file
load('hw6-part3.mat', 'X');

% Set the parameters
No = 2;  % Sparsity level
MaxIter = 100;  % Maximum number of iterations

% Initialize the dictionary matrix D using random values
D = randn(10, 40);

% Perform MOD algorithm
tic;
for iter = 1:MaxIter
    % Sparse recovery using OMP
    S_hat = zeros(size(D, 2), size(X, 2));
    for i = 1:size(X, 2)
        r = X(:, i);
        Omega = [];
        res_norm = norm(r);
        j = 0;
        
        while j < No
            j = j + 1;
            
            % Find the index of the atom that best represents the residual
            proj = abs(D' * r);
            [~, idx] = max(proj);
            
            % Add the index to the support set
            Omega = union(Omega, idx);
            
            % Compute the least squares solution on the support set
            S_hat(Omega, i) = pinv(D(:, Omega)) * X(:, i);
            
            % Update the residual
            r = X(:, i) - D(:, Omega) * S_hat(Omega, i);
            
            % Calculate the norm of the residual
            res_norm = norm(r);
            
            % If the residual is small enough, terminate
            if res_norm < 1e-6
                break;
            end
        end
    end
    
    % Update the dictionary
    D = X * pinv(S_hat);
    
    % Calculate the objective function (representation error)
    error = norm(X - D * S_hat, 'fro')^2;
    
    % Store the objective function value for the convergence graph
    objective(iter) = error;
end
mod_time = toc;

% Calculate successful recovery rate
recovered_atoms = [];
num_atoms = size(D, 2);
for i = 1:num_atoms
    corr_values = abs(D(:, i)' * D);
    corr_values(i) = 0;  % Exclude self-correlation
    max_corr = max(corr_values);
    
    if max_corr > 0.98
        recovered_atoms = [recovered_atoms i];
        D(:, i) = zeros(size(D, 1), 1);  % Remove the recovered atom
    end
end

recovery_rate = numel(recovered_atoms) / num_atoms * 100;

% Display the successful recovery rate
disp("Successful Recovery Rate (MOD): " + recovery_rate + "%");

load('hw6-part3.mat', 'S');
diff_S = S_hat - S;
norm_diff_S = norm(diff_S, 'fro');
norm_S = norm(S, 'fro');
E_MOD = norm_diff_S / norm_S;
disp("E_MOD: " + E_MOD);


%% K-SVD 

load('hw6-part3.mat');
No = 2;                % Sparsity level
maxIterations = 50;    % Maximum number of iterations
convergence = zeros(1, maxIterations);  % Array to store convergence values

tic;
[D_hat, S_hat_ksvd, convergence] = ksvd(X, randn(size(X, 1), 40), No, maxIterations);
toc;

figure;
plot(1:maxIterations, convergence, 'b-o');
xlabel('Iteration');
ylabel('Representation Error');
title('Convergence of K-SVD');
grid on;


load('hw6-part3.mat', 'S');
diff_S = S_hat_ksvd - S;
norm_diff_S = norm(diff_S, 'fro');
norm_S = norm(S, 'fro');
E_k_SVD = norm_diff_S / norm_S;
disp("E_K-SVD: " + E_k_SVD);



function [x] = omp(D, y, K)
    % D: Dictionary matrix
    % y: Observation matrix
    % K: Sparsity level

    N = size(D, 2);      % Dictionary size
    T = size(y, 2);      % Number of signals

    x = zeros(N, T);     % Sparse representations

    for t = 1:T
        r = y(:, t);     % Residual vector
        omega = [];      % Support set

        for k = 1:K
            proj = abs(D' * r);     % Compute inner products
            [~, idx] = max(proj);   % Find index of maximum inner product

            omega = [omega, idx];   % Update support set
            A = D(:, omega);        % Active columns of dictionary

            % Solve the least squares problem
            x_omp = pinv(A) * y(:, t);

            % Update the residual
            r = y(:, t) - A * x_omp;

            x(omega, t) = x_omp;     % Update sparse representation
        end
    end
end
function [D_hat, S_hat, convergence] = ksvd(X, D, No, maxIterations)
    % X: Observation matrix
    % D: Initial dictionary matrix
    % No: Sparsity level
    % maxIterations: Maximum number of iterations

    N = size(D, 2);        % Dictionary size
    T = size(X, 2);        % Number of signals

    S_hat = zeros(N, T);   % Sparse representations
    convergence = zeros(1, maxIterations);  % Convergence values

    for iter = 1:maxIterations
        % Sparse coding step (using OMP)
        S_hat = omp(D, X, No);

        % Dictionary update step
        for k = 1:N
            indices = find(S_hat(k, :) ~= 0);   % Find signals that use atom k

            if ~isempty(indices)
                % Update the k-th atom
                E = X(:, indices) - D * S_hat(:, indices) + D(:, k) * S_hat(k, indices);
                [U, ~, V] = svd(E, 'econ');
                D(:, k) = U(:, 1);
                S_hat(k, indices) = V(:, 1);  % Update with compatible dimensions
            end
        end

        % Calculate convergence value (Representation Error)
        convergence(iter) = norm(X - D * S_hat, 'fro')^2;
    end

    D_hat = D;   % Estimated dictionary
end






