clc;clear;close all;

%% load Data :

load('hw5.mat')
[M, N] = size(D);

%% Subset Selection Method , minimize Norm 0 
N0 = 3;
tic;
% iterate over all possible N0-sparse subsets of s
max_err = Inf;
for subset = combnk(1:N, N0).' % generate all possible subsets of size N0
    Dsubset = D(:, subset);
    if rank(Dsubset) < N0 % skip linearly dependent subsets
        continue;
    end
    s_temp = Dsubset \ x;
    err_temp = norm(x - Dsubset * s_temp);
    if err_temp < max_err % update if found a better subset
        s = zeros(N, 1);
        s(subset) = s_temp;
        max_err = err_temp;
    end
end
disp(['Subset Selection , minimize norm 0 : s = [' num2str(s(:).') '], error = ' num2str(max_err)])
nz_idx = find(s);
disp(['Non-zero elements: ' num2str(s(nz_idx).')])
disp(['Positions: ' sprintf('%d ', nz_idx)])
toc;

%% L2 norm, minimize ||s||_2^2
tic;
s = pinv(D) * x;
err = norm(x - D * s);
disp(['L2 norm, minimize ||s||_2^2: s = [' num2str(s(:).') '], error = ' num2str(err)])
nz_idx = find(s);
disp(['Non-zero elements: ' num2str(s(nz_idx).')])
disp(['Positions: ' sprintf('%d ', nz_idx)])
toc;
% plot s_hat
figure;
stem(s);
title('Recovered sparse signal (L2 norm)');
% check if s is sparse
figure;
stem(abs(s));
title('Magnitude of recovered sparse signal (L2 norm)');

%% MP
%% MP ( Known No ) :
N0 = 3;
tic;
x_mp = x;
posMP = zeros(1,N0);
sMP = zeros(N,1);
for i = 1:N0
   ro = x_mp'*D;
   [~,posMP(i)] = max(abs(ro));
   sMP(posMP(i)) = ro(posMP(i));
   x_mp = x_mp - sMP(posMP(i))*D(:,posMP(i));
end
disp('MP for Known N0:');
disp(['Non-zero elements: ' num2str(sMP(posMP).')])
disp(['Positions: ' sprintf('%d ', posMP)])
disp(['Error: ', num2str(norm(x-D*sMP))])
disp(['Run Time: ', num2str(toc), ' seconds'])

%% MP ( UnKnown No ) :
N0_max = 10; % maximum expected value of N0
x_mp1 = x;
posMP = zeros(1,N0_max);
sMP = zeros(N,1);
max_err = Inf;
for N0 = 1:N0_max
    ro = x_mp1' * D;
    [~, pos] = max(abs(ro));
    posMP(N0) = pos;
    s_temp = pinv(D(:, posMP(1:N0))) * x;
    err_temp = norm(x - D(:, posMP(1:N0)) * s_temp);
    if err_temp < max_err
        sMP(posMP(1:N0)) = s_temp;
        max_err = err_temp;
    end
    x_mp1 = x - D(:, posMP(1:N0)) * sMP(posMP(1:N0));
    if norm(x_mp1) < 1e-1 % stop iteration if residual is too small
        break;
    end
end
sMP = sMP(posMP(1:N0)); % trim the sMP vector to remove zeros
posMP = posMP(1:N0); % trim the posMP vector to remove zeros
disp('MP with unknown N0:')
disp(['Non-zero elements: ' num2str(sMP.')])
disp(['Positions: ' sprintf('%d ', posMP)])

%% OMP 
%% OMP algorithm for known sparsity N0
tic;
N0 = 3
x1 = x;
posOMP = zeros(1,N0);
sOMP = zeros(N,1);
for i=1:N0
   ro = x1'*D;
   [~,posOMP(i)] = max(abs(ro));
   Dsub = D(:,posOMP(1:i));
   sOMP_sub = pinv(Dsub)*x;
   sOMP(posOMP(1:i)) = sOMP_sub(1:i);
   x1 = x - Dsub*sOMP_sub;
end
disp('OMP with known N0 :');
disp(['Non-zero elements: ' num2str(sOMP(find(sOMP~=0))')])
disp(['Error: ' num2str(norm(x-D*sOMP))]);
toc;

%% OMP using IHT algorithm for unknown N0
% Initialize parameters
tol = 1e-4;
max_iter = 1000;
s_omp = zeros(N, 1);
residual = x;

% Main loop
for i = 1:max_iter
    % Compute correlation of residual with dictionary
    corr = abs(D' * residual);
    
    % Find index with maximum correlation
    [~, index] = max(corr);
    
    % Add index to support
    support = unique([find(s_omp); index]);
    
    % Compute least-squares solution for current support
    s_omp(support) = pinv(D(:, support)) * x;
    
    % Compute new residual
    residual = x - D * s_omp;
    
    % Check stopping criterion
    if norm(residual) < tol
        break;
    end
end

% Print results
disp('OMP using IHT algorithm for unknown N0:')
disp(['Number of iterations: ' num2str(i)])
disp(['Non-zero elements: ' num2str(length(find(s_omp ~= 0)))])
disp(['Non-zero elements: ' num2str(s_omp(find(s_omp ~= 0))')])
disp(['Error: ' num2str(norm(x - D * s_omp))])

%% BP
%% BP using linear programming
tic 
f = ones(2 * N, 1); % objective function
Aeq = [D, -D]; % equality constraints
beq = x;
lb = zeros(2 * N, 1); % lower bound for variables
yhat = linprog(f, [], [], Aeq, beq, lb, []); % solve linear program
splus = yhat(1:N); % extract splus and sminus
sminus = yhat(N+1:end);
sBP = splus - sminus; % reconstruct sparse signal
posBP = find(abs(sBP) > 0.01)'; % find non-zero elements
disp('BP:')
disp(['Non-zero elements: ' num2str(sBP(posBP)')]) % display non-zero elements
disp(['Positions: ' num2str(posBP)]) % display positions
disp(['Runtime: ' num2str(toc) ' seconds']) % display runtime

%% IRLS : 
% initialize weights and set maximum iterations
w = ones(N,1); % initialize weights with ones of size N
ITRmax = 100; % set the maximum iteration

% initialize variables
sIRLS = zeros(N,1); % initialize sIRLS with zeros of size N
sIRLS_prev = zeros(N, 1); % initialize sIRLS_prev with zeros of size N
y = zeros(M,1); % initialize y with zeros of size M
tic 

% iterate until maximum iterations or convergence
for itr = 1:ITRmax
W = diag(w); % create a diagonal matrix W with the weight vector w
y = pinv(D*(W^-1))*x; % compute y using the Moore-Penrose pseudoinverse of the product of D and the inverse of W, and the input data x
sIRLS = y./sqrt(w); % update sIRLS with the element-wise division of y by the square root of w

% update weights
for n = 1:N % for each element of the weight vector
    if abs(sIRLS(n)) < eps % if the absolute value of the corresponding sIRLS element is less than the deviation eps
        w(n) = 1e10; % set the weight to a large value
        sIRLS(n) = 0; % set the corresponding sIRLS element to zero
    elseif abs(sIRLS(n)) > 1e6 % if the absolute value of the corresponding sIRLS element is greater than 1e6
        w(n) = 1e-10; % set the weight to a small value
    else % otherwise
        w(n) = 1./abs(sIRLS(n)); % update the weight with the inverse of the absolute value of the corresponding sIRLS element
    end
end

% check for convergence
if norm(sIRLS - sIRLS_prev) < eps % if the L2-norm of the difference between the current and previous sIRLS vectors is less than the deviation eps
    break % exit the loop
else % otherwise
    sIRLS_prev = sIRLS; % update the previous sIRLS vector
end
end

% display results
posIRLS = find(abs(sIRLS) > 0.1)'; % find the indices of the non-zero elements of sIRLS
fprintf('IRLS:\n');
fprintf('Non-zero elements of S:\n');
fprintf('%d ', posIRLS); % display the indices of the non-zero elements of sIRLS
fprintf('\n');
fprintf('Values of non-zero elements of S:\n');
fprintf('%.2f ', sIRLS(posIRLS)); % display the values of the non-zero elements of sIRLS
fprintf('\n');
fprintf('Runtime: %.4f seconds\n', toc); % display the elapsed time
fprintf('Error: %.4f\n', norm(x - D*sIRLS)); % display the error between the input data x and the estimated sparse signal DsIRLS