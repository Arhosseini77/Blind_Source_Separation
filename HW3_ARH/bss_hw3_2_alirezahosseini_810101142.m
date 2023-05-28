clc;clear;close all;
%% Section 2
%% Load data
data_file = load('hw3-2.mat');
excitation_freqs = data_file.freq; % Vector of excitation frequencies
data_labels = data_file.label; % Vector of labels for each data
recorded_data = data_file.data; % Matrix of recorded data

%% Generate template matrices for each frequency
sampling_freq = 250; % Hz
time_samples = size(recorded_data,2); % Number of time samples
time_vector = 0:(1/sampling_freq):(time_samples-1)*(1/sampling_freq); % Time vector
num_freqs = length(excitation_freqs); % Number of frequencies used in experiment
template_matrices = cell(1,num_freqs); % Cell array to store template matrices

for i = 1:num_freqs
    % Generate sine and cosine values for the current frequency and its harmonics up to 40 Hz
    template = [sin(2*pi*excitation_freqs(i)*time_vector); cos(2*pi*excitation_freqs(i)*time_vector)];
    for k = 2:7
        if (excitation_freqs(i)*k > 40)
            break;
        end
        template = [template; sin(2*pi*k*excitation_freqs(i)*time_vector); cos(2*pi*k*excitation_freqs(i)*time_vector)];
    end
    template_matrices{i} = template; % Save the template matrix for the current frequency
end

%% CCA
num_trials = size(recorded_data, 3);
estimated_labels = zeros(size(data_labels));
for trial = 1:num_trials
    trial_data = recorded_data(:,:,trial);
    Ryy = trial_data*trial_data'; % Calculate cross-covariance matrix of recorded data
    correlation_coeffs = zeros(1,num_freqs);
    for i = 1:num_freqs
        template_matrix = template_matrices{i};
        Rxx = template_matrix*template_matrix'; % Calculate autocovariance matrix of template
        Rxy = template_matrix*trial_data'; % Calculate cross-covariance matrix between template and recorded data
        Ryx = Rxy'; % Calculate cross-covariance matrix between recorded data and template
        % Apply CCA to calculate correlation coefficient
        Sigma_1 = (Rxx^-0.5)*Rxy*(Ryy^-1)*Ryx*(Rxx^-0.5);
        [V, Lambda] = eig(Sigma_1);
        [lambda_vals, indices] = sort(diag(Lambda),'descend');
        V = V(:,indices);
        c = V(:,1);
        correlation_coeffs(i) = lambda_vals(1);

        Sigma_2 = (Ryy^-0.5)*Ryx*(Rxx^-1)*Rxy*(Ryy^-0.5);
        [V, Lambda] = eig(Sigma_2);
        [lambda_vals, indices] = sort(diag(Lambda),'descend');
        V = V(:,indices);
        d = V(:,1);
        correlation_coeffs(i) = lambda_vals(1);
    end
    [~, max_index] = max(correlation_coeffs); % Choose the frequency with the highest correlation coefficient
    estimated_labels(trial) = excitation_freqs(max_index);
end

%% Calculate accuracy of estimated labels
num_correct = sum(estimated_labels == data_labels);
accuracy = num_correct/length(data_labels);
fprintf('Accuracy of estimated labels: %0.2f\n', accuracy);
