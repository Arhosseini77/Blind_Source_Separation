clc;clear;close all;

%% Section 2 :

%% A 
N = 20;  % Set the value of N

% Generate the frame
theta = pi/N;
vectors = zeros(2, N);
for i = 1:N
    angle = theta * (i-1);
    vectors(:, i) = [cos(angle); sin(angle)];
end

% Plot the frame
figure;
hold on;
for i = 1:N
    plot([0, vectors(1,i)], [0, vectors(2,i)], 'b');
end
axis equal;
axis([-1 1 -1 1]);  % Set the axis limits
title('Frame for N = 20');
xlabel('X');
ylabel('Y');
hold off;

% Calculate mutual coherence for N in range 1 to 10
MC = zeros(1, N);
for n = 1:N
    G = vectors(:, 1:n)' * vectors(:, 1:n);
    G = abs(G) - eye(n);
    MC(n) = max(max(G));
end

% Plot mutual coherence
figure;
plot(1:N, MC, 'r', 'LineWidth', 2);
title('Mutual Coherence for N in range 1 to 10');
xlabel('N');
ylabel('Mutual Coherence');
grid on;

%% B 
N = 10;  % Set the value of N
% Generate the frame
frame = randn(3, N);
frame = frame ./ vecnorm(frame);
% Plot the frame
figure;
hold on;
for i = 1:N
    plot3([0, frame(1,i)], [0, frame(2,i)], [0, frame(3,i)], 'b', 'LineWidth', 2);
    view(3)
end
axis equal;
title('Frame for 3x10');
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;
hold off;
% Calculate mutual coherence
mutual_coherence = calculate_mutual_coherence(frame);
disp("Mutual Coherence: " + mutual_coherence);
function mutual_coherence = calculate_mutual_coherence(frame)
    % Calculate the mutual coherence
    G = frame' * frame;
    G = abs(G) - eye(size(frame, 2));
    mutual_coherence = max(max(G));
end
