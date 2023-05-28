clc;clear;close all;
%% Section 1 
s1=-3 + rand(1,1000)*6;
s2=-2 + rand(1,1000)*4;
s1= s1 - mean(s1);
s2= s2 - mean(s2);
A=[1 -2;2 -1;3 -2];
x = A * [s1;s2];
x1 = x(1,:);
x2 = x(2,:);
x3 = x(3,:);
%% Section 1 Part 1 
scatter3(x1,x2,x3);
Rx = x*transpose(x);
[U,L] = eig(Rx);
%% Section 1 Part 2
C = linsolve([U(:,2),U(:,3)],A);
%% Section 1 Part 3
B = [L(2,2)^(-0.5) 0;0 L(3,3)^(-0.5)] * transpose([U(:,2),U(:,3)]);
Z = B*x;
Z1 = Z(1,:);
Z2 = Z(2,:);
scatter(Z1,Z2);

%% Section 1 Part 4
[Q,S,V] = svd(x) ;
rank_x = rank(x) ;

%% Section 1 Part 5
source = [s1;s2] ; 
F_t = linsolve(transpose(Z),transpose(source));
F = transpose(F_t);
%% Section 1 Part 6
sum = L(1,1)+L(2,2)+L(3,3);
B1 = [L(3,3)^(-0.5)] * transpose([U(:,3)]);
z = B1*x;
scatter3(z,z,z);

%% Section 2

rng(1)
t = 0:1/10^6:1/10^3; 
c = 3*10^8;
f1 = 20*10^3;
f2 = 10*10^3;
fc = 150*10^6;
t1=10*pi/180;
t2=20*pi/180;
s1 = exp(1i*2*pi*f1*t);
s2 = exp(1i*2*pi*f2*t);
a1 = ones(10,1);
a2 = ones(10,1);

%% Section 2 part A
for i=2:10
    a1(i,1) = exp(-1i*2*pi*fc*(i-1)*sin(t1)/c);
    a2(i,1) = exp(-1i*2*pi*fc*(i-1)*sin(t2)/c);
end
y1 = a1*s1;
y2 = a2*s2;
y = y1 + y2 ;
y = y + randn(size(y));
%% Section 2 part B , C
[U,L,V] = svd(y);
Usig = [U(:,1),U(:,2)];
Unull = U(:,3:10);
beamforming = zeros(90,1);
music = zeros(90,1);
idx = 1;

for teta_priod1 = 0:90
    a1 = ones(10,1);
    a2 = ones(10,1);
    for i=2:10
        a1(i,1) = exp(-1i * 2 * pi * fc * (i-1) *sin(teta_priod1 * pi / 180) /c );
    end
    beamforming(idx) =  norm( (a1)' * Usig ) ;
    music(idx) =  1/ (norm( (a1)' * Unull ) ) ;
    idx = idx + 1;
end
teta = 0:90;
plot(teta,music);
title('music method');
xlabel('degree');
figure;
plot(teta,beamforming);
title('beamforming method');
xlabel('degree');

%% Section 2 part D,E

Vsig = [V(1,:);V(2,:)];
Vnull = V(3:1001,:);

beamforming = zeros(51,1);
music = zeros(51,1);
idx = 1;
for f = 0:50
    s1 = exp(1i*2*pi*f*1000*t);
    beamforming(idx) =  norm( (s1) * Vsig' ) ;
    music(idx) =  1/ (norm( (s1) * Vnull' ) ) ;
    idx = idx + 1;
end
t_plot = 0:50;
plot(0:50,music);
title('MUSIC');
xlabel('K Hz');
figure;
plot(0:50,beamforming);
title('beamforming');
xlabel('K Hz');




