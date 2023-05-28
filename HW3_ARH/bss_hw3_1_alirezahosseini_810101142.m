clc;clear;close all;

%% section 1 :

%% load data:
dataset = load('hw3-1.mat');
TrainData_class1 = dataset.TrainData_class1;
TrainData_class2 = dataset.TrainData_class2;
TestData = dataset.TestData;
TestLabel = dataset.TestLabel;
%% Zero mean
TrainData_class1 = TrainData_class1 - mean(TrainData_class1,2);
TrainData_class2 = TrainData_class2 - mean(TrainData_class2,2);
TestData = TestData - mean(TestData,2);

%% A , find W csp
for i=1:60
    if(i==1)
        Rx = TrainData_class1(:,:,i) * (TrainData_class1(:,:,i))';
    else
        Rx = Rx + TrainData_class1(:,:,i) * (TrainData_class1(:,:,i))';
    end
end
Rx = Rx/60;

for i=1:60
    if(i==1)
        Ry = TrainData_class2(:,:,i) * (TrainData_class2(:,:,i))';
    else
        Ry = Ry + TrainData_class2(:,:,i) * (TrainData_class2(:,:,i))';
    end
end
Ry = Ry/60;

[V, landa_o] = eig(Rx , Ry, 'vector');
[V, landa] = eig(Rx , Ry);
[landa2, ind] = sort(landa_o,'descend');
V_sorted = V(:, ind);
landa_sorted = landa(ind, ind);
subplot(1,2,1);
plot((V_sorted(:,1))'*TrainData_class1(:,:,49));
hold on;
plot((V_sorted(:,1))'*TrainData_class2(:,:,49));
legend('W1.X1(49)','W1.X2(49)');
subplot(1,2,2);
plot((V_sorted(:,30))'*TrainData_class1(:,:,49));
hold on;
plot((V_sorted(:,30))'*TrainData_class2(:,:,49));
legend('W30.X1(49)','W30.X2(49)');

var((V_sorted(:,30))'*TrainData_class1(:,:,49))
var((V_sorted(:,30))'*TrainData_class2(:,:,49))

var((V_sorted(:,1))'*TrainData_class1(:,:,49))
var((V_sorted(:,1))'*TrainData_class2(:,:,49))

%% B  - plot
subplot(1,2,1);
plot(abs(V_sorted(:,1)));
title('W1');
subplot(1,2,2);
plot(abs(V_sorted(:,30)));
title('W30');

%% C - W LDA
%% C
clc;
W=[V_sorted(:,1:7),V_sorted(:,24:30)];
W1=[V_sorted(:,1:30)];
for i=1:60
    if(i==1)
        mo1 = var((W.' * TrainData_class1(:,:,i)).').';
    end
    mo1 = mo1 + var((W.' * TrainData_class1(:,:,i)).').'; 
end
mo1 = mo1/60;
for i=1:60
    if(i==1)
        mo2 = var((W.' * TrainData_class2(:,:,i)).').';
    end
    mo2 = mo2 + var((W.' * TrainData_class2(:,:,i)).').';
end
mo2 = mo2/60;

x1 = zeros(14,14);
for i=1:60
    x1 = x1 + ( var((W.' * TrainData_class1(:,:,i)).').' - mo1 )*( var((W.' * TrainData_class1(:,:,i)).').' - mo1 ).'; 
end
x1 = x1/60;
x2 = zeros(14,14);
for i=1:60
    x2 = x2 + ( var((W.' * TrainData_class2(:,:,i)).').' - mo2 )*( var((W.' * TrainData_class2(:,:,i)).').' - mo2 ).'; 
end
x2 = x2/60;


[WLDA , landaLDA] = eig( (mo1 - mo2)*(mo1 - mo2).' , x2 + x1);
mo1_new = WLDA(:,14).'*mo1;
mo2_new = WLDA(:,14).'*mo2;
C = (mo1_new + mo2_new)/2 ;

%% D, E , Test Model
count = 0;
pred = zeros(1,40);
for i=1:40
    data = var((W' * TestData(:,:,i))')';
    offset = WLDA(:,14)'*data;
    if(offset > C )
        label = 1;
        pred(i) = 1;
    else
        label = 2;
        pred(i) = 2;
    end
    if( label == TestLabel(i) )
        count = count + 1;

    end
    
end

acc = count / 40 ;

figure;
plot(pred,'*');
hold on;
plot(TestLabel,'p');
legend('model_predicted','real_label');
xlabel('index');
ylabel('label');
