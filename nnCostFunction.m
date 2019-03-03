function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));
% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1) X]'; %'
z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2];
a3 = sigmoid(Theta2 * a2);
K = num_labels;
y_k = eye(num_labels);
cost = zeros(K,1);
for i=1:m
    cost =cost+(-y_k(:,y(i)).*log(a3(:,i))-(1 -y_k(:,y(i))).*log(1-a3(:,i)));
end
J = sum(cost)/m;
regularizationTerm =sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2));%from 2 to ignore bias
J = J + regularizationTerm*lambda/(2 * m);

Delta1_2 = zeros(size(Theta2));
Delta1_1 = zeros(size(Theta1));
for r=1:m
    a1 = [1 ; X(r,:)'];
    z2 = Theta1 *a1 ;
    a2 = sigmoid(z2);
    a2 = [1;a2];%bias
    z3 = Theta2 *a2;
    a3 = sigmoid(z3);
    delta_3 = a3 - y_k(:,y(r));
    delta_2 = (Theta2'* delta_3).*[1;sigmoidGradient(z2)];
    delta_2 = delta_2(2:end,1);
    Delta1_2 = Delta1_2+delta_3*a2';
    Delta1_1 = Delta1_1+delta_2*a1';
end
Theta1_grad=Delta1_1/m+(lambda/m)*[zeros(size(Theta1,1),1),Theta1(:,2:end)];
Theta2_grad=Delta1_2/m +(lambda/m)*[zeros(size(Theta2,1),1),Theta2(:,2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];
end














