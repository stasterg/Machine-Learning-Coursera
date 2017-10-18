function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%size(Theta1)
%size(Theta2)
                 
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
%size(Theta1_grad)
%size(Theta2_grad)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Recode the labels into unitary vector-like form
y_h = zeros(m, num_labels);
for l = 1 : num_labels;
  y_h(:, l) = (y(:) == l);
end;  
%max(max(y_h))

% FORWARD PROPAGATION
% Add ones to the X data matrix (or a1)
X = [ones(m, 1) X];
% Initialize activation matrices
z2 = zeros(size(Theta1, 1), size(X, 1));
a2 = zeros(size(z2, 1), size(z2, 2));
z3 = zeros(size(Theta2, 1), size(X, 1));
a3 = zeros(size(z3, 1), size(z3, 2));

% Hidden layer
z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(m, 1)' ; a2];
%size(a2)

% Output layer
z3 = Theta2 * a2;
a3 = sigmoid(z3);
%size(a3)

% Cost function J for the neural network wo/ regularization
for i = 1 : m;
  for l = 1 : num_labels;
    J += 1/m * (-y_h(i,l)*log(a3(l,i)) - (1-y_h(i,l))*log(1-a3(l,i))) ;
  end;
end;

% REGULARIZATION TERMS
J += lambda/2/m * (sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2)));

% BACKWARD PROPAGATION
% Initialization
v_a1 = zeros(size(X,2));
v_a2 = zeros(size(a2,1));
v_a3 = zeros(size(a3,1));
v_z2 = zeros(size(z2,1));
v_z3 = zeros(size(z3,1));
delta3 = zeros(size(v_a3), 1);
delta2 = zeros(size(v_a2), 1);
D1 = 0; D2 = 0;


for t = 1 : m;
  
  v_a1 = X(t, :)';
  %size(v_a1)
  %v_a1 = [1 ; v_a1];
  
  % Hidden layer
  v_z2 = Theta1 * v_a1;
  v_a2 = sigmoid(v_z2);
  v_a2 = [1 ; v_a2];
  %printf('v_a2 = %i5', size(v_a2))

  % Output layer
  v_z3 = Theta2 * v_a2;
  v_a3 = sigmoid(v_z3);
  %size(v_a3)
  
  %size(y_h(1, :))
  %size(delta3)
  delta3(:,:) = v_a3(:,:) - y_h(t, :)';
  delta2 = Theta2' * delta3 .* sigmoidGradient([1; v_z2]) ; 
  %v_a2 .*(1-v_a2.^2); will return computational error !!! 
  %(at least for testing purposes)
  
  D2 += delta3 * v_a2';
  D1 += delta2(2:end) * v_a1';
  
end;
%size(D2)
%size(D1)

D1 = D1 / m ;
D2 = D2 / m ;

% REGULARIZATION
D1(:, 2:end) += lambda / m * Theta1(:, 2:end) ; 
D2(:, 2:end) += lambda / m * Theta2(:, 2:end) ;

Theta1_grad = D1 ; 
Theta2_grad = D2 ;


%delta3 = a3 - y_h';
%delta2 = Theta2' * delta3 .* a2 .* (1-a2.^2);


%J = 1/m * sum(-y.*log(sigmoid(X * theta)) - (1-y).*log(1-sigmoid(X * theta)));
%J = J + lambda/2/m*sum(theta(2:end).*theta(2:end))
%size(J)


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
