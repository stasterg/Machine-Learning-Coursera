function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0; Jr = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1 : m ;
  J += - y(i)*log(sigmoid(theta'*X(i,:)')) - (1-y(i))*log(1-sigmoid(theta'*X(i,:)')) ;
  grad += (sigmoid(theta'*X(i,:)') - y(i)) * X(i,:)' ;
end;

for j = 2 : size(theta);
  Jr += theta(j) * theta(j);  
  grad(j) = grad(j) + lambda * theta(j);
end;

J = (J + lambda * Jr * 0.5) / m;
grad = grad / m;
%grad = (grad + lambda * theta) / m;

% =============================================================

end
