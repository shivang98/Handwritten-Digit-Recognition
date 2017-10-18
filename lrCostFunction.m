function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

theta1 = [0;theta(2:length(theta))];
h = X * theta;
J = (1 / m) * sum(-y' * log(sigmoid(h)) - (1-y)' * log(1-sigmoid(h)));
J += (lambda/(2*m)) * sum(theta1 .^2);
grad = (1 / m) * sum((sigmoid(h)-y).*X)' + (lambda/m) * theta1;
% =============================================================

grad = grad(:);

end
