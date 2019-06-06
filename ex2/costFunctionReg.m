function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J0 = 0;
J = 0;
grad = zeros(size(theta));
g = sigmoid ( X*theta);
X2 = X(:, 2:size(X,2));
theta2 = theta(2: size(theta));
g2 = sigmoid ( X2*theta2)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J = ( 1 / m )*( -log(g2')*y -log(1-g2')*(1-y) ) + ( lambda/(2*m) )* sum(theta2.^2);
J0 =  -y(1,1)*log(g(1,1)) -(1-y(1,1))*log(1-g(1,1));
J = J + J0 ;
grad0 = ( 1/m ) * sum(( g(1,1) - y(1,1) )* X(:,1));
grad = ( 1 / m ) * sum( ( g2 - y ) .* X2 ) + ( lambda/m ).*theta2';
grad = [grad0, grad];
% =============================================================

end
