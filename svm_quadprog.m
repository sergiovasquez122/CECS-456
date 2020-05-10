function [u] = svm_quadprog(X,y)
	A = [y y.*X];
	N = size(X, 1);
	d = size(X, 2);
	p = zeros(d + 1, 1);
	c = ones(N,1);
	Q = [0 zeros(d, 1)';zeros(d, 1) eye(d)];
	u = quadprog(Q, p, -A, -c);
end
