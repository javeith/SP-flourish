function [ k1, k2, p1, p2, k3, err ] = Equi2RadTan( imageSize, D, K )

% Split up K and D into parameters of interest
fx = K(1,1);
fy = K(2,2);
cx = K(1,3);
cy = K(2,3);
k1 = D(1);
k2 = D(2);
k3 = D(3);
k4 = D(4);

image = zeros(imageSize);
[i, j] = find(~isnan(image));

% Xp = the xyz vals of points on the z plane
Xp = K\[j i ones(length(i),1)]';

% Now we calculate how those points distort i.e forward map them through the distortion
x = Xp(1,:);
y = Xp(2,:);

% Get distorted points
theta = atan(sqrt(x.^2 + y.^2));
theta = (theta + k1*theta + k2*theta.^3 + k3*theta.^5 + k4*theta.^7);
omega = atan2(y,x);
u = theta.*cos(omega);
v = theta.*sin(omega);

% Optimize
err_func = @(d) getErr(u,v,x,y,d);
[D,err] = fminunc(err_func, [0,0,0,0,0]);

k1 = D(1);
k2 = D(2);
k3 = D(5);
p1 = D(3);
p2 = D(4);

end

function [ err ] = getErr( u, v, x, y, D)

k1 = D(1);
k2 = D(2);
k3 = D(5);
p1 = D(3);
p2 = D(4);

r2 = x.^2 + y.^2;
u_radtan = x.*(1+k1*r2 + k2*r2.^2 + k3*r2.^3) + 2*p1.*x.*y + p2*(r2 + 2*x.^2);
v_radtan = y.*(1+k1*r2 + k2*r2.^2 + k3*r2.^3) + 2*p2.*x.*y + p1*(r2 + 2*y.^2);

err = mean((u-u_radtan).^2 + (v-v_radtan).^2);

end