function [ c, ceq ] = rotMatCon( x )

R = [x(4:6);x(7:9);x(10:12)];
t = x(13:15)';
ceq = [det(R) - 1;...
    norm(R) - 1];
c = [norm(t) - 0.1];

end

