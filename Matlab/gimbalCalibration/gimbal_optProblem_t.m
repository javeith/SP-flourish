function [ err ] = optProblem_t(T_FA,T_SF,T_gimbalAngles, R_constant, tVector)

for i = 1:size(T_FA,3)
    tmp = (T_FA(:,:,i) * T_gimbalAngles(:,:,i) * [R_constant,tVector;0,0,0,1] * T_SF(:,:,i)) - eye(4);
    temp(i) = sum(tmp(1:3,4).^2);
end 

err = sum(temp);

end

