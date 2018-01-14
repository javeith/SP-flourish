function [ err ] = optProblem_R(T_FA,T_SF,T_gimbalAngles, tMatrix)

for i = 1:size(T_FA,3)
    tmp = (T_FA(1:3,1:3,i) * T_gimbalAngles(1:3,1:3,i) * tMatrix * T_SF(1:3,1:3,i)) - eye(3);
    temp(i) = sum(tmp(:).^2);
end 

err = sum(temp);

end

