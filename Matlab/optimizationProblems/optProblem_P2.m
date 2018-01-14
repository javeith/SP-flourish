function [ err ] = optProblem_P2(F, P1, P2)

tmp = [P2,1] * F * [P1,1]';
err = tmp;

end

