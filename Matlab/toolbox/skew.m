function skew_ret = skew(v)
skew_ret = [0 -v(3) v(2);...
            v(3) 0 -v(1);...
            -v(2) v(1) 0];