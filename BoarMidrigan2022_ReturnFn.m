function F=BoarMidrigan2022_ReturnFn(h,aprime,a,z,r,tau_s,tau,xi,tau_a,xi_a,iota,theta,gamma,delta,alpha)

F=-Inf;

w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));

pretaxincome=w*h*z+r*a;

incometax=pretaxincome-((1-tau)/(1-xi))*(pretaxincome^(1-xi)) - iota;
wealthtax=a-((1-tau_a)/(1-xi_a))*(a^(1-xi_a));

resources=(pretaxincome-incometax)+(a-wealthtax)-aprime;

c=resources/(1+tau_s); % consumption

if c>0
    if theta==1
        F=log(c)-(h^(1+gamma))/(1+gamma);
    else
        F=(c^(1-theta))/(1-theta)-(h^(1+gamma))/(1+gamma);
    end
end


end