function c=BM2022_Consumption(h,aprime,a,z,r,tau_s,tau,xi,tau_a,xi_a,iota,delta,alpha)

w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));

pretaxincome=w*h*z+r*a;

incometax=pretaxincome-((1-tau)/(1-xi))*(pretaxincome^(1-xi)) - iota;
wealthtax=a-((1-tau_a)/(1-xi_a))*(a^(1-xi_a));

resources=(pretaxincome-incometax)+(a-wealthtax)-aprime;

c=resources/(1+tau_s); % consumption

end