function pretaxincome=BM2022_PreTaxIncome(h,aprime,a,z,r,delta,alpha)

w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));

pretaxincome=w*h*z+r*a;

end