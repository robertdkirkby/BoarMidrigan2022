% Boar & Midrigan (2022) - Efficient Redistribution

% Essentially, we are going to be solving transition paths in an largely plain-vanilla Aiyagari model with endogenous labour.
% I denote the markov process on per-unit-time-worked earnings as z, they called it e.

% Paper is missing info about how Government works.
% There is a per-period Government budget constraint, and government debt in the initial stationary general eqm is 100% of GDP. 
% But what is missing is how B_t is determined over the transition path? Is is constant?, or always stays at 100% of GDP?, or?
% BM2022, pg 85: "For each [tax rate] experiment we adjust the lump-sum transfer iota_t to ensure that the government budget constraint is satisfied at all dates."
% I am assuming this is inteded to mean that both G_t and B_t are assumed to be constant.
% Relatedly, footnote 13 says "We have experimented with allowing the planner to also choose debt optimally and found that raising government
% debt has similar implications to increasing the wealth tax." [there are two more sentences in footnote 13 that I omit here] [Rob's note: makes senses as 
% it depresses r, which is indistinguishable from a wealth tax] 
% Based on all this I am just keeping B_t constant over the transition path.

% Note: the transition paths in this model are pretty fragile relative to
% most papers, this is because of iota, the lump-sum transfers which make
% the transition paths easy to fail. The 'update' of transition path
% iterations is key to avoiding them failing.

n_d=101;
n_a=501;
n_z=12; % 11 to discretize AR(1), plus one for the super-star state

figure_c=0; % counter for figures

%% Parameters

% Preferences
Params.beta=0.975; % discount factor
Params.theta=1; % curvature of utility of consumption (CRRA parameter)
Params.gamma=2; % curvature of utility of leisure (inverse Frisch elasticity)

% Production
Params.alpha=1/3; % capital share of income
Params.delta=0.06; % depreciation rate

% Taxes
% Consumption tax
Params.tau_s=0.065; 
% Income tax
Params.tau=0.263;
Params.xi=0.049;
% Wealth tax
Params.tau_a=0;
Params.xi_a=0;
% Capital gains tax
Params.tau_k=0.2;
% Lump-sum transfer, relative to per-capita GDP
Params.iota_target=0.167;
Params.iota=0.1; % guess, will be calibrated so we hit iota_target

% Government debt-to-GDP
Params.Bbar=1;

% Earnings productivity process
Params.rho_z=0.982; % autocorrelation of z
Params.sigma_e=0.2; % std dev of innovations to z
Params.p=2.2e-6; % probability to enter the super-star state
Params.q=0.990; % probability to stay in the super-star state
Params.zbar=504.3; % ability super-star state relative to mean

% Following parameters are deteremined in general eqm, these are just
% initial guesses (I had worse guesses on the first run, these are updated
% so the initial general eqm is quicker to solve by using less bad guesses)
Params.r=0.05;   % 0.035
Params.iota=0.3; % 0.31
Params.B=1.8;    % 1.84
Params.G=0.05;   % 0.025

%% Grids
maxh=1.3; % I set this to 2, solved initial and final stationary eqm, and based on those no-one chooses more than 1.2, so using 1.3 as the max
d_grid=linspace(0,maxh,n_d)'; % labor supply

Params.maxa=100; % max assets [10 is the max on the x-axis of Fig 4, so seems reasonable? Solved model, clearly too low as people hit the top; later on, turns out Fig 4 x-axis was 'relative to mean wealth', so no wonder 10 was too low]
a_grid=Params.maxa*(linspace(0,1,n_a)'.^3); % assets: 0 to max, ^3 adds curvature so more points near 0


[z_grid_pre,pi_z_pre]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_e,n_z-1); % BM2022 used Rouwenhorst
% Make it so that mean of z is 1
z_grid_pre=exp(z_grid_pre);
[meanz,~,~,~]=MarkovChainMoments(z_grid_pre,pi_z_pre);
z_grid_pre=z_grid_pre./meanz;
[meanz,~,~,statdistz]=MarkovChainMoments(z_grid_pre,pi_z_pre);
% Add in the super-star state
z_grid=[z_grid_pre; Params.zbar*meanz]; % meanz is very close to one in any case
pi_z=[(1-Params.p)*pi_z_pre, Params.p*ones(n_z-1,1); (1-Params.q)*statdistz', Params.q];
% Page 82, "We assume that agents transit from the normal to the super-star
% state with a constant probability p and remain there with probability q.
% When agents return to the normal state, they draw a new ability from the
% ergodic distribution associated with the AR(1) process."


%% Return fn and discount factor
DiscountFactorParamNames={'beta'};

ReturnFn=@(h,aprime,a,z,r,tau_s,tau,xi,tau_a,xi_a,iota,theta,gamma,delta,alpha)...
    BoarMidrigan2022_ReturnFn(h,aprime,a,z,r,tau_s,tau,xi,tau_a,xi_a,iota,theta,gamma,delta,alpha);

%% Test the value fn and policy fn
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=20;
simoptions.gridinterplayer=vfoptions.gridinterplayer;
simoptions.ngridinterp=vfoptions.ngridinterp;

tic;
[V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
vftime=toc

%% Test for Agent Dist
StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z,simoptions);

%% Setup model moments
FnsToEvaluate.A=@(h,aprime,a,z) a;
FnsToEvaluate.L=@(h,aprime,a,z) h*z;
FnsToEvaluate.TaxRevenue=@(h,aprime,a,z,r,tau,xi,iota,delta,alpha,tau_a,xi_a) BM2022_IncomeTaxRevenue(h,aprime,a,z,r,tau,xi,iota,delta,alpha) + BM2022_WealthTaxRevenue(h,aprime,a,z,tau_a,xi_a);

% Test the model moments
AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,d_grid,a_grid,z_grid,simoptions);

%% Solve initial stationary general eqm
GEPriceParamNames_pre={'r','iota','B','G'};

% intermediateEqns take Params and AggVars as inputs, output can be used in
% GeneralEqmEqns. They get evaluated in order, so can use output from one as an input to a later one.
heteroagentoptions.intermediateEqns.K=@(A,B) A-B; % physical capital K: A is asset supply, K and B are asset demand (some saving are used for gov debt, rest goes to physical capital)
heteroagentoptions.intermediateEqns.Y=@(K,L,alpha) (K^(alpha))*(L^(1-alpha)); % output Y

GeneralEqmEqns_pre.capitalmarket=@(r,K,L,alpha,delta) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta); % interest rate equals marginal product of capital (net of depreciation)
GeneralEqmEqns_pre.iotacalib=@(iota,iota_target,Y) iota_target - iota/Y; % get iota to GDP-per-capita ratio correct
% Note: because iota is same for everyone, just use it directly here rather than needing to calculate it as an aggregate var.
GeneralEqmEqns_pre.govbudget=@(r,B,G,TaxRevenue) (1+r)*B+G-(B+TaxRevenue); % Balance the goverment budget
% Include the calibration target as a general eqm constraint
GeneralEqmEqns_pre.govdebtcalib=@(Bbar,B,Y) Bbar - B/Y; % Gov Debt-to-GDP ratio of Bbar

heteroagentoptions.verbose=1;
[p_eqm_init,GECondns_init]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns_pre, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames_pre,heteroagentoptions, simoptions, vfoptions);
% Update Params based on general eqm
Params.r=p_eqm_init.r;
Params.iota=p_eqm_init.iota;
Params.B=p_eqm_init.B;
Params.G=p_eqm_init.G;

[V_init,Policy_init]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
StationaryDist_init=StationaryDist_Case1(Policy_init,n_d,n_a,n_z,pi_z,simoptions);

% Add some further FnsToEvaluate so can plot the kinds of outputs shown in Figure 3
FnsToEvaluate2=FnsToEvaluate;
FnsToEvaluate2.Consumption=@(h,aprime,a,z,r,tau_s,tau,xi,tau_a,xi_a,iota,delta,alpha) BM2022_Consumption(h,aprime,a,z,r,tau_s,tau,xi,tau_a,xi_a,iota,delta,alpha);

AggVars_init=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_init,Policy_init,FnsToEvaluate2,Params,[],n_d,n_a,n_z,d_grid,a_grid,z_grid,simoptions);
AllStats_init=EvalFnOnAgentDist_AllStats_Case1(StationaryDist_init,Policy_init,FnsToEvaluate2,Params,[],n_d,n_a,n_z,d_grid,a_grid,z_grid,simoptions);

K_init=AggVars_init.A.Mean-Params.B;
wage_init=(1-Params.alpha)*((p_eqm_init.r+Params.delta)/Params.alpha)^(Params.alpha/(Params.alpha-1));
output_init=(K_init^(Params.alpha))*(AggVars_init.L.Mean^(1-Params.alpha));

C_t=AggVars_init.Consumption.Mean;
C_tplus1=AggVars_init.Consumption.Mean^(-Params.theta); % C_t and C_t+1 are same thing in stationary general eqm
laborwedge_init=wage_init*(C_t^(-Params.theta))/(AggVars.L.Mean^Params.gamma); % Rearrange eqn at bottom of page 80 for varthetabar
savingswedge_init=Params.beta*(1+Params.r)*(C_t^(-Params.theta))/(C_tplus1^(-Params.theta)); % Rearrange eqn (4) on pg 81 to get zetabar_t, the aggregate capital wedge 

%% From now on, just keep G and B unchanged
GEPriceParamNames={'r','iota'};
GeneralEqmEqns.capitalmarket=GeneralEqmEqns_pre.capitalmarket;
GeneralEqmEqns.govbudget=GeneralEqmEqns_pre.govbudget;

%% Boar and Midrigan do optimal, which involves comparing lots of possible tax reforms.
% According to their computational appendix, they first did a rough grid on
% tax rates, then started an optimization routine from the best point on
% this grid [to get the welfare maximizing tax rates].
% Note, once we solve one we can use solution as initial guess for the next, so
% runtime would be much less than simply repeating this exercise.

%% Here we just do one tax reform, to show how it is done.

% Note, pre-tax reform the initial eqm is based on
% Params.tau=0.263;
% Params.xi=0.049;
% Params.tau_a=0;
% Params.xi_a=0;

% Tax reform (this is the one they found to be optimal for 'utilitarian welfare'; I had to eyeball the tau and tau_a out of Figure 4 as the exact number does not appear to be in paper, the xi and xi_a are explicit in Figure 4)
Params.tau=0.56; % income tax
Params.xi=0.065;
Params.tau_a=-0.002; % wealth tax
Params.xi_a=0.0017;

Params.r=0.05; % need to substantially increase r as otherwise A is less than B with the tax reform setup, and so K is negative and things don't work

%% Final stationary general eqm
heteroagentoptions.verbose=2
heteroagentoptions.constrainpositive={'r'}; % it kept trying negative r, so ruling that out
tic;
[p_eqm_final,GECondns_final]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
GEtime2=toc
% Update Params based on general eqm
Params.r=p_eqm_final.r;
Params.iota=p_eqm_final.iota;

[V_final,Policy_final]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
StationaryDist_final=StationaryDist_Case1(Policy_final,n_d,n_a,n_z,pi_z,simoptions);

% Double-check that no-one is hitting the max of the a_grid
asset_cdf=cumsum(sum(StationaryDist_init,2));
asset_cdf2=cumsum(sum(StationaryDist_final,2));
figure_c=figure_c+1;
figure(figure_c);
plot(a_grid,asset_cdf,a_grid,asset_cdf2)
legend('initial eqm', 'final eqm')
title('cdf of assets in stationary general eqm (check not hitting the top of grid)')


save BM2022pre.mat


%% Solve the transition path
T=100; % BM2022 Fig 3 has this as x-axis, so seems appropriate

% Initial guess for price path
% Normally my guess would be the following, but we saw in the stationary general eqm that p_eqm_init.r gives negative K with the final eqm taxes.
% PricePath0.r=[linspace(p_eqm_init.r,p_eqm_final.r,ceil(T/3)),p_eqm_final.r*ones(1,T-ceil(T/3))];
% So instead I try out
PricePath0.r=[linspace(0.8*p_eqm_final.r,p_eqm_final.r,ceil(T/3)),p_eqm_final.r*ones(1,T-ceil(T/3))];
PricePath0.iota=[linspace(p_eqm_init.iota,p_eqm_final.iota,ceil(T/3)),p_eqm_final.iota*ones(1,T-ceil(T/3))];
% PricePath0.iota=[linspace(p_eqm_init.iota,0.8*p_eqm_final.iota,ceil(T/3)),0.8*p_eqm_final.iota*ones(1,T-ceil(T/3)-1),p_eqm_final.iota]; % deliberately start without enough iota

% Parameter path is trivial, as they are preannounced one-off reforms
ParamPath.tau=Params.tau*ones(1,T);
ParamPath.xi=Params.xi*ones(1,T);
ParamPath.tau_a=Params.tau_a*ones(1,T);
ParamPath.xi_a=Params.xi_a*ones(1,T);
% B is constant, so can skip putting path on it as long as value in Params is correct one.

% intermediateEqns take Params and AggVars as inputs, output can be used in
% GeneralEqmEqns. They get evaluated in order, so can use output from one as an input to a later one.
transpathoptions.intermediateEqns.K=@(A,B) A-B; % physical capital K: A is asset supply, K and B are asset demand (some saving are used for gov debt, rest goes to physical capital)
transpathoptions.intermediateEqns.Y=@(K,L,alpha) (K^(alpha))*(L^(1-alpha)); % output Y

% Same as before, except that the government budget now has to be explicit that it is last period gov debt.
GeneralEqmEqns_TransPath.capitalmarket=GeneralEqmEqns.capitalmarket;
% GeneralEqmEqns_TransPath.govbudget=@(r,B,B_tminus1,G,TaxRevenue) (1+r)*B_tminus1+G-(B+TaxRevenue);
% Note: As discussed at start of this script BM2022 are presumably solving for a path where B is constant over time, but the commented out line above still writes out B_tminus1 so that it would still work for other setups.
GeneralEqmEqns_TransPath.govbudget=@(r,B,G,TaxRevenue) (1+r)*B+G-(B+TaxRevenue); % Take advantage of constant B

transpathoptions.GEnewprice=3;
% Need to explain to transpathoptions how to use the GeneralEqmEqns to update the general eqm transition prices (in PricePath).
transpathoptions.GEnewprice3.howtoupdate=... % a row is: GEcondn, price, add, factor
    {'capitalmarket','r',0,0.3;...  % captialMarket GE condition will be positive if r is too big, so subtract
    'govbudget','iota',0,0.2;... % govbudget GE condition will be positive if iota is too big, so subtract [iota is subtracted from the tax, so bigger iota means smaller TaxRevenue; note, units of iota are essentially the same as units of TaxRevenue, so don't need to think too hard about the size of the factor]
    };
% Note: the update is essentially new_price=price+factor*add*GEcondn_value-factor*(1-add)*GEcondn_value
% Notice that this adds factor*GEcondn_value when add=1 and subtracts it what add=0
% A small 'factor' will make the convergence to solution take longer, but too large a value will make it 
% unstable (fail to converge). Technically this is the damping factor in a shooting algorithm.


% For the transition path, turn on divide and conquer
vfoptions.divideandconquer=1;
vfoptions.level1n=25; % might be slightly faster or slower with a higher/lower value (do a tic-toc on ValueFnOnTransPath_Case1() to find the fastest if you want to find out what to set this to)

transpathoptions.tolerance=5*10^(-5); % This seems to be about how accurate we can get with n_d=101 (default would be 10^(-5), probably need a few more points to get this)
transpathoptions.verbose=1;
tic;
PricePath=TransitionPath_Case1(PricePath0, ParamPath, T, V_final, StationaryDist_init, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, transpathoptions, vfoptions, simoptions);
tpathtime=toc

[VPath,PolicyPath]=ValueFnOnTransPath_Case1(PricePath, ParamPath, T, V_final, Policy_final, Params, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);

AgentDistPath=AgentDistOnTransPath_Case1(StationaryDist_init, PolicyPath,n_d,n_a,n_z,pi_z,T,simoptions);

AggVarsPath=EvalFnOnTransPath_AggVars_Case1(FnsToEvaluate2,AgentDistPath,PolicyPath,PricePath,ParamPath, Params, T, n_d, n_a, n_z, d_grid, a_grid,z_grid,simoptions);
AllStatsPath=EvalFnOnTransPath_AllStats_InfHorz(FnsToEvaluate2,AgentDistPath,PolicyPath,PricePath,ParamPath, Params, T, n_d, n_a, n_z, d_grid, a_grid,z_grid,simoptions);

K_path=AggVarsPath.A.Mean-Params.B;
wagepath=((1-Params.alpha)*((PricePath.r+Params.delta)/Params.alpha).^(Params.alpha/(Params.alpha-1)));
outputpath=(K_path.^(Params.alpha)).*(AggVarsPath.L.Mean.^(1-Params.alpha));

C_t=AggVarsPath.Consumption.Mean;
C_tplus1=[AggVarsPath.Consumption.Mean(2:end),AggVarsPath.Consumption.Mean(end)].^(-Params.theta); % C_t+1 becomes constant in period T, so I just duplicate this
laborwedgepath=wagepath.*(C_t.^(-Params.theta))./(AggVarsPath.L.Mean.^Params.gamma); % Rearrange eqn at bottom of page 80 for varthetabar
savingswedgepath=Params.beta*(1+Params.r)*(C_t.^(-Params.theta))./(C_tplus1.^(-Params.theta)); % Rearrange eqn (4) on pg 81 to get zetabar_t, the aggregate capital wedge 

save BM2022.mat

figure_c=figure_c+1;
figure(figure_c);
subplot(2,4,1); plot(0:1:T, [laborwedge_init,laborwedgepath]) 
title('Labor wedge')
subplot(2,4,2); plot(0:1:T, [savingswedge_init,savingswedgepath]) 
title('Savings wedge')
subplot(2,4,3); plot(0:1:T, [output_init,outputpath]) 
title('Output')
subplot(2,4,4); plot(0:1:T, [K_init,K_path]) 
title('Capital')
subplot(2,4,5); plot(0:1:T, [AggVars_init.L.Mean,AggVarsPath.L.Mean]) 
title('Labor')
subplot(2,4,6); plot(0:1:T, [p_eqm_init.r,PricePath.r]) 
title('Interest rate')
subplot(2,4,7); plot(0:1:T, [wage_init,wagepath])
title('Wage')
subplot(2,4,8); plot(0:1:T,[AllStats_init.A.Gini,AllStatsPath.A.Gini])
title('Gini wealth')


%% Welfare calculations
% Paper doesn't seem to specify which period welfare is defined in, but presumably 
% the period in which the reform is revealed. In which case we already have V, 
% it is the first time period in VPath, and we already have the agent
% distribution, it is StationaryDist_init (which is also the first time
% period in StationaryDistPath)

% Social welfare preferences
Params.Delta=1; 
  % 0: average welfare
  % 1: utilitarian
  % Inf: Rawlsian

% Eqn on page 83 defining omega, with infinite sum algebra (as omega is
% constant and beta is less than 1) gives us
omegamod=VPath(:,:,1)*(1-Params.beta);
if Params.theta==1
    omega=exp(omegamod); % as (omega^(1-theta))/(1-theta) becomes ln(omega)
else
    omega=(omegamod*(1-Params.theta))^(1/(1-Params.theta));
end

% Evaluate the social welfare function (from pg 83)
if ~isfinite(Params.Delta) % Rawlsian
    % Not clear how they actually did Rawlsian, taking the min seems extreme but 
    % they do not specify a value of Delta, only say limit as Delta goes to infinity
    temp=omega(StationaryDist_init>0);
    SocialWelfare=min(temp);
else
    temp=(omega.^(1-Params.Delta)).*StationaryDist_init;
    temp(isnan(temp))=0; % in case there are any -Inf*0, which would give nan
    SocialWelfare=sum(sum(temp)).^(1/(1-Params.Delta));
end

% Repeat the welfare calculation, but now for the initial welfare level in
% the initial stationary general eqm
% Eqn on page 83 defining omega, with infinite sum algebra (as omega is
% constant and beta is less than 1) gives us
omegamod0=V_init*(1-Params.beta);
if Params.theta==1
    omega0=exp(omegamod0); % as (omega^(1-theta))/(1-theta) becomes ln(omega)
else
    omega0=(omegamod0*(1-Params.theta))^(1/(1-Params.theta));
end
% Evaluate the social welfare function (from pg 83)
if ~isfinite(Params.Delta) % Rawlsian
    % Not clear how they actually did Rawlsian, taking the min seems extreme but 
    % they do not specify a value of Delta, only say limit as Delta goes to infinity
    temp=omega0(StationaryDist_init>0);
    SocialWelfare0=min(temp);
else
    temp=(omega0.^(1-Params.Delta)).*StationaryDist_init;
    temp(isnan(temp))=0; % in case there are any -Inf*0, which would give nan
    SocialWelfare0=sum(sum(temp)).^(1/(1-Params.Delta));
end

%% Plot some other things from paper. First, the tax functions like in Figure 4

income_grid=linspace(0,10,1000);
marginalincometaxrate=1-(1-Params.tau)*(income_grid.^(-Params.xi)); % derivative of tax fn on pg 80 w.r.t. income
averageincometaxrate=(income_grid-(1-Params.tau)*((income_grid.^(1-Params.xi))/(1-Params.xi))-Params.iota)./income_grid; % taxes paid divided by income
wealth_grid=linspace(0,10,1000);
marginalwealthtaxrate=1-(1-Params.tau_a)*(wealth_grid.^(-Params.xi_a)); % derivative of tax fn on pg 80 w.r.t. income
averagewealthtaxrate=(wealth_grid-(1-Params.tau_a)*((wealth_grid.^(1-Params.xi_a))/(1-Params.xi_a)))./wealth_grid; % taxes paid divided by income

figure_c=figure_c+1;
figure(figure_c);
yyaxis left
subplot(1,2,1); plot(income_grid,averageincometaxrate,income_grid,marginalincometaxrate)
hold on
yyaxis right
% subplot(1,2,1); plot(income_grid,income_cdf,'.')
ylim([0,1])
hold off
xlabel('income')
ylabel('%')
legend('average income tax', 'marginal income tax','initial cdf of income')
yyaxis left
subplot(1,2,2); plot(wealth_grid,averagewealthtaxrate,wealth_grid,marginalwealthtaxrate)
hold on
yyaxis right
subplot(1,2,2); plot(a_grid,asset_cdf,'.')
ylim([0,1])
hold off
xlabel('wealth')
ylabel('%')
legend('average wealth tax', 'marginal wealth tax','initial cdf of wealth')
% I modified this figure so it also shows the pdf of agent distribution
% Otherwise you cannot tell where anyone actually is

%% Calculate welfare for the terciles of welfare distribution (like what gets plotted in Fig 2)


%% Calculate tax revenue for the terciles of the ??? paper calls them 'richest third', but is it wealth or income, or different for top and bottom panels?? (like what gets plotted in Fig 5)



