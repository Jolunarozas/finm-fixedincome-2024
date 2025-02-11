from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew,kurtosis,norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import interpolate



### Bootstrapping ###

# RESTRICT_YLD = True
# RESTRICT_TIPS = True

# RESTRICT_DTS_MATURING = True
# RESTRICT_REDUNDANT = True

# data = filter_treasuries(quotes, t_date=t_current, filter_yld = RESTRICT_YLD, filter_tips = RESTRICT_TIPS, drop_duplicate_maturities=RESTRICT_REDUNDANT)
# CF = filter_treasury_cashflows(calc_cashflows(data),filter_maturity_dates=RESTRICT_DTS_MATURING)


### OLS Regression ###

# RESTRICT_YLD = True
# RESTRICT_TIPS = True

# RESTRICT_DTS_MATURING = True
# RESTRICT_REDUNDANT = False

# data = filter_treasuries(quotes, t_date=t_current, filter_yld = RESTRICT_YLD, filter_tips = RESTRICT_TIPS, drop_duplicate_maturities=RESTRICT_REDUNDANT)
# CF = filter_treasury_cashflows(calc_cashflows(data),filter_maturity_dates=RESTRICT_DTS_MATURING)


## Nelson Siegel Yield Curve Model ## 

# RESTRICT_YLD = True
# RESTRICT_TIPS = True

# RESTRICT_DTS_MATURING = False
# RESTRICT_REDUNDANT = False

# data = filter_treasuries(quotes, t_date=t_current, filter_yld = RESTRICT_YLD, filter_tips = RESTRICT_TIPS, drop_duplicate_maturities=RESTRICT_REDUNDANT)
# CF = filter_treasury_cashflows(calc_cashflows(data),filter_maturity_dates=RESTRICT_DTS_MATURING)




def get_coupon_dates(quote_date, maturity_date):

    if isinstance(quote_date, str):
        quote_date = datetime.strptime(quote_date, '%Y-%m-%d')
    if isinstance(maturity_date, str):
        maturity_date = datetime.strptime(maturity_date, '%Y-%m-%d')
    
    if quote_date >= maturity_date:
        raise ValueError("Quote date must be earlier than maturity date.")

    semiannual_offset = pd.DateOffset(months=6)
    dates = []
    current_date = maturity_date

    while current_date > quote_date:
        dates.append(current_date)
        current_date -= semiannual_offset

    return sorted(dates)
def calc_cashflows(data, adj_end_month = False):

    columns = ["CALDT", "TMATDT", "ITYPE", "TDYLD"]
    quote_data = data.copy()
    
    if quote_data.columns.isin(columns).sum() != len(columns):
        quote_data.rename(columns={"quote date":"CALDT", "maturity date":"TMATDT", "type":"ITYPE", "ytm":"TDYLD", "cpn rate": "TCOUPRT"}, inplace=True) 
    
    CF = pd.DataFrame(data=0.0, index=quote_data.index, columns=quote_data['TMATDT'].unique(), dtype=float)


    for i in quote_data.index:
        coupon_dates = get_coupon_dates(quote_data.loc[i,'CALDT'],quote_data.loc[i,'TMATDT'])

        if coupon_dates is not None:
            CF.loc[i,coupon_dates] = quote_data.loc[i,'TCOUPRT']/2

        CF.loc[i,quote_data.loc[i,'TMATDT']] += 100

    if adj_end_month:
        CF = CF.T.resample('ME').sum().T
        CF.drop(columns=CF.columns[(CF==0).all()],inplace=True)
        CF = CF.fillna(0).sort_index(axis=1)

    else:
        CF = CF.fillna(0).sort_index(axis=1)
        CF.drop(columns=CF.columns[(CF==0).all()],inplace=True)

        
    return CF


def discount_to_intrate(discount, maturity, n_compound=None):
        
    if n_compound is None:
        intrate = - np.log(discount) / maturity
    
    else:
        intrate = n_compound * (1/discount**(1/(n_compound * maturity)) - 1)    
        
    return intrate

def intrate_to_discount(intrate, maturity, n_compound=None):
    
    if n_compound is None:
        discount = np.exp(-intrate * maturity)
    else:
        discount = 1 / (1+(intrate / n_compound))**(n_compound * maturity)

    return discount   

def compound_rate(intrate,compound_input,compound_output):    
    if compound_input is None:
        outrate = compound_output * (np.exp(intrate/compound_output) - 1)
    elif compound_output is None:
        outrate = compound_input * np.log(1 + intrate/compound_input)
    else:
        outrate = ((1 + intrate/compound_input) ** (compound_input/compound_output) - 1) * compound_output

    return outrate

def filter_treasury_cashflows(CF, filter_maturity_dates=False, filter_benchmark_dates=False, filter_CF_strict=True):

    mask_benchmark_dts = []
    
    # Filter by using only benchmark treasury dates
    for col in CF.columns:
        if filter_benchmark_dates:
            if col.month in [2,5,8,11] and col.day == 15:
                mask_benchmark_dts.append(col)
        else:
            mask_benchmark_dts.append(col)
    
    if filter_maturity_dates:
        mask_maturity_dts = CF.columns[(CF>=100).any()]
    else:
        mask_maturity_dts = CF.columns
    
    mask = [i for i in mask_benchmark_dts if i in mask_maturity_dts]

    CF_filtered = CF[mask]
          
    if filter_CF_strict:
        # drop issues that had CF on excluded dates
        mask_bnds = CF_filtered.sum(axis=1) == CF.sum(axis=1)
        CF_filtered = CF_filtered[mask_bnds]

    else:
        # drop issues that have no CF on included dates
        mask_bnds = CF_filtered.sum(axis=1) > 0
        CF_filtered = CF_filtered[mask_bnds]
        
        
    # update to drop dates with no CF
    CF_filtered = CF_filtered.loc[:,(CF_filtered>0).any()]
    
    return CF_filtered
def filter_treasuries(data, t_date=None, filter_maturity = None, filter_maturity_min=None, drop_duplicate_maturities = False, filter_tips=True, filter_yld=True):
    
    columns = ["CALDT", "TMATDT", "ITYPE", "TDYLD"]
    
    if data.columns.isin(columns).sum() != len(columns):
        outdata = data.copy()
        outdata.rename(columns={"quote date":"CALDT", "maturity date":"TMATDT", "type":"ITYPE", "ytm":"TDYLD"}, inplace=True) 

        outdata = outdata[outdata['ITYPE'].str.contains('TIPS') == (not filter_tips)]
        reversed_name = True
    else:
        outdata = data.copy()
        outdata = outdata[outdata['ITYPE'].isin([11,12]) == (not filter_tips)]
        reversed_name = False


    if t_date is None:
        t_date = outdata['CALDT'].values[-1]
    
    outdata = outdata[outdata['CALDT'] == t_date]
    
    # Filter out redundant maturity
    if drop_duplicate_maturities:
        outdata = outdata.drop_duplicates(subset=['TMATDT'])
    
    # Filter by max maturity
    if filter_maturity is not None:
        mask_truncate = outdata['TMATDT'] < (t_date + np.timedelta64(365*filter_maturity+1,'D'))
        outdata = outdata[mask_truncate]

    # Filter by min maturity
    if filter_maturity_min is not None:
        mask_truncate = outdata['TMATDT'] > (t_date + np.timedelta64(365*filter_maturity_min-1,'D'))
        outdata = outdata[mask_truncate]

        
    if filter_yld:
        outdata = outdata[outdata['TDYLD']>0]
    
    if reversed_name:
        outdata.rename(columns={"CALDT":"quote date", "TMATDT":"maturity date", "ITYPE":"type", "TDYLD":"ytm"}, inplace=True)
    return outdata
import numpy as np
import pandas as pd
from scipy.optimize import minimize


import numpy as np

def discount_factor_nelson_siegel(params, maturity):
    """
    Calculate the discount factor using the Nelson-Siegel model.

    :param params: List of parameters [beta0, beta1, beta2, lambda]
    :param maturity: Time to maturity in years (scalar or array)
    :return: Discount factor (scalar or array)
    """
    beta0, beta1, beta2, lambd = params
    num = (1 - np.exp(-maturity / lambd))
    denom = np.where(np.isclose(maturity, 0), 1e-12, (maturity / lambd))
    
    yield_curve = beta0 + (beta1 + beta2) * (num / denom) - beta2 * np.exp(-maturity / lambd)
    return np.exp(-yield_curve * maturity)  # Convert yield to discount factor

def discount_factor_nelson_siegel_extended(params, maturity):
    """
    Calculate the discount factor using the Extended Nelson-Siegel model.

    :param params: List of parameters [beta0, beta1, beta2, lambda1, beta3, lambda2]
    :param maturity: Time to maturity in years (scalar or array)
    :return: Discount factor (scalar or array)
    """
    beta0, beta1, beta2, lambda1, beta3, lambda2 = params

    num1 = (1 - np.exp(-maturity / lambda1))
    den1 = np.where(np.isclose(maturity, 0), 1e-12, (maturity / lambda1))
    
    num2 = (1 - np.exp(-maturity / lambda2))
    den2 = np.where(np.isclose(maturity, 0), 1e-12, (maturity / lambda2))
    
    yield_curve = (beta0 
                   + (beta1 + beta2) * (num1 / den1) 
                   - beta2 * np.exp(-maturity / lambda1) 
                   + beta3 * ((num2 / den2) - np.exp(-maturity / lambda2)))
    
    return np.exp(-yield_curve * maturity)  # Convert yield to discount factor


def estimate_nelson_siegel_models(CF, t_current, prices, 
                                  x0_ns=None, x0_nse=None, 
                                  return_modeled_prices=True):

    col_dates = pd.to_datetime(CF.columns)
    t_val = pd.to_datetime(t_current)
    maturity = (col_dates - t_val).days / 365.25

    if isinstance(prices, pd.Series):
        y = prices.values
    else:
        y = np.array(prices)

    def nelson_siegel(p, m):
        # p = [beta0, beta1, beta2, lambda]
        num = (1 - np.exp(-m/p[3]))
        denom = np.where(np.isclose(m, 0), 1e-12, (m/p[3]))
        return p[0] + (p[1] + p[2])*(num/denom) - p[2]*np.exp(-m/p[3])

    def nelson_siegel_extended(p, m):
        # p = [beta0, beta1, beta2, lambda1, beta3, lambda2]
        num1 = (1 - np.exp(-m/p[3]))
        den1 = np.where(np.isclose(m, 0), 1e-12, (m/p[3]))
        num2 = (1 - np.exp(-m/p[5]))
        den2 = np.where(np.isclose(m, 0), 1e-12, (m/p[5]))
        return (p[0]
                + (p[1] + p[2])*(num1/den1)
                - p[2]*np.exp(-m/p[3])
                + p[4]*( (num2/den2) - np.exp(-m/p[5]) ))

    def model_price(params, model):
        # Continuous discount from the rates
        rates = model(params, maturity)
        discs = np.exp(-rates*maturity)
        return CF.values @ discs, pd.DataFrame(data=discs, index=maturity, columns=["Discount Factor"])

    def sse_loss(params, model):
        mp = model_price(params, model)[0]
        return np.sum((y - mp)**2)


    if x0_ns is None:
        # [beta0, beta1, beta2, lambda]
        x0_ns = np.ones(4)/10
    res_ns = minimize(sse_loss, x0_ns, args=(nelson_siegel,))
    ns_params = res_ns.x

    # Fit Extended NS with x0_nse = np.concatenate( (ns_params, [0.1, 0.1]) )
    if x0_nse is None:
        # [beta0, beta1, beta2, lambda1, beta3, lambda2]
        x0_nse = np.concatenate((ns_params,(1/10,1/10)))
    if x0_nse.size == 2:
        x0_nse = np.concatenate((ns_params,(x0_nse[0],x0_nse[1])))

    res_nse = minimize(sse_loss, x0_nse, args=(nelson_siegel_extended,))
    nse_params = res_nse.x


    df_params = pd.DataFrame(
        index=["Nelson-Siegel", "Nelson-Siegel Extended"],
        columns=["theta 0","theta 1","theta 2","lambda_1","theta_3","lambda_2"]
    )


    df_params.loc["Nelson-Siegel", ["theta 0","theta 1","theta 2","lambda_1"]] = ns_params
    df_params.loc["Nelson-Siegel", ["theta_3","lambda_2"]] = [None, None]

    df_params.loc["Nelson-Siegel Extended", :] = nse_params

    out = {"params": df_params}
    if return_modeled_prices:
        out["DF_ns"] = model_price(ns_params, nelson_siegel)[1]
        out["DF_nse"] = model_price(nse_params, nelson_siegel_extended)[1]

    return out

def calculate_hedge_position(agg_info, key_long, key_short, long_position=1000000, leverage=50, spread=0.01/100):


    # Calculate leverage and hedge ratio
    leverage_long = long_position * leverage
    hedge_ratio = -agg_info.loc["duration", key_long] / agg_info.loc["duration", key_short]
    hedge_ratio_dollar_duration = -(agg_info.loc["duration", key_long]*agg_info.loc["dirty price", key_long]) / (agg_info.loc["duration", key_short]*agg_info.loc["dirty price", key_short])


    # Initialize Hedge Position DataFrame
    hedge_position = pd.DataFrame(index=[key_long, key_short], columns=["Equity", "Assets", "Contracts"])
    hedge_position.loc[key_long, "Equity"] = long_position
    hedge_position.loc[key_short, "Equity"] = long_position * hedge_ratio
    hedge_position.loc[key_long, "Assets"] = leverage_long
    hedge_position.loc[key_short, "Assets"] = leverage_long * hedge_ratio
    hedge_position.loc[key_long, "Contracts"] = hedge_position.loc[key_long, "Assets"] / agg_info.loc["dirty price", key_long]
    hedge_position.loc[key_short, "Contracts"] = hedge_position.loc[key_short, "Assets"] / agg_info.loc["dirty price", key_short]
    hedge_position.loc[key_long, "Dollar_Duration"] =  agg_info.loc["dirty price", key_long] * agg_info.loc["duration", key_long]
    hedge_position.loc[key_short, "Dollar_Duration"] =  agg_info.loc["dirty price", key_short] * agg_info.loc["duration", key_short]
    hedge_position.loc[key_long, "dirty price"] = agg_info.loc["dirty price", key_long]
    hedge_position.loc[key_short, "dirty price"] = agg_info.loc["dirty price", key_short]


    # Create Hedge Position PnL DataFrame
    hedge_position_pnl = hedge_position.copy()
    hedge_position_pnl["Mod Duration"] = [
        agg_info.loc["modified duration", key_long],
        agg_info.loc["modified duration", key_short]
    ]
    hedge_position_pnl["YTM Change"] = spread / 2 * np.array([-1, 1])
    hedge_position_pnl["Dirty Price"] = [
        agg_info.loc["dirty price", key_long],
        agg_info.loc["dirty price", key_short]
    ]
    hedge_position_pnl["Change in Dirty Price"] = -hedge_position_pnl["Dirty Price"] * \
                                                   hedge_position_pnl["Mod Duration"] * \
                                                   hedge_position_pnl["YTM Change"]
    hedge_position_pnl["PnL"] = hedge_position_pnl["Change in Dirty Price"] * hedge_position_pnl["Contracts"]
    hedge_position_pnl.loc['total', 'PnL'] = hedge_position_pnl['PnL'].sum()
    hedge_position_pnl["Return"] = hedge_position_pnl["PnL"] / hedge_position_pnl["Equity"].abs().sum()

    return hedge_position_pnl, hedge_ratio_dollar_duration

def price_treasury_ytm(time_to_maturity, ytm, cpn_rate,freq=2,face=100):
    c = cpn_rate/freq
    y = ytm/freq
    
    rem = freq * (time_to_maturity % (1/freq))
    tau = freq * time_to_maturity - rem
    
    if round(tau)!=tau:
        print('warning')
    else:
        tau = round(tau)    
    
    pv = 0
    for i in range(1,tau):
        pv += 1 / (1+y)**i
    
    pv = c*pv + (1+c)/(1+y)**tau
    pv *= face
    
    if rem>0:
        pv += c*face
        pv /= (1+y)**rem
        
    return pv

def duration_closed_formula(tau, ytm, cpnrate=None, freq=2):

    if cpnrate is None:
        cpnrate = ytm
        
    y = ytm/freq
    c = cpnrate/freq
    T = tau * freq
        
    if cpnrate==ytm:
        duration = (1+y)/y  * (1 - 1/(1+y)**T)
        
    else:
        duration = (1+y)/y - (1+y+T*(c-y)) / (c*((1+y)**T-1)+y)

    duration /= freq
    
    return duration