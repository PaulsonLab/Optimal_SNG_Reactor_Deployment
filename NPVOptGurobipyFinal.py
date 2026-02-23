"""
Created on Oct 10 2025

@author: nmassa@wisc.edu (Nathaniel Massa UW Madison)
"""

import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt

#Initialization Parameters
Case = 3
years = 60
quarters = years*4
wells_per_reactor = 8
#price = 4
e = False
#Yearly lng prices: https://www.eia.gov/outlooks/aeo/data/browser/#/?id=6-AEO2025&region=0-0&cases=ref2025&start=2023&end=2050&f=A&linechart=ref2025-d032025a.4-6-AEO2025~ref2025-d032025a.15-6-AEO2025&ctype=linechart&sourcekey=0
#This is in $/MMBTU
price_lng = [
    24.60,
    23.34,
    22.83,
    22.50,
    22.22,
    21.97,
    22.01,
    21.90,
    22.23,
    22.36,
    22.61,
    22.72,
    22.90,
    23.01,
    23.15,
    23.37,
    23.61,
    23.71,
    24.05,
    24.25,
    24.32,
    24.57,
    25.09,
    25.10,
    25.09,
    25.20,
    #Data runs out so we will approx with a linear regression from 2032 to 2050
] + [25.52894737, 25.71742105, 25.90589474, 26.09436842, 26.28284211, 26.47131579,
    26.65978947, 26.84826316, 27.03673684, 27.22521053, 27.41368421, 27.60215789,
    27.79063158, 27.97910526, 28.16757895, 28.35605263, 28.54452632, 28.733,
    28.92147368, 29.10994737, 29.29842105, 29.48689474, 29.67536842, 29.86384211,
    30.05231579, 30.24078947, 30.42926316, 30.61773684, 30.80621053, 30.99468421,
    31.18315789, 31.37163158, 31.56010526, 31.74857895, 31.93705263]


x_quarters = np.linspace(0,(quarters-1)/4,quarters)
x_p = np.arange(years)
liquid_price = np.interp(x_quarters,x_p,price_lng[:int(years)])*0.137381 #MMBTU/GAL

flowrate = [100,200,400]
GAL_LF_per_MCF_SNG = []
MR_Cap_Cost = []
D_Cap_Cost = []
Total_Cap_Cost = []
F_Op_Cost = []
V_Op_Cost = []
MT_CO2_per_MCF_SNG = []
F_Op_Cost_Flares = 151991.15
V_Op_Cost_Flares = 4.805 #$/mcf burnt

#|||||||||||||||||||||||||| PLACEMENTS IN EXCEL ACROSS CASES ARE NOT CONSISTENT |||||||||||||||||
for i in range(len(flowrate)):
    if e:
        path = f'Final Economics calculations and ASPEN for all cases/Case {Case}/Case {Case} - {flowrate[i]} mcfd finalized ASPEN and costing/DRM_techno_economic_Case_{Case}_{flowrate[i]} mcfd.xlsx'
    else:
        path = f'Final Economics calculations and ASPEN for all cases/Case {Case}/Case {Case} - {flowrate[i]} mcfd finalized ASPEN and costing/DRM_techno_economic_Case_{Case}_{flowrate[i]} mcfd_without electricity.xlsx'
    excel = pd.ExcelFile(path)
    
    costing = pd.read_excel(excel, 'Costing corrected')
    
    #LNG
    GAL_LF_per_MCF_SNG.append(costing['Reference capacity value'][75])
        
    #CAPITAL COST
    MR_Cap_Cost.append(costing['Total Plant Cost (x $1000)'][9]*1000) #Modular Reactor
    D_Cap_Cost.append(costing['Total Plant Cost (x $1000)'][19]*1000+1942000) #Downstream Including 2root(10) + 2root(2) piping to transport SNG from 3x1mi plot
    Total_Cap_Cost.append(MR_Cap_Cost+D_Cap_Cost) #Total
    
    #OPERATING COST
    F_Op_Cost.append(costing['Units.1'][40]) #Fixed
    V_Op_Cost.append((costing['Units.1'][50]  + costing['Units.1'][53] + costing['Units.1'][57] + costing['Units.1'][65])/(flowrate[i]*90)) #Varaible (Consumables + Waste Disposal + Byproducts + Feedstock)

    #Carbon Tax
    MT_CO2_per_MCF_SNG.append(costing['Reference capacity value'][73]) #Metric Ton of CO2 produced for Thousand Cubic Feet of Standard Natural Gas

def calc_CO2_tax(quarter):
    initial = 36.79
    year = np.floor(quarter/4) #Might relax that step fxn if too difficult
    return initial + 3.5*year

def discounted_cash_flow(t):
    discount_rate = 0.028737 # quarterly rate
    return 1 / ((1 + discount_rate) ** (t))

def well_production(t):
    initial = 9000*wells_per_reactor #We need to sum this monthly production to yearly
    prod = sum(initial/((1+1.4*0.197*tau)**(1/1.4)) for tau in range(3*t,3*t+3))
    return prod


def NPV_t(m,t):
    
    if t<120:
        depreciation = (
            gp.quicksum(MR_Cap_Cost[j] / 120 * inv[t,j] for j in J) #Reactor depriciation
            +D_Cap_Cost[0] / 120 * gp.quicksum(well_open[tau,w] for w in W for tau in range(t+1)) #Downstream depreciation
            +7801450 / 120 * gp.quicksum(well_open[tau,w] for w in W for tau in range(t+1))*wells_per_reactor #Flare depreciation
        )
    else:
        depreciation = (
            gp.quicksum(MR_Cap_Cost[j] / 120 * inv[t,j] for j in J) #Reactor depriciation
            +D_Cap_Cost[0] / 120 * gp.quicksum(well_open[tau,w] for w in W for tau in range(t-120,t+1)) #Downstream depreciation
            +7801450 / 120 * gp.quicksum(well_open[tau,w] for w in W for tau in range(t-120+1,t+1))*wells_per_reactor #Flare depreciation
        )

    PBT = (
        liquid_price[t]*gp.quicksum(GAL_LF_per_MCF_SNG[j]*x[t,w,j] for w in W for j in J) #Revenue from SNG
        -gp.quicksum((F_Op_Cost[j])*y[t,w,j] + V_Op_Cost[j]*x[t,w,j] for w in W for j in J) #Operating cost of refineries
        -calc_CO2_tax(t)*gp.quicksum(MT_CO2_per_MCF_SNG[j]*x[t,w,j] for w in W for j in J) #co2 tax from operating refineries

        -gp.quicksum(F_Op_Cost_Flares*well_open[tau,w] for w in W for tau in range(t+1)) * wells_per_reactor #Fixed operating cost of flares
        -(calc_CO2_tax(t)*0.0619+V_Op_Cost_Flares)*gp.quicksum(prod[t,w]-gp.quicksum(x[t,w,j] for j in J) for w in W) #c02 tax cost of flaring the unused gas

        -depreciation
    )

    m.addConstr(income_tax[t] >= PBT*.21)
    
    net_cash = (
        PBT
        -gp.quicksum(MR_Cap_Cost[j]*buy[t,j] for j in J) #Captial cost for refineries
        -gp.quicksum(MR_Cap_Cost[j]*0.01*(y[t,w,j]-delta[t,w,j]) for w in W for j in J) #Moving cost of refineries
        -D_Cap_Cost[0]*gp.quicksum(well_open[t,w] for w in W) #Capital cost of downstream equipment which is left at wells
        -7801450*gp.quicksum(well_open[t,w] for w in W)*wells_per_reactor #Capital cost of flares
        -income_tax[t]
        +depreciation
        )
    return discounted_cash_flow(t)*net_cash


m = gp.Model('Standard Natural Gas Deployment')
m.setParam("Threads", 32)
m.setParam('LogFile', 'gurobi_log.txt')

T = range(quarters) #number of quarters
W = range(int(years/2)) #number of wells
J = range(len(flowrate)) #plant sizes

decay_profile = ([0] + [well_production(t) for t in T])
well_open = np.zeros((quarters,int(years/2)))
for t in T:
    for w in W:
        if w*4 == t:
            well_open[t,w]=1


#Decision Variables
income_tax = m.addVars(T, vtype=gp.GRB.CONTINUOUS, name='tax')
#well_open = m.addVars(T,W, vtype='B', name='open')
y = m.addVars(T,W,J, vtype='B', name='deploy') #Decides to allocate plant type j to source w at time t
#z ||| We are assuming a gas gathering system is present at all open wells
delta = m.addVars(T,W,J, vtype='B', name='delta') #delta tracks when plants move. 0 when they move 1 when they dont
x = m.addVars(T,W,J, vtype=gp.GRB.CONTINUOUS, name='feed') #Gas feed to plant j at source i at time t
#sold = m.addVars(T, vtype=gp.GRB.CONTINUOUS, name='disel_sold')
buy = m.addVars(T,J, vtype=gp.GRB.INTEGER, name='buy') #Number of plant of type j purchased at time t
#sell ||| We assume we sell all plants at the end of 30 years
inv = m.addVars(T,J, vtype=gp.GRB.INTEGER, name='inventory')
prod = m.addVars(T,W, vtype=gp.GRB.CONTINUOUS, name='prod', lb=0)



#Objective
m.setObjective(gp.quicksum(NPV_t(m,t) for t in T), gp.GRB.MAXIMIZE)

#Constraints
for t in T:
    for j in J:
        
        #Inventory constraints, keep track of how many of each size reactor we have in inventory or deployed
        if t == 0:
            m.addConstr(inv[t, j] == buy[t, j])
        elif t < 120:
            m.addConstr(inv[t, j] == inv[t - 1, j] + buy[t, j])
        else:
            m.addConstr(inv[t,j] == inv[t-1,j] + buy[t,j] - buy[t-120,j])

        #Makes sure that we do not deploy more reactors than we have
        m.addConstr(
            gp.quicksum(y[t,w,j] for w in W) <= inv[t,j]
            )

#One plant per well
for t in T:
    for w in W:
        m.addConstr(
            gp.quicksum(y[t,w,j] for j in J) <= 1
            )

#One well opens up every year, we cannot have more deployed reactors than wells
for t in T:
    for w in W:
        m.addConstr(
            gp.quicksum(y[t,w,j] for j in J) <= gp.quicksum(well_open[tau,w] for tau in range(t+1))
            )


for t in T:
    for w in W:
        m.addConstr(
            prod[t, w] ==
            gp.quicksum(
                decay_profile[tau] * well_open[t - tau, w]
                for tau in range(t + 1)
            )
        )

        for j in J:
            #Well production to the plant must be less than or equal to what the source can provide
            m.addConstr(
                x[t,w,j] <= prod[t,w]
            )
            #Gas flow to plant cannot exceed plant capacity 
            m.addConstr(
                x[t,w,j] <= flowrate[j]*90*y[t,w,j]*delta[t,w, j]
            )
            #Gas flow to plant must be greater than 50% plant capacity
            m.addConstr(
                x[t,w,j] >= flowrate[j]*90*.5*y[t,w,j]*delta[t,w, j]
            )

#Delta tracks when plants move, Delta is 0 when plants move and 1 when they dont
for t in range(quarters):
    for w in W:
        for j in J:
            if t == 0:
                # Initial condition: y[-1, w, j] = 0 for all W,J
                m.addConstr(delta[t, w, j] == 0)
            elif t==quarters-1:
                m.addConstr(delta[t,w,j] == 0)
            else:
                # General case: delta[t, w] = 1 iff y[t, w, j] == y[t-1, w, j], otherwise delta=0
                m.addConstr(delta[t,w,j] <= y[t-1,w,j])
                m.addConstr(delta[t,w,j] <= y[t,w,j])
                m.addConstr(delta[t,w,j] <= y[t+1,w,j])
                m.addConstr(delta[t,w,j] >= y[t-1,w,j] + y[t,w,j] + y[t+1,w,j] - 2)


m.optimize()

print(f"Optimal objective value: {m.objVal}")

#///////////////////// MODEL END ////////////////////////////////

# Convert variables to numpy arrays
inv_np = np.array([[inv[t, j].X for j in J] for t in T])
buy_np = np.array([[buy[t, j].X for j in J] for t in T])
x_np = np.array([[[x[t, w, j].X for j in J] for w in W] for t in T])
y_np = np.array([[[y[t, w, j].X for j in J] for w in W] for t in T])
delta_np = np.array([[[delta[t, w, j].X for j in J] for w in W] for t in T])
prod_np = np.array([[prod[t, w].X for w in W] for t in T])
obj_val = m.objVal
income_tax_np = np.array([income_tax[t].X for t in T])


m.dispose()
del m

# Plot inventory over time
for j in J:
    plt.plot(T, inv_np[:, j], label=f"Flowrate {flowrate[j]}")
plt.xlabel("Time (quarters)")
plt.ylabel("Inventory")
plt.legend()
plt.show()

# Gas flow to reactors per well over time
x_values = np.sum(x_np, axis=2)  # shape: (T, W)

plt.figure(figsize=(12, 6))
for w in W:
    plt.plot(T, x_values[:, w], label=f"Well {w}")
plt.xlabel("Time (quarters)")
plt.ylabel("Gas Flow to Reactors (MCF)")
plt.title("Gas Production Routed to Reactors per Well Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Production per well over time
plt.figure(figsize=(12, 6))
for w in W:
    plt.plot(T, prod_np[:, w], label=f"Well {w}")
plt.xlabel("Time (quarters)")
plt.ylabel("Available Production (MCF)")
plt.title("Available Gas Production per Well Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# NPV calculation over time
npv_by_time = []
npv = []
co2_total = 0
for t in T:
    #Depreciation
    depreciation = (
        sum(MR_Cap_Cost[j] / 120 * inv_np[t,j] for j in J) #Reactor depriciation
        +D_Cap_Cost[0] / 120 * sum(well_open[tau,w] for w in W for tau in range(t)) #Downstream depreciation
        +7801450 / 120 * sum(well_open[tau,w] for w in W for tau in range(t))*wells_per_reactor #Flare depreciation
    )

    PBT = (
        liquid_price[t]*sum(GAL_LF_per_MCF_SNG[j]*x_np[t,w,j] for w in W for j in J) #Revenue from SNG

        -sum((F_Op_Cost[j])*y_np[t,w,j] + V_Op_Cost[j]*x_np[t,w,j] for w in W for j in J) #Operating cost of refineries
        -calc_CO2_tax(t)*sum(MT_CO2_per_MCF_SNG[j]*x_np[t,w,j] for w in W for j in J) #co2 tax from operating refineries

        -sum(F_Op_Cost_Flares*well_open[tau,w] for w in W for tau in range(t)) #operating cost of flares
        -(calc_CO2_tax(t)*0.0619+V_Op_Cost_Flares)*sum(prod_np[t,w]-sum(x_np[t,w,j] for j in J) for w in W) #c02 tax cost of flaring the unused gas

        -depreciation
    )
    
    net_cash = (
        PBT
        -sum(MR_Cap_Cost[j]*buy_np[t,j] for j in J) #Captial cost for refineries
        -sum(MR_Cap_Cost[j]*0.01*(y_np[t,w,j]-delta_np[t,w,j]) for w in W for j in J) #Moving cost of refineries
        -D_Cap_Cost[0]*sum(well_open[t,w] for w in W) #Capital cost of downstream equipment which is left at wells
        -7801450*sum(well_open[t,w] for w in W)*wells_per_reactor #Capital cost of flares
        -income_tax_np[t]
        +depreciation
        )
    
    co2_total += 0.0619*sum(prod_np[t,w]-sum(x_np[t,w,j] for j in J) for w in W) + sum(MT_CO2_per_MCF_SNG[j]*x_np[t,w,j] for w in W for j in J)
    
    npv_by_time.append(discounted_cash_flow(t) * net_cash)

    if t == 0:
        npv.append(npv_by_time[t])
    else:
        npv.append(npv[-1] + npv_by_time[t])

# Plot NPV over time
plt.figure(figsize=(12, 6))
plt.plot(T, npv_by_time, marker='o', label='Per Timestep')
plt.plot(T, npv, marker='o', label='Sum Total')
plt.xlabel("Time (quarters)")
plt.ylabel("Net Present Value")
plt.title("Objective Contribution Over Time")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Reactor colors (1 for each type)
colors = ['red', 'blue', 'green', 'orange']  # 4 reactor types

fig, ax = plt.subplots()

rect_width = 0.1   # horizontal size of rectangle
rect_height = 0.8  # vertical size of rectangle

for j in J:  # reactor size
    for t in T:
        for w in W:
            if y_np[t, w, j] > 0.5:  # deployed
                # bottom-left corner = (t - rect_width/2, w - rect_height/2)
                rect = Rectangle(
                    (t - rect_width / 2, w - rect_height / 2),
                    rect_width,
                    rect_height,
                    color=colors[j],
                    label=f'Size {flowrate[j]}'
                )
                ax.add_patch(rect)

# Remove duplicate legends
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

ax.set_xlim(-4, quarters)
ax.set_ylim(-1, int(years) / 2)
ax.set_xlabel("Time (quarters)")
ax.set_ylabel("Well index")
ax.set_title("Reactor Deployments Over Time by Well and Size")
ax.grid(True)
plt.tight_layout()
#plt.savefig(f'Figs/ReactorDeployment_{quarters}q_case{Case}_marketprice_wpr{wells_per_reactor}_we',dpi=500)
plt.show()


# Save all arrays to a single .npz file
if e:
    sp = f"gurobi_results_{quarters}q_case{Case}_marketprice_wpr{wells_per_reactor}.npz"
else:
    sp = f"gurobi_results_{quarters}q_case{Case}_marketprice_wpr{wells_per_reactor}_we.npz"
np.savez(sp,
         inv=inv_np,
         buy=buy_np,
         x=x_np,
         y=y_np,
         delta=delta_np,
         well_open=well_open,
         prod=prod_np,
         obj=obj_val,
         income_tax=income_tax_np,
         liquid_price=liquid_price,
         wells_per_reactor=wells_per_reactor,
         years=years,
         npv_by_time=npv_by_time,
         co2_total=co2_total
         )




yrs_flare = 0
yrs_sold = 0
yrs_depreciation = 0
yrs_opcost = 0
yrs_co2 = 0
for t in T:
    if t < 120:
        depreciation = (
            sum(MR_Cap_Cost[j] /120 * inv_np[t,j] for j in J) #Reactor depriciation
            +D_Cap_Cost[0] / 120 * sum(well_open[tau,w] for w in W for tau in range(0,t)) #Downstream depreciation
        )
    else:
        depreciation = (
            sum(MR_Cap_Cost[j] / 120 * inv_np[t,j] - inv_np[t-120,j] for j in J) #Reactor depriciation
            +D_Cap_Cost[0] / 120 * sum(well_open[tau,w] for w in W for tau in range(t-120,t)) #Downstream depreciation
        )
    yrs_depreciation += discounted_cash_flow(t)*depreciation

    yrs_flare += discounted_cash_flow(t)*calc_CO2_tax(t)*0.0619*sum(prod_np[t,w]-sum(x_np[t,w,j] for j in J) for w in W)

    yrs_sold += discounted_cash_flow(t)*liquid_price[t] * sum([GAL_LF_per_MCF_SNG[j] * x_np[t, w, j] for w in W for j in J])

    yrs_opcost += discounted_cash_flow(t)*sum([F_Op_Cost[j] * y_np[t, w, j] + V_Op_Cost[j] * x_np[t, w, j] for w in W for j in J])

    yrs_co2 += discounted_cash_flow(t)*calc_CO2_tax(t) * sum([MT_CO2_per_MCF_SNG[j] * x_np[t, w, j] for w in W for j in J])

    if (t+1)%30 == 0:
        print(f'Flare CO2 tax: {-yrs_flare:.4e}')
        print(f'LNG Sold: {yrs_sold:.4e}')
        print(f'Depreciation: {yrs_depreciation:.4e}')
        print(f'Op_cost: {-yrs_opcost:.4e}')
        print(f'Reactor CO2 tax: {-yrs_co2:.4e}')
        print(f'|||||||TOTAL|||||||| {(yrs_sold+yrs_depreciation-yrs_flare-yrs_opcost-yrs_co2):.4e}')
        print()
        yrs_co2 = 0
        yrs_opcost = 0
        yrs_depreciation = 0
        yrs_sold = 0
        yrs_flare = 0

