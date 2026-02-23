import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


price_lng = np.array([
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
    25.20])*0.137381

years = np.linspace(2025,2050,len(price_lng))

plt.scatter(years[7:], price_lng[7:])
plt.show()


m, b  = np.polyfit(years[7:],price_lng[7:],1)

p_years = np.linspace(2051,2085,35)
price_preds = p_years*m+b

plt.scatter(years,price_lng,label='EIA Predicitons')
plt.scatter(p_years, price_preds,label='Linear Projection')
#plt.title('New Cost of a Gallon of Diesel')
#plt.legend(fontsize=18)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#plt.tick_params(labelsize=16)
#plt.xlabel('Years', fontsize=18)
#plt.ylabel('Cost of disel ($/gal)', fontsize=18)
plt.show()


plt.scatter(years,price_lng,label='EIA Predicitons')
plt.scatter(p_years, np.ones(len(p_years))*25.10*0.137381,label='Flat Projection')
plt.title('Previous Cost of a Gallon of Diesel')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Cost of disel ($/gal)')
plt.show()


excel_data = {'Year': np.hstack((years, p_years)),
              'Price $/gal': np.hstack((price_lng, price_preds))}
df = pd.DataFrame(excel_data)


df.to_csv(f"Price_Projections.csv")









