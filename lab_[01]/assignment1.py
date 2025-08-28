import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\SML\\lab_[01]\\ecommerce_data.csv")
print(df.head())

plt.figure(figsize=(8, 5))
plt.plot(df['Total Profit'], df['Month'], marker='o', color='b')
plt.title("Total Profit Month-wise (E-commerce Website)")
plt.xlabel("Total Profit")
plt.ylabel("Month")
plt.grid(True)
plt.savefig("total_profit_monthwise.png", dpi=200)  
plt.show()

#4.	Read T-Shirts data of each month and show it using a scatter plot. Add following style properties in the graph:
plt.figure(figsize=(8, 5))
plt.scatter(df['T-Shirts'], df['Month'], s=50, alpha=0.5, edgecolors='red')
plt.title("T-Shirts Sold Month-wise (E-commerce Website)")
plt.xlabel("T-Shirts sold")
plt.ylabel('Month')
plt.legend(loc='upper right')
plt.grid(True, linewidth=1, linestyle='--')
plt.savefig("tshirts_sales.png", dpi=200)
plt.show()


# 5. Read trousers and shirts product data and show it using a bar chart
plt.figure(figsize=(8, 5))

# Bar positions
x = df['Month']  
plt.bar(x, df['Trousers'], width=0.4, label='Trousers', color='skyblue')
plt.bar(x, df['Shirts'], width=0.4, label='Shirts', color='orange', bottom=None)
plt.xlabel("Month")
plt.ylabel("Number of Products Sold")

plt.legend(loc='upper right')
plt.grid(True, linewidth=1, linestyle='--')
plt.title("Trousers and Shirts Sales Month-wise")
plt.savefig("trousers_shirts_sales.png", dpi=200)
plt.show()
