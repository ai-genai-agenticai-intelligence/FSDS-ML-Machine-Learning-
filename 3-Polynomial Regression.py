import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# 1. Generate synthetic dataset
# -----------------------------
x = 6 * np.random.rand(200, 1) - 3
y = 0.8 * x**2 + x + 2 + np.random.randn(200, 1)

# Plot raw data
plt.scatter(x, y, color='blue', alpha=0.6, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Synthetic Quadratic Data')
plt.legend()
plt.show()

# -----------------------------
# 2. Train-test split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# -----------------------------
# 3. Linear Regression
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred_lin = lin_reg.predict(x_test)

print("Linear Regression R²:", r2_score(y_test, y_pred_lin))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lin))

# Plot linear regression fit
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.plot(x_train, lin_reg.predict(x_train), color='red', label='Linear fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()



# -----------------------------
# 4. Polynomial Regression (degree 4)
# -----------------------------
poly = PolynomialFeatures(degree=4)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)

poly_reg = LinearRegression()
poly_reg.fit(x_train_trans, y_train)

y_pred_poly = poly_reg.predict(x_test_trans)

print("Polynomial Regression R²:", r2_score(y_test, y_pred_poly))
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred_poly))


# -----------------------------
# 5. Plot Polynomial Regression Curve
# -----------------------------
# Generate smooth curve for visualization
x_curve = np.linspace(-3, 3, 200).reshape(-1, 1)
x_curve_trans = poly.transform(x_curve)
y_curve = poly_reg.predict(x_curve_trans)

plt.scatter(x, y, color='blue', alpha=0.5, label='Data')
plt.plot(x_curve, y_curve, color='red', linewidth=2, label='Polynomial fit (deg=4)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()

# -----------------------------
# 6. Inspect transformation
# -----------------------------
print("Original sample:", x_train[0])
print("Transformed sample (degree=4):", x_train_trans[0])


m_coef = lin_reg.coef_
print("Linear Regression Coefficients:", m_coef)


c_intercept = lin_reg.intercept_
print(c_intercept)





