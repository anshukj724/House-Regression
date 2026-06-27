import kagglehub

import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt



path = kagglehub.dataset_download("shibumohapatra/house-price")

df = pd.read_csv(os.path.join(path, "1553768847-housing.csv"))



# clean

df['total_bedrooms'] = df['total_bedrooms'].fillna(df["total_bedrooms"].median())



# feature engineering

df['rooms_per_households']      = df['total_rooms']    / df['households']

df['bedrooms_per_room']         = df['total_bedrooms'] / df['total_rooms']

df['population_per_household']  = df['population']     / df['households']



X = df.drop(["median_house_value", "ocean_proximity"], axis=1).to_numpy()

y = df["median_house_value"].to_numpy()



# shuffle

np.random.seed(42)

indices = np.arange(len(X))

np.random.shuffle(indices)

X, y = X[indices], y[indices]



# split

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]

y_train, y_test = y[:split], y[split:]



# standardize

mean = X_train.mean(axis=0)

std  = X_train.std(axis=0)

X_train = (X_train - mean) / std

X_test  = (X_test  - mean) / std





class LinearRegression:

    def __init__(self, lr=0.01, epochs=1000):

        self.weights      = None

        self.bias         = 0

        self.lr           = lr

        self.epochs       = epochs

        self.loss_history = []          



    def fit(self, X, y):

        n, features  = X.shape

        self.weights = np.zeros(features)



        for i in range(self.epochs):

            y_pred = X @ self.weights + self.bias

            loss   = np.mean((y - y_pred) ** 2)



            self.loss_history.append(loss) 



            error         = y_pred - y

            dw            = (2 / n) * X.T @ error

            db            = (2 / n) * np.sum(error)

            self.weights -= self.lr * dw

            self.bias    -= self.lr * db



            if i % 100 == 0:

                print(f"Epoch {i}  Loss: {loss:,.2f}")



    def predict(self, X):

        return X @ self.weights + self.bias





model = LinearRegression(lr=0.01, epochs=1000)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



# ── Metrics ──────────────────────────────────────────────────────

mse  = np.mean((y_test - y_pred) ** 2)

rmse = np.sqrt(mse)

mae  = np.mean(np.abs(y_test - y_pred))                        

r2   = 1 - (np.sum((y_test - y_pred) ** 2) /

             np.sum((y_test - y_test.mean()) ** 2))            



print(f"\nMSE:  {mse:,.2f}")

print(f"RMSE: ${rmse:,.2f}")

print(f"MAE:  ${mae:,.2f}")

print(f"R²:   {r2:.4f}")



# ── Plots ─────────────────────────────────────────────────────────

plt.style.use("dark_background")

PURPLE, GREEN, ORANGE = "#7c6cfc", "#3dd68c", "#f0a050"



fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fig.patch.set_facecolor("#0f0f13")

for ax in axes:

    ax.set_facecolor("#1a1a24")

    ax.tick_params(colors="#5a5a6e")

    for spine in ax.spines.values():

        spine.set_edgecolor("#2a2a38")



# plot 1 — predicted vs actual

axes[0].scatter(y_test, y_pred, alpha=0.35, s=12, color=PURPLE)

lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]

axes[0].plot(lims, lims, "--", color=GREEN, linewidth=1.5)

axes[0].set_title(f"Predicted vs Actual  |  R²={r2:.3f}", color="#e2e0d8")

axes[0].set_xlabel("Actual ($)", color="#aaa")

axes[0].set_ylabel("Predicted ($)", color="#aaa")



# plot 2 — residuals

residuals = y_pred - y_test

axes[1].hist(residuals, bins=40, color=PURPLE, alpha=0.7, edgecolor="none")

axes[1].axvline(0, color=GREEN, linewidth=1.5, linestyle="--")

axes[1].set_title("Residual Distribution", color="#e2e0d8")

axes[1].set_xlabel("Residual ($)", color="#aaa")

axes[1].set_ylabel("Count", color="#aaa")



# plot 3 — loss curve

axes[2].plot(model.loss_history, color=ORANGE, linewidth=1.8)

axes[2].fill_between(range(len(model.loss_history)),

                     model.loss_history, alpha=0.08, color=ORANGE)

axes[2].set_title("Training Loss", color="#e2e0d8")

axes[2].set_xlabel("Epoch", color="#aaa")

axes[2].set_ylabel("MSE Loss", color="#aaa")



plt.tight_layout()

plt.savefig("housing_dashboard.png", dpi=150, facecolor="#0f0f13", bbox_inches="tight")

plt.show()

        

