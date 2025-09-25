import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Synthetic Data
# -----------------------------
np.random.seed(42)
N = 80
X = np.linspace(-3, 3, N)
true_w, true_b = 2.0, -1.0
Y = true_w * X + true_b + np.random.normal(scale=1.0, size=N)

X = X.reshape(-1, 1)  # column vector

# -----------------------------
# Ridge Regression (L2)
# -----------------------------
def ridge_grad_descent(X, Y, lr=0.05, steps=100, lam=0.5):
    n = X.shape[0]
    w, b = 0.0, 0.0
    history = {"w": [], "b": [], "loss": []}

    for _ in range(steps):
        Yhat = w * X[:,0] + b
        error = Yhat - Y

        dw = (2/n) * np.sum(error * X[:,0]) + 2 * lam * w
        db = (2/n) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        loss = np.mean(error**2) + lam * (w**2)
        history["w"].append(w)
        history["b"].append(b)
        history["loss"].append(loss)
    return history

# -----------------------------
# Lasso Regression (L1)
# -----------------------------
def lasso_grad_descent(X, Y, lr=0.05, steps=100, lam=0.5):
    n = X.shape[0]
    w, b = 0.0, 0.0
    history = {"w": [], "b": [], "loss": []}

    for _ in range(steps):
        Yhat = w * X[:,0] + b
        error = Yhat - Y

        dw = (2/n) * np.sum(error * X[:,0])
        db = (2/n) * np.sum(error)

        # Subgradient for L1
        if w > 0:
            dw += lam
        elif w < 0:
            dw -= lam
        else:
            dw += 0

        w -= lr * dw
        b -= lr * db

        loss = np.mean(error**2) + lam * np.abs(w)
        history["w"].append(w)
        history["b"].append(b)
        history["loss"].append(loss)
    return history

# -----------------------------
# Run both
# -----------------------------
ridge_hist = ridge_grad_descent(X, Y, lam=1.0)
lasso_hist = lasso_grad_descent(X, Y, lam=1.0)

# -----------------------------
# Plot Loss Curves
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(ridge_hist["loss"], label="Ridge (L2)", color="blue")
plt.plot(lasso_hist["loss"], label="Lasso (L1)", color="red")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("ridge_lasso_loss.png", dpi=150)
plt.close()
print("✅ Saved: ridge_lasso_loss.png")

# -----------------------------
# Contour Plot of Loss Surface
# -----------------------------
w_vals = np.linspace(-3, 3, 100)
b_vals = np.linspace(-3, 3, 100)
W, B = np.meshgrid(w_vals, b_vals)

def ridge_loss_grid(W, B, X, Y, lam):
    Yhat = W[...,None]*X[None,None,:] + B[...,None]
    return np.mean((Yhat - Y)**2, axis=2) + lam*(W**2)

def lasso_loss_grid(W, B, X, Y, lam):
    Yhat = W[...,None]*X[None,None,:] + B[...,None]
    return np.mean((Yhat - Y)**2, axis=2) + lam*np.abs(W)

ridge_grid = ridge_loss_grid(W,B,X[:,0],Y, lam=1.0)
lasso_grid = lasso_loss_grid(W,B,X[:,0],Y, lam=1.0)

fig, axes = plt.subplots(1,2, figsize=(12,5))

cs1 = axes[0].contourf(W, B, ridge_grid, levels=30, cmap="Blues")
axes[0].plot(ridge_hist["w"], ridge_hist["b"], marker="o", color="red")
axes[0].set_title("Ridge Regression Path")
axes[0].set_xlabel("w"); axes[0].set_ylabel("b")

cs2 = axes[1].contourf(W, B, lasso_grid, levels=30, cmap="Reds")
axes[1].plot(lasso_hist["w"], lasso_hist["b"], marker="o", color="blue")
axes[1].set_title("Lasso Regression Path")
axes[1].set_xlabel("w"); axes[1].set_ylabel("b")

plt.savefig("ridge_lasso_contours.png", dpi=150)
plt.close()
print("✅ Saved: ridge_lasso_contours.png")

# -----------------------------
# Computation Graph (Generic)
# -----------------------------
def draw_computation_graph(filename="regression_comp_graph.png"):
    G = nx.DiGraph()
    G.add_nodes_from(["X", "w", "b", "y_pred", "loss"])
    G.add_edges_from([
        ("X","y_pred"), ("w","y_pred"), ("b","y_pred"),
        ("y_pred","loss")
    ])
    pos = {"X":(-1,0), "w":(-1,1), "b":(-1,-1), "y_pred":(0,0), "loss":(1,0)}
    plt.figure(figsize=(6,3))
    nx.draw(G, pos, with_labels=True, node_color="#ffcc99", node_size=2000, font_weight="bold", arrowsize=20)
    plt.title("Computation Graph (Ridge/Lasso)")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {filename}")

draw_computation_graph()

# -----------------------------
# Print Results
# -----------------------------
print(f"Final Ridge: w={ridge_hist['w'][-1]:.3f}, b={ridge_hist['b'][-1]:.3f}, loss={ridge_hist['loss'][-1]:.3f}")
print(f"Final Lasso: w={lasso_hist['w'][-1]:.3f}, b={lasso_hist['b'][-1]:.3f}, loss={lasso_hist['loss'][-1]:.3f}")
