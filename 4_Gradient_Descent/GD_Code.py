# gd_with_graph.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import imageio
import os

# -----------------------------
# Synthetic data (simple linear)
# -----------------------------
np.random.seed(0)
N = 60
X = np.linspace(-3, 3, N)
true_w, true_b = 1.7, -0.5
Y = true_w * X + true_b + np.random.normal(scale=0.8, size=N)

# -----------------------------
# Model, loss and gradient
# -----------------------------
def predict(X, w, b):
    return w * X + b

def mse_loss(Y, Yhat):
    return np.mean((Y - Yhat) ** 2)

def gradients(X, Y, w, b):
    N = X.shape[0]
    Yhat = predict(X, w, b)
    error = Yhat - Y
    # gradients for MSE: dL/dw, dL/db
    dw = (2.0 / N) * np.sum(error * X)
    db = (2.0 / N) * np.sum(error)
    return dw, db

# -----------------------------
# Batch Gradient Descent
# -----------------------------
def gradient_descent(X, Y, w0=0.0, b0=0.0, lr=0.05, steps=80):
    w, b = w0, b0
    history = {"w": [], "b": [], "loss": []}
    for i in range(steps):
        dw, db = gradients(X, Y, w, b)
        w -= lr * dw
        b -= lr * db
        loss = mse_loss(Y, predict(X, w, b))
        history["w"].append(w)
        history["b"].append(b)
        history["loss"].append(loss)
    return history

# run GD
history = gradient_descent(X, Y, w0=0.0, b0=0.0, lr=0.05, steps=80)

# -----------------------------
# Prepare cost surface for contour
# -----------------------------
w_vals = np.linspace(min(history["w"]) - 1.5, max(history["w"]) + 1.5, 120)
b_vals = np.linspace(min(history["b"]) - 1.5, max(history["b"]) + 1.5, 120)
W, B = np.meshgrid(w_vals, b_vals)

def cost_surface(W, B, X, Y):
    # vectorized computation of MSE over grid
    Yhat_grid = W[..., None] * X[None, None, :] + B[..., None]
    loss_grid = np.mean((Yhat_grid - Y[None, None, :]) ** 2, axis=2)
    return loss_grid

Loss_grid = cost_surface(W, B, X, Y)

# -----------------------------
# Plot static loss curve
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(history["loss"], marker='o', linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Loss vs Iteration (Gradient Descent)")
plt.grid(alpha=0.3)
plt.savefig("gd_loss.png", dpi=150)
plt.close()
print("Saved: gd_loss.png")

# -----------------------------
# Draw computation graph (static)
# -----------------------------
def draw_computation_graph(filename="gd_comp_graph.png"):
    G = nx.DiGraph()
    # nodes: X, w, b, w*x, w*x + b, loss
    G.add_node("X", shape="ellipse")
    G.add_node("w", shape="ellipse")
    G.add_node("b", shape="ellipse")
    G.add_node("w*x", shape="box")
    G.add_node("w*x+b", shape="box")
    G.add_node("loss", shape="oval")

    G.add_edges_from([("X","w*x"), ("w","w*x"), ("w*x","w*x+b"), ("b","w*x+b"), ("w*x+b","loss")])

    pos = {
        "X": (-1, 0.5),
        "w": (-1, -0.5),
        "b": (0.2, -1.0),
        "w*x": (0.5, 0.0),
        "w*x+b": (1.3, 0.0),
        "loss": (2.3, 0.0)
    }
    plt.figure(figsize=(7,3))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="#a6cee3", font_weight="bold", arrowsize=20)
    plt.title("Computation Graph for y = w*x + b  → loss")
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

draw_computation_graph()

# -----------------------------
# Make GIF: contour + path + loss subplot
# -----------------------------
def make_animation(history, W, B, Loss_grid, X, Y, filename="gd_contour_path.gif"):
    # ensure frame dir exists / remove old frames
    frames = []
    tmp_files = []
    iters = len(history["w"])

    # Precompute levels for contour
    levels = np.logspace(np.log10(Loss_grid.min()+1e-6), np.log10(Loss_grid.max()+1e-6), 25)

    for t in range(iters):
        fig, axes = plt.subplots(1,2, figsize=(10,4))
        ax1, ax2 = axes

        # left: contour and path up to t
        cs = ax1.contourf(W, B, Loss_grid, levels=levels, norm=None, alpha=0.9)
        ax1.contour(W, B, Loss_grid, levels=levels, colors='k', linewidths=0.3, alpha=0.3)
        ax1.set_xlabel("w"); ax1.set_ylabel("b")
        ax1.set_title("Cost surface and parameter path")

        # plot true parameter point
        ax1.scatter([true_w], [true_b], color='white', edgecolor='k', s=80, label='true (w,b)')
        # path
        ws = history["w"][:t+1]
        bs = history["b"][:t+1]
        ax1.plot(ws, bs, marker='o', color='red', linewidth=1.5, label='GD path')
        ax1.scatter(ws[-1], bs[-1], color='yellow', edgecolor='k', s=70, label='current (w,b)')
        ax1.legend(loc='upper right', fontsize='small')

        # right: loss curve up to t
        ax2.plot(history["loss"], color='gray', alpha=0.5)
        ax2.plot(range(t+1), history["loss"][:t+1], color='blue', marker='o')
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("MSE Loss")
        ax2.set_title("Loss vs Iteration")
        ax2.grid(alpha=0.3)

        fig.suptitle(f"Gradient Descent step {t+1}/{iters}    w={ws[-1]:.3f}, b={bs[-1]:.3f}, loss={history['loss'][t]:.4f}", fontsize=10)

        frame_name = f"_gd_frame_{t:03d}.png"
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(frame_name, dpi=120)
        plt.close(fig)
        tmp_files.append(frame_name)
        frames.append(imageio.imread(frame_name))

    # save GIF
    imageio.mimsave(filename, frames, duration=0.35)
    print(f"Saved animation GIF: {filename}")

    # cleanup temp frames
    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass

make_animation(history, W, B, Loss_grid, X, Y)

# -----------------------------
# Print final results
# -----------------------------
final_w = history["w"][-1]
final_b = history["b"][-1]
final_loss = history["loss"][-1]
print("Final parameters:")
print(f"  w = {final_w:.4f}, b = {final_b:.4f}, loss = {final_loss:.6f}")
print("Generated files: gd_loss.png, gd_comp_graph.png, gd_contour_path.gif")
