import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Utility
def calculate_tour_length(tour, dist_matrix):
    length = 0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i]][tour[i + 1]]
    length += dist_matrix[tour[-1]][tour[0]]
    return length



# Base ACO
class AntColonyBase:
    def __init__(self, dist_matrix, n_ants, n_iter, alpha, beta, rho, Q):
        self.dist = dist_matrix
        self.n = len(dist_matrix)
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.pheromone = np.ones((self.n, self.n))
        self.heuristic = 1 / (dist_matrix + 1e-10)

    def _select_next_city(self, current, visited):
        probs = []
        for j in range(self.n):
            if j not in visited:
                tau = self.pheromone[current][j] ** self.alpha
                eta = self.heuristic[current][j] ** self.beta
                probs.append((j, tau * eta))

        cities, weights = zip(*probs)
        weights = np.array(weights)
        weights /= weights.sum()

        return np.random.choice(cities, p=weights)

    def _construct_solution(self):
        tours, lengths = [], []

        for _ in range(self.n_ants):
            start = np.random.randint(0, self.n)
            tour = [start]

            while len(tour) < self.n:
                nxt = self._select_next_city(tour[-1], set(tour))
                tour.append(nxt)

            length = calculate_tour_length(tour, self.dist)
            tours.append(tour)
            lengths.append(length)

        return tours, lengths



# Ant System
class AntSystem(AntColonyBase):
    def run(self):
        best_len = float('inf')
        best_hist = []

        for _ in range(self.n_iter):
            tours, lengths = self._construct_solution()

            for l in lengths:
                if l < best_len:
                    best_len = l

            best_hist.append(best_len)

            self.pheromone *= (1 - self.rho)

            for t, l in zip(tours, lengths):
                for i in range(len(t)):
                    a, b = t[i], t[(i + 1) % self.n]
                    self.pheromone[a][b] += self.Q / l
                    self.pheromone[b][a] += self.Q / l

        return best_len, best_hist, self.pheromone



# MMAS
class MaxMinAntSystem(AntColonyBase):
    def __init__(self, dist_matrix, tau_min, tau_max, **kwargs):
        super().__init__(dist_matrix, **kwargs)
        self.tau_min = tau_min
        self.tau_max = tau_max

    def run(self):
        best_len = float('inf')
        best_hist = []

        for _ in range(self.n_iter):
            tours, lengths = self._construct_solution()

            idx = np.argmin(lengths)
            best_tour = tours[idx]
            iter_best = lengths[idx]

            if iter_best < best_len:
                best_len = iter_best

            best_hist.append(best_len)

            self.pheromone *= (1 - self.rho)

            for i in range(len(best_tour)):
                a, b = best_tour[i], best_tour[(i + 1) % self.n]
                self.pheromone[a][b] += self.Q / iter_best
                self.pheromone[b][a] += self.Q / iter_best

            self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

        return best_len, best_hist, self.pheromone



# Streamlit UI
st.title("Ant Colony Optimization (TSP)")

# Sidebar parameters
st.sidebar.header("Parameters")

n_ants = st.sidebar.slider("Number of Ants", 5, 100, 20)
n_iter = st.sidebar.slider("Iterations", 10, 500, 100)
alpha = st.sidebar.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic importance)", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Evaporation Rate", 0.1, 0.9, 0.5)
Q = st.sidebar.slider("Q (pheromone deposit)", 0.1, 10.0, 1.0)

tau_min = st.sidebar.slider("Tau Min (MMAS)", 0.01, 1.0, 0.1)
tau_max = st.sidebar.slider("Tau Max (MMAS)", 1.0, 10.0, 5.0)

# Input method
st.subheader("Distance Matrix Input")

option = st.radio("Select Input Method", ["Default", "Upload CSV", "Random"])

if option == "Default":
    dist_matrix = np.array([
        [0, 10, 12, 11, 14],
        [10, 0, 13, 15, 8],
        [12, 13, 0, 9, 14],
        [11, 15, 9, 0, 16],
        [14, 8, 14, 16, 0]
    ])

elif option == "Upload CSV":
    file = st.file_uploader("Upload CSV")
    if file:
        dist_matrix = pd.read_csv(file, header=None).values
    else:
        st.stop()

elif option == "Random":
    n = st.slider("Number of Cities", 3, 20, 5)
    mat = np.random.randint(1, 50, size=(n, n))
    np.fill_diagonal(mat, 0)
    dist_matrix = mat

st.write("Distance Matrix:")
st.dataframe(dist_matrix)

# Run button
if st.button("Run ACO"):

    # Run AS
    as_model = AntSystem(dist_matrix, n_ants, n_iter, alpha, beta, rho, Q)
    as_len, as_hist, as_pher = as_model.run()

    # Run MMAS
    mmas_model = MaxMinAntSystem(
        dist_matrix,
        tau_min=tau_min,
        tau_max=tau_max,
        n_ants=n_ants,
        n_iter=n_iter,
        alpha=alpha,
        beta=beta,
        rho=rho,
        Q=Q
    )
    mmas_len, mmas_hist, mmas_pher = mmas_model.run()

    st.subheader("Results")
    st.write(f"AS Best Length: {as_len}")
    st.write(f"MMAS Best Length: {mmas_len}")


    # Convergence Plot (Seaborn)
    st.subheader("Convergence Plot")

    fig1, ax1 = plt.subplots()
    sns.lineplot(x=range(len(as_hist)), y=as_hist, label="AS", ax=ax1)
    sns.lineplot(x=range(len(mmas_hist)), y=mmas_hist, label="MMAS", ax=ax1)

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Best Tour Length")
    ax1.set_title("AS vs MMAS Convergence")
    ax1.grid(True)

    st.pyplot(fig1)


    # Heatmaps
    st.subheader("Pheromone Heatmaps")

    fig2, ax2 = plt.subplots()
    sns.heatmap(as_pher, annot=True, fmt=".2f", ax=ax2)
    ax2.set_title("AS Pheromone")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.heatmap(mmas_pher, annot=True, fmt=".2f", ax=ax3)
    ax3.set_title("MMAS Pheromone")
    st.pyplot(fig3)