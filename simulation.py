import numpy as np
import matplotlib.pylab as plt

def calc_prob(r, s, mu):
    return (1 - mu) * (1+s) * r / (1 + s*r)

def calc_equilibium(s):
    return max(0, 1 - LOSS_RATE - LOSS_RATE / s)

def simulation(
        population_size, 
        n_enhanced_init, 
        wildtype_fitness,
        enhanced_fitness,
        loss_rate,
        n_iter):

    list_n_enhanced = []
    s = (enhanced_fitness - wildtype_fitness) / wildtype_fitness
    m = n_enhanced_init
    rng = np.random.default_rng()
    
    list_n_enhanced.append(m)
    for _ in range(n_iter):
        r = m / population_size
        p = calc_prob(r, s, loss_rate)
        m = rng.binomial(population_size, p)
        list_n_enhanced.append(m)
        
    return list_n_enhanced


def run_simulations():

    list_s = np.arange(0, 1.5, 0.1)
    results = []
    for s in list_s:
        tmp_results = []
        ENHANCED_FITNESS = 1.0 + s
        for _ in range(5):
            list_n_enhanced = simulation(
                POPULATION_SIZE,
                N_ENHANCED_INIT,
                WILDTYPE_FITNESS,
                ENHANCED_FITNESS,
                LOSS_RATE,
                N_ITER
            )
            tmp_results.append(list_n_enhanced[-1])
        results.append(tmp_results)

    for i, f in enumerate(list_s):
        plt.scatter([f for _ in range(5)], results[i])

    xvals = np.arange(0.01, 1.5, 0.01)
    plt.plot(
        xvals, 
        [POPULATION_SIZE * calc_equilibium(x) for x in xvals]
        )

    plt.savefig("figure.png", format="png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    POPULATION_SIZE = 15000
    N_ENHANCED_INIT = 7500
    WILDTYPE_FITNESS = 1.0
    LOSS_RATE = 0.2
    N_ITER = 100000
    
    run_simulations()