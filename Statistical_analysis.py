import os
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.stats import t
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import weibull_min

from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.special import gamma as special_gamma

#=========FUNCTIONS=========
def n_necessario(mean_, std_, n_, alpha, k):

	average = mean_
	std = std_
	n = n_

	gl = n - 1
	critical_t = t.ppf(1 - (alpha / 2), gl)

	n_needed = math.ceil((2 * critical_t * std / (k * average))**2)
	return n_needed

#=========DATA=========

dados = []
sample_label = []
bins_dados = []
sample_average = []
sample_std = []
sample_n = []

sample_label.append("Slag Clean")
sample_label.append("Mineral Clean")
sample_label.append("Slag+CAP50/70")
sample_label.append("Mineral+CAP50/70")
sample_label.append("Mineral+CAP50/70")
sample_label.append("Mineral+CAP50/70(fully cover)")
sample_label.append("Mineral2+CAP50/70+1%LCC")
sample_label.append("Mineral+CAP50/70+1%LCC")
sample_label.append("Mineral+AMP60/85-E")
sample_label.append("Mineral+CAP50/70 (low cover)")
sample_label.append("Mineral3+CAP50/70")
sample_label.append("Mineral3+CAP50/70")

#=========GETTING DATA=========
current_directory = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(current_directory + '/data')

for file_ in files:
    dados.append(np.loadtxt(current_directory + '/data/' + file_)/100)	

#=========GAUSSIAN=========
gaussian_mean = []
gaussian_std = []

gaussian_ks = []


#=========BETA=========
beta_alpha = []
beta_beta = []

beta_ks = []


#=========GAMMA=========
gamma_alpha = []
gamma_beta = []

gamma_ks = []

#=========LOG-NORM=========
lognorm_shape = []
lognorm_loc = []
lognorm_scale = []

lognorm_ks = []

#=========WEIBULL=========
weibull_shape = []
weibull_scale = []

weibull_ks = []

#=========MINIMUM N=========
n_min = []

#=========Finding Paramenters=========
for i in range(len(dados)):
  #----Data----
  size = len(dados.copy()[i])
  sample_n.append(size)
  bins_dados.append(int(math.sqrt(size)))

  media = np.mean(dados.copy()[i])
  std = np.std(dados.copy()[i], ddof=1)
  variancia = np.power(std, 2)

  sample_average.append(media)
  sample_std.append(std)

  #----N min----
  n_needed = n_necessario(media, std, size,  0.05, 0.1)
  n_min.append(n_needed)
    
  #----Gaussian----
  gaussian_mean.append(media)
  gaussian_std.append(std)

  n_ks_e, n_ks_p = kstest(dados.copy()[i], cdf='norm', args=(media, std), N=size)
  gaussian_ks.append((n_ks_e, n_ks_p))

  #----Beta----
  #   b_alpha, b_beta, _, _ = beta.fit(dados.copy()[i])
  b_alpha = media * (((media*(1 - media)) / variancia) - 1)
  b_beta = (1 - media) * (((media*(1 - media)) / variancia) - 1)
  beta_alpha.append(b_alpha)
  beta_beta.append(b_beta)

  b_ks_e, b_ks_p = kstest(dados.copy()[i], cdf='beta', args=(b_alpha, b_beta), N=size)
  beta_ks.append((b_ks_e, b_ks_p))

  #----Gamma----
  g_alpha = np.power(media, 2)/variancia
  g_beta = variancia/media
  gamma_alpha.append(g_alpha)
  gamma_beta.append(g_beta)
  g_ks_e, g_ks_p = kstest(dados.copy()[i], cdf='gamma', args=(g_alpha, 0, g_beta), N=size)
  gamma_ks.append((g_ks_e, g_ks_p))

  #----Log-norm----
  
  ln_shape, ln_loc, ln_scale = lognorm.fit(dados.copy()[i])
  
  lognorm_shape.append(ln_shape)
  lognorm_loc.append(ln_loc)
  lognorm_scale.append(ln_scale)

  ln_ks_e, ln_ks_p = kstest(dados.copy()[i], cdf='lognorm', args=(ln_shape, ln_loc, ln_scale), N=size)
  lognorm_ks.append((ln_ks_e, ln_ks_p))

  #----Weibull----
  k = (std/media)**(-1.086)
  g = special_gamma(1+(1/k))
  c = media/g
  weibull_shape.append(k)
  weibull_scale.append(c)

  w_ks_e, w_ks_p = kstest(dados.copy()[i], cdf='weibull_min', args=(k, 0, c), N=size)
  weibull_ks.append((w_ks_e, w_ks_p))

#=========Visualization=========
fig, axs = plt.subplots(3, 4, figsize=(16, 8))
count = 0
for i in range(3):
  for j in range(4):
    axs[i, j].hist(
      dados[count],
      bins=bins_dados[count],
      density=True,
      alpha=0.2,
      color='red',
      label = (f'x̅={round(sample_average[count], 2)}, s={round(sample_std[count], 2)}\n'
               rf'n={sample_n[count]}, $n_{{\mathrm{{min}}}}$={n_min[count]}')
    )
    axs[i, j].set_title(f'Sample A{str(count+1).zfill(2)}: {sample_label[count]}', fontsize=8, fontweight='bold')
    axs[i, j].set_xlabel(f'Coated', fontsize=8)
    axs[i, j].set_ylabel(f'Probability of densities (%)', fontsize=8)
    axs[i, j].grid(True)
    axs[i, j].set_xlim(0, 1)

    xmin, xmax = 0, 1
    x = np.linspace(xmin, xmax, 5000)

    #----plot gaussian----
    gaussian_distribution = norm.pdf(x, gaussian_mean[count], gaussian_std[count])
    axs[i, j].plot(x, gaussian_distribution, color='#e74c3c', linestyle='-', linewidth=1, label=f'gaussian p-value: {round(gaussian_ks[count][1], 2)}')

    #----plot beta----
    beta_distribution = beta.pdf(x, beta_alpha[count], beta_beta[count])
    axs[i, j].plot(x, beta_distribution, color='#f39c12', linestyle='-', linewidth=1, label=f'beta p-value: {round(beta_ks[count][1], 2)}')

    #----plot gamma----
    gamma_distribution = gamma.pdf(x, gamma_alpha[count], scale=gamma_beta[count])
    axs[i, j].plot(x, gamma_distribution, color='#2ecc71', linestyle='--', linewidth=1, label=f'gamma p-value: {round(gamma_ks[count][1], 2)}')
    
    #----plot log-norm----
    lognorm_distribution = lognorm.pdf(x, lognorm_shape[count], loc=lognorm_loc[count], scale=lognorm_scale[count])
    axs[i, j].plot(x, lognorm_distribution, color='#3498db', linestyle='-.', linewidth=1, label=f'log-norm p-value: {round(lognorm_ks[count][1], 2)}')

    #----plot weibull----
    weibull_min_distriution = weibull_min.pdf(x, weibull_shape[count], scale=weibull_scale[count])
    axs[i, j].plot(x, weibull_min_distriution, color='#9b59b6', linestyle=':', linewidth=1, label=f'weibull_min p-value: {round(weibull_ks[count][1], 2)}')

    #print(f"{round(gaussian_ks[count][1], 3)}\t{round(beta_ks[count][1], 3)}\t{round(gamma_ks[count][1], 3)}\t{round(lognorm_ks[count][1], 3)}\t{round(weibull_ks[count][1], 3)}")

    axs[i, j].legend(ncol=1, fontsize=6)
    count += 1
count = 0

# Ajuste o layout para evitar sobreposição
plt.tight_layout()

# Exiba o gráfico
plt.show()

# #====Visualization for an especific sample=========
# count = 0
# for a in range(12):
#   plt.hist(
#     dados[count],
#     bins=bins_dados[count],
#     density=True,
#     alpha=0.2,
#     color='red',
#     label = (f'x̅={round(sample_average[count], 2)}, s={round(sample_std[count], 2)}\n'
#               rf'n={sample_n[count]}, $n_{{\mathrm{{min}}}}$={n_min[count]}')
#   )
#   plt.title(f'Sample A{str(count+1).zfill(2)}: {sample_label[count]}', fontsize=14, fontweight='bold')
#   plt.xlabel(f'Coated', fontsize=14)
#   plt.ylabel(f'Probability of densities (%)', fontsize=14)
#   plt.grid(True)
#   plt.xlim(0, 1)

#   xmin, xmax = 0, 1
#   x = np.linspace(xmin, xmax, 5000)

#   #----plot gaussian----
#   gaussian_distribution = norm.pdf(x, gaussian_mean[count], gaussian_std[count])
#   plt.plot(x, gaussian_distribution, color='#e74c3c', linestyle='-', linewidth=3, label=f'gaussian p-value: {round(gaussian_ks[count][1], 2)}')

#   #----plot beta----
#   beta_distribution = beta.pdf(x, beta_alpha[count], beta_beta[count])
#   plt.plot(x, beta_distribution, color='#f39c12', linestyle='-', linewidth=3, label=f'beta p-value: {round(beta_ks[count][1], 2)}')

#   #----plot gamma----
#   gamma_distribution = gamma.pdf(x, gamma_alpha[count], scale=gamma_beta[count])
#   plt.plot(x, gamma_distribution, color='#3498db', linestyle='--', linewidth=3, label=f'gamma p-value: {round(gamma_ks[count][1], 2)}')

#   #----plot log-norm----
#   lognorm_distribution = lognorm.pdf(x, lognorm_shape[count], loc=lognorm_loc[count], scale=lognorm_scale[count])
#   plt.plot(x, lognorm_distribution, color='#2ecc71', linestyle='-.', linewidth=3, label=f'log-norm p-value: {round(lognorm_ks[count][1], 2)}')

#   #----plot weibull----
#   weibull_min_distriution = weibull_min.pdf(x, weibull_shape[count], scale=weibull_scale[count])
#   plt.plot(x, weibull_min_distriution, color='#9b59b6', linestyle=':', linewidth=3, label=f'weibull_min p-value: {round(weibull_ks[count][1], 2)}')

#   #print(f"{round(gaussian_ks[count][1], 3)}\t{round(beta_ks[count][1], 3)}\t{round(gamma_ks[count][1], 3)}\t{round(lognorm_ks[count][1], 3)}\t{round(weibull_ks[count][1], 3)}")

#   plt.legend(ncol=1, fontsize=14)

#   # Ajuste o layout para evitar sobreposição
#   plt.tight_layout()

#   # Ajuste o tamanho da fonte dos números dos eixos
#   plt.tick_params(axis='both', which='major', labelsize=14)

#   # Exiba o gráfico
#   plt.show()
#   count += 1