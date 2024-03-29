# PAR.jl
Repository with Periodic Autorregressive models.

The main goal of the repository is to implement Periodic Autorregressive (PAR) models and some simple variations of the model.
The full methodology of PAR models can be found in this article (Maceira, Maria & Penna, Débora & Damazio, Jorge. (2006). Geração de Cenários Sintéticos de Energia e Vazão para o Planejamento da Operação Energética. Cadernos do IME : Série Estatística. 21. 10.12957/cadest.2006.15760.) a link to the article's pdf can be found [here](https://www.e-publicacoes.uerj.br/index.php/cadest/article/download/15760/11931) (Available only in Portuguese.)

To add the package you can do:
```julia
add https://github.com/psrenergy/PAR.jl.git
```

and to run an example you can do:
```julia
using PAR
funil_grande = include(joinpath(pkgdir(PAR), "test", "data", "funil_grande.jl"))
batalha = include(joinpath(pkgdir(PAR), "test", "data", "batalha.jl"))

n_stages = 12
p_lim = 6
par_1 = PARp(funil_grande, n_stages, p_lim; information_criteria = "aic");
par_2 = PARp(batalha, n_stages, p_lim; information_criteria = "aicc");
fit_par!(par_1);
fit_par!(par_2);

steps_ahead = 100
n_scenarios = 1000
scen = simulate_par([par_1; par_2], steps_ahead, n_scenarios)
```
