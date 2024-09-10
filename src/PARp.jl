mutable struct AR
    y::Vector{Float64}
    p::Int
    ϕ::Vector{Float64}
    aic::Float64
    aicc::Float64
    var_resid::Float64
    resid::Vector{Float64}
    p_values::Vector{Float64}
    coef_table
    ols
    fitted_y::Vector{Float64}
    fitted_X::Matrix{Float64}
    function AR(y::Vector{Float64}, p::Int)
        assert_series_without_missing(y)
        return new(y,
            p,
            zeros(Float64, p),
            zero(Float64),
            zero(Float64),
            zero(Float64),
            zeros(Float64, 1),
            zeros(Float64, p),
            nothing,
            nothing,
            zeros(Float64, 1),
            zeros(Float64, 1, 1)
        )
    end
end

aic(ar::AR) = ar.aic
aicc(ar::AR) = ar.aicc
coef(ar::AR) = ar.ϕ
residuals(ar::AR) = ar.resid
residuals_variance(ar::AR) = ar.var_resid

mutable struct PARp
    y::Vector{Float64}
    y_normalized::Vector{Float64}
    μ_stage::Vector{Float64}
    σ_stage::Vector{Float64}
    candidate_AR_stage::Vector{Vector{AR}}
    best_AR_stage::Vector{AR}
    seasonal::Int
    p_lim::Int
    information_criteria::String
    last_stage::Int
    series_with_zeros::Bool

    function PARp(y::Vector{Float64}, seasonal::Int, p_lim::Int; information_criteria::String = "aic")
        assert_series_without_missing(y)
        series_with_zeros = series_with_only_zeros(y)
        y_normalized, μ_stage, σ_stage = normalize_series(y, seasonal)
        candidate_AR_stage = Vector{Vector{AR}}(undef, 0)
        best_AR_stage = Vector{AR}(undef, 0)
        last_stage = mod1(length(y), seasonal)
        return new(y, 
                y_normalized, 
                μ_stage, 
                σ_stage, 
                candidate_AR_stage,
                best_AR_stage,
                seasonal, 
                p_lim,
                information_criteria,
                last_stage,
                series_with_zeros
            )
    end
end

num_stages(par::PARp) = par.seasonal
p_limit(par::PARp) = par.p_lim

function residuals_of_best_model_at_stage(par::PARp, stage::Int)
    return residuals(par.best_AR_stage[stage])
end
function residuals_of_best_models_at_stage(par_models::Vector{PARp}, current_stage_to_predict::Int)
    residuals = Vector{Vector{Float64}}(undef, 0)
    for pm in par_models
        push!(residuals, residuals_of_best_model_at_stage(pm, current_stage_to_predict))
    end
    return concatenate_from_the_bottom_elements(residuals)
end

function build_y_X(y_normalized::Vector{Float64}, p::Int, stage::Int, seasonal::Int)
    n_total = length(y_normalized)
    correct_stage_idx = collect(stage:seasonal:n_total)
    y_normalized_at_correct_stage = y_normalized[correct_stage_idx]
    n_correct_stage = length(y_normalized_at_correct_stage)
    X = Matrix{Float64}(undef, n_correct_stage, p)
    for i in 1:p
        lag_stage_idx = correct_stage_idx .- i
        for (t, idx) in enumerate(lag_stage_idx)
            if idx < 1
                X[t, i] = NaN
            else
                X[t, i] = y_normalized[idx]
            end
        end
    end
    return y_normalized_at_correct_stage[p+1:end], X[p+1:end, :]
end

function fit_ar!(ar::AR; stage::Int = 1, par_seasonal::Int = 1)
    y_to_fit, X_to_fit = build_y_X(ar.y, ar.p, stage, par_seasonal)
    ar.fitted_X = X_to_fit
    ar.fitted_y = y_to_fit
    if all(iszero, y_to_fit) || all(iszero, X_to_fit)
        ar.ols = nothing
        ar.ϕ = Float64[]
        ar.coef_table = nothing
        ar.p_values = zeros(1)
        ar.resid = y_to_fit
        ar.var_resid = Inf
        ar.aic = Inf
        ar.aicc = Inf
    else
        ols = lm(X_to_fit, y_to_fit)
        ar.ols = ols
        ar.ϕ = coef(ols)
        ar.coef_table = coeftable(ols)
        ar.p_values = ar.coef_table.cols[ar.coef_table.pvalcol]
        ar.resid = y_to_fit - X_to_fit * ar.ϕ
        ar.var_resid = var(ar.resid)
        n = length(y_to_fit) + ar.p
        # A different but similiar form of aic can be found on Akaike original paper.
        # A note on the difference between the general form -2L + 2k and this one 
        # can also be found on wikipeadia talking about least squares estimators
        ar.aic = n * log(ar.var_resid * (n - 1)) + 2 * ar.p
        ar.aicc = ar.aic + (2 * ar.p^2 + 2 * ar.p)/(n - ar.p - 1)
    end
    return ar
end

function fit_par!(par::PARp)
    # fit all AR models
    for stage in 1:num_stages(par)
        candiate_ar_per_stage = AR[]
        for p in 0:par.p_lim
            candidate_ar_at_stage = AR(par.y_normalized, p)
            fit_ar!(candidate_ar_at_stage; stage = stage, par_seasonal = par.seasonal)
            push!(candiate_ar_per_stage, candidate_ar_at_stage)
        end
        push!(par.candidate_AR_stage, candiate_ar_per_stage)
        best_model = select_best_model(candiate_ar_per_stage, par.information_criteria)
        push!(par.best_AR_stage, best_model)
    end
    return par
end

function assert_same_number_of_stages(par_models::Vector{PARp})
    @assert length(unique(num_stages.(par_models))) == 1
end
function assert_same_p_limit(par_models::Vector{PARp})
    @assert length(unique(p_limit.(par_models))) == 1
end

simulate_par(
    par::PARp,
    stepds_ahead::Int,
    n_scenarios::Int;
    lognormal_noise::Bool = true,
    return_noise::Bool = false,
    seed::Int = 1234,
) = simulate_par(
    [par],
    stepds_ahead,
    n_scenarios;
    lognormal_noise = lognormal_noise,
    return_noise = return_noise,
    seed = seed,
)
function simulate_par(par_models::Vector{PARp}, steps_ahead::Int, n_scenarios::Int; lognormal_noise::Bool = true, return_noise::Bool = false, seed::Int = 1234)
    Random.seed!(seed)
    assert_same_number_of_stages(par_models)
    assert_same_p_limit(par_models)
    n_stages = num_stages(par_models[1])
    p_lim = p_limit(par_models[1])
    n_models = length(par_models)
    scenarios_normalized = zeros(steps_ahead + p_lim, n_models, n_scenarios)
    scenarios = zeros(steps_ahead + p_lim, n_models, n_scenarios)
    noise_matrix = zeros(steps_ahead, n_models, n_scenarios)
    # Fill the first part of scenarios with historical data
    for (i, pm) in enumerate(par_models)
        scenarios_normalized[1:p_lim, i,  :] .= pm.y_normalized[end-p_lim+1:end]
        scenarios[1:p_lim, i,  :] .= pm.y[end-p_lim+1:end]
    end
    # Simulate on the standardized series
    for t in 1:steps_ahead
        current_stage_to_predict = mod1(par_models[1].last_stage + t, n_stages)
        t_scen_idx = t + p_lim
        residuals_matrix = residuals_of_best_models_at_stage(par_models, current_stage_to_predict)
        cor_matrix = cor(residuals_matrix)
        ruido_normal = randn(n_scenarios, n_models) 
        # ruido_correlacionado = ruido_normal * cholesky(cor_matrix).U
        ruido_correlacionado = ruido_normal
        for (i, pm) in enumerate(par_models)
            if pm.series_with_zeros
                scenarios[t_scen_idx, i, :] .= zero(Float64)
                continue
            end
            # Every observation is the same on a certain stage
            if all(iszero, pm.best_AR_stage[current_stage_to_predict].fitted_y)
                if t_scen_idx == 1
                    scenarios[t_scen_idx, i, :] .= pm.y[end]
                else
                    scenarios[t_scen_idx, i, :] .= scenarios[t_scen_idx-1, i, :]
                end
                continue
            end
            current_model_p = pm.best_AR_stage[current_stage_to_predict].p
            for s in 1:n_scenarios
                # Evaluate the deterministic parts of the scenarios
                autorregressive_normalized = dot(
                    scenarios_normalized[t_scen_idx-1 : -1 : t_scen_idx-current_model_p, i, s],
                    pm.best_AR_stage[current_stage_to_predict].ϕ
                )

                # Calculate the noise and the parameters of the viable 3 parameter log normal
                if lognormal_noise
                    lower_bound_log_normal = - (pm.μ_stage[current_stage_to_predict] / (pm.σ_stage[current_stage_to_predict] + 1e-5)) - 
                                                autorregressive_normalized
                    λ = (pm.best_AR_stage[current_stage_to_predict].var_resid/lower_bound_log_normal^2) + 1
                    μ_log_normal = 0.5 * log(pm.best_AR_stage[current_stage_to_predict].var_resid/ (λ * (λ - 1)))
                    σ_log_normal = sqrt(log(λ))
                    ruido = exp(ruido_correlacionado[s, i]*σ_log_normal + μ_log_normal) + lower_bound_log_normal
                else
                    ruido = ruido_correlacionado[s, i]
                end
                noise_matrix[t, i, s] = ruido
                # Evaluate the scenario value
                scenarios_normalized[t_scen_idx, i, s] = autorregressive_normalized + ruido
                scenarios[t_scen_idx, i, s] = scenarios_normalized[t_scen_idx, i, s] * pm.σ_stage[current_stage_to_predict] + pm.μ_stage[current_stage_to_predict]
            end
        end
    end
    if return_noise
        return scenarios[p_lim+1:end, :, :], noise_matrix
    else
        return scenarios[p_lim+1:end, :, :]
    end
end

function simulate_par_f_b(par_models::Vector{PARp}, steps_ahead::Int, n_scenarios::Int, n_backw::Int; has_global_lower_bound::Bool = true)
    assert_same_number_of_stages(par_models)
    assert_same_p_limit(par_models)
    n_stages = num_stages(par_models[1])
    p_lim = p_limit(par_models[1])
    n_models = length(par_models)
    scenarios_normalized = zeros(steps_ahead + p_lim, n_models, n_scenarios)

    # TODO: reorder this matrix we are having too many cache misses....
    scenarios = zeros(steps_ahead + p_lim, n_models, n_scenarios, n_backw)
    # Fill the first part of scenarios with historical data
    for (i, pm) in enumerate(par_models)
        scenarios_normalized[1:p_lim, i,  :] .= pm.y_normalized[end-p_lim+1:end]
        scenarios[1:p_lim, i,  :, :] .= pm.y[end-p_lim+1:end]
    end
    # Simulate on the standardized series
    for t in 1:steps_ahead
        current_stage_to_predict = mod1(par_models[1].last_stage + t, n_stages)
        t_scen_idx = t + p_lim
        MAX = 7.0
        ruido_normal = clamp.(randn(n_backw, n_models), -MAX, MAX)
        ruido_correlacionado = ruido_normal
        # shufle one of the "back" noises generated for this stage
        # for each of the scenarios (needs to be here to preserve correlation)
        noise_index_sf = rand(1:n_backw, n_scenarios)
        for (i, pm) in enumerate(par_models)
            if pm.series_with_zeros
                scenarios[t_scen_idx, i, :, :] .= zero(Float64)
                continue
            end
            # Every observation is the same on a certain stage
            if all(iszero, pm.best_AR_stage[current_stage_to_predict].fitted_y)
                scenarios[t_scen_idx, i, :, :] .= scenarios[t_scen_idx-1, i, :, :]
                continue
            end
            current_model_p = pm.best_AR_stage[current_stage_to_predict].p
            psiser = zeros(n_scenarios)
            for s_f in 1:n_scenarios
                # Evaluate the deterministic parts of the scenarios
                autorregressive_normalized = dot(
                    scenarios_normalized[t_scen_idx-1:-1:t_scen_idx-current_model_p, i, s_f],
                    pm.best_AR_stage[current_stage_to_predict].ϕ
                )
                # Calculate the noise and the parameters of the viable 3 parameter log normal
                psiser[s_f] = - (pm.μ_stage[current_stage_to_predict] / (pm.σ_stage[current_stage_to_predict] + 1e-5)) - 
                                autorregressive_normalized
            end
            psimax = maximum(psiser)
            for s_f in 1:n_scenarios
                # Evaluate the deterministic parts of the scenarios
                autorregressive_normalized = dot(
                    scenarios_normalized[t_scen_idx-1:-1:t_scen_idx-current_model_p, i, s_f],
                    pm.best_AR_stage[current_stage_to_predict].ϕ
                )
                # Calculate the noise and the parameters of the viable 3 parameter log normal
                lower_bound_log_normal = if has_global_lower_bound
                    psimax
                else
                    - (pm.μ_stage[current_stage_to_predict] / pm.σ_stage[current_stage_to_predict]) - 
                      autorregressive_normalized
                end

                lower_bound_log_normal = min(-0.01, lower_bound_log_normal)

                λ = (pm.best_AR_stage[current_stage_to_predict].var_resid/lower_bound_log_normal^2) + 1
                σ_log_normal = sqrt(log(λ))
                μ_log_normal = 0.5 * log(pm.best_AR_stage[current_stage_to_predict].var_resid/ (λ * (λ - 1)))
                ruido = exp(ruido_correlacionado[noise_index_sf[s_f], i] * σ_log_normal + μ_log_normal) + lower_bound_log_normal
                # Evaluate the scenario value
                scenarios_normalized[t_scen_idx, i, s_f] = autorregressive_normalized + ruido
                for s_b in 1:n_backw
                    _s_b = s_b
                    if _s_b == 1
                        _s_b = noise_index_sf[s_f]
                    elseif _s_b == noise_index_sf[s_f]
                        _s_b = 1
                    end
                    ruido = exp(ruido_correlacionado[_s_b, i] * σ_log_normal + μ_log_normal) + lower_bound_log_normal
                    scenarios_normalized_back = autorregressive_normalized + ruido
                    scenarios[t_scen_idx, i, s_f, s_b] =
                        scenarios_normalized_back * pm.σ_stage[current_stage_to_predict] +
                            pm.μ_stage[current_stage_to_predict]
                    if scenarios[t_scen_idx, i, s_f, s_b] <= 0
                        scenarios[t_scen_idx, i, s_f, s_b] =
                            pm.σ_stage[current_stage_to_predict] * (0.04 + 0.01 * rand())
                    end
                end
            end
        end
    end
    return scenarios[p_lim+1:end, :, :, :], scenarios[1:p_lim, :,  1, 1]
end
