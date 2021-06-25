@testset "test gaussian likelihood" begin
  # Set random seed for reproducibility.
  Random.seed!(123)

  # Number of observations.
  N = 50000

  # Predictors. Required to be two-dimensional.
  X = rand(N, 1) * 2

  # True function.
  f(x::Real) = 3 * exp(-0.5 * x) * sin(6 * x)

  # True observation error.
  sigma_true = 0.1

  # Responses with random noise. Expected to be univariate.
  Y = f.(X[:, 1]) + randn(N) * sigma_true

  # Initialize model.
  num_inducing_points = 10  # number of inducing points
  model = HSVGP.SVGP_obj(X, Y, num_inducing_points)

  # Fit model.
  opt_trace, p_traces = HSVGP.fit_svgp!(model, n_iters=20000, batch_size=20);  

  # Locations at which to precit.
  Xnew = hcat(range(0, step=0.05, stop=2))

  # True predictions (without noise).
  Ynew_true = f.(Xnew[:, 1])

  # Get posterior predictive means and SDs.
  predY, predSD = HSVGP.pred_vgp(Xnew, model)

  # Get observation error.
  err_sigma = exp(model.params.log_sigma[1])  

  # Assert that the predictions are close to the truth.
  @test rmse(predY, Ynew_true) < 0.15

  # Assert that the observation error term is close to the truth.
  @test abs(err_sigma - sigma_true) < 0.05
end
