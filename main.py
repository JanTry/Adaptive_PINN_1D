from src.enums.adaptations import EAdaptations
from src.enums.problems import EProblems
from src.plots.plot_specific_run import plot_specific_run
from src.runners.adaptive_PINN import train_PINN


train_PINN(0, problem_type=EProblems.DIFFUSION, adaptation_type=EAdaptations.DENSITY_SAMPLING)
train_PINN(0, problem_type=EProblems.DIFFUSION, adaptation_type=EAdaptations.MIDDLE_POINT)
train_PINN(0, problem_type=EProblems.DIFFUSION, adaptation_type=EAdaptations.NO_ADAPTATION)
train_PINN(0, problem_type=EProblems.P07_01, adaptation_type=EAdaptations.DENSITY_SAMPLING)
train_PINN(0, problem_type=EProblems.P07_01, adaptation_type=EAdaptations.MIDDLE_POINT)
train_PINN(0, problem_type=EProblems.P07_01, adaptation_type=EAdaptations.NO_ADAPTATION)
train_PINN(0, problem_type=EProblems.TAN_03, adaptation_type=EAdaptations.DENSITY_SAMPLING)
train_PINN(0, problem_type=EProblems.TAN_03, adaptation_type=EAdaptations.MIDDLE_POINT)
train_PINN(0, problem_type=EProblems.TAN_03, adaptation_type=EAdaptations.NO_ADAPTATION)

plot_specific_run(run_id=0, problem_type=EProblems.DIFFUSION, adaptation=EAdaptations.MIDDLE_POINT, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.DIFFUSION, adaptation=EAdaptations.DENSITY_SAMPLING, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.DIFFUSION, adaptation=EAdaptations.NO_ADAPTATION, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.P07_01, adaptation=EAdaptations.MIDDLE_POINT, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.P07_01, adaptation=EAdaptations.DENSITY_SAMPLING, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.P07_01, adaptation=EAdaptations.NO_ADAPTATION, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.TAN_03, adaptation=EAdaptations.MIDDLE_POINT, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.TAN_03, adaptation=EAdaptations.DENSITY_SAMPLING, plot_training_points=True)
plot_specific_run(run_id=0, problem_type=EProblems.TAN_03, adaptation=EAdaptations.NO_ADAPTATION, plot_training_points=True)