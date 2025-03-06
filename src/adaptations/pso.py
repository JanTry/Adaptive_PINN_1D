from typing import Callable

import torch
from src.adaptations.adaptation_interface import AdaptationInterface
from src.adaptations.de import mirror_bounds


class PSOAdaptation(AdaptationInterface):
    def __init__(
        self,
        max_iterations: int = 1,
        omega: float = 0.5,  # Inertia weight
        phi_p: float = 1.5,  # Cognitive parameter
        phi_g: float = 1.5,  # Social parameter
    ) -> None:
        self.max_iterations = max_iterations
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = old_x.detach().clone()[1:-1]
        x = x.requires_grad_(True)

        population_size, number_of_dimensions = x.shape
        device = x.device

        # Initialize particles' positions and velocities
        velocities = torch.zeros_like(x, device=device)
        personal_best_positions = x.clone()
        personal_best_scores = loss_function(x).abs().view(-1)

        global_best_position = personal_best_positions[
            torch.argmax(personal_best_scores)
        ].clone()
        global_best_score = personal_best_scores.max()

        for _ in range(self.max_iterations):
            # Update velocities
            r_p = torch.rand((population_size, number_of_dimensions), device=device)
            r_g = torch.rand((population_size, number_of_dimensions), device=device)

            velocities = (
                self.omega * velocities
                + self.phi_p * r_p * (personal_best_positions - x)
                + self.phi_g * r_g * (global_best_position - x)
            )
            # Update positions
            x = x + velocities
            x = mirror_bounds(x, self.x_range[0], self.x_range[1])

            # Evaluate fitness
            current_scores = loss_function(x).abs().view(-1)

            # Update personal bests
            better_scores = current_scores > personal_best_scores
            personal_best_positions = torch.where(
                better_scores.unsqueeze(-1), x, personal_best_positions
            )
            personal_best_scores = torch.where(
                better_scores, current_scores, personal_best_scores
            )

            # Update global best
            best_particle_idx = torch.argmax(personal_best_scores)
            if personal_best_scores[best_particle_idx] > global_best_score:
                global_best_position = personal_best_positions[
                    best_particle_idx
                ].clone()
                global_best_score = personal_best_scores[best_particle_idx]

        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort(dim=0)[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return "pso" if self.max_iterations == 1 else f"pso_{self.max_iterations}"
