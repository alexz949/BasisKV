"""Small CPU checks against the defining objectives and representation identities."""
import itertools
import unittest

import torch

from core.als import quadratic_from_target, solve_decoder, evaluate_loss, encoder_hessian_vector_product
from core.allocate import allocate, predicted_costs, sensitivities
from core.page_fisher import residual_fisher_gram
from core.router import align_value_basis, fit_affine_map


class AlgorithmTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_joint_decoder_matches_stacked_least_squares(self):
        z = torch.randn(40, 4, 5, dtype=torch.float64)
        weight = torch.randn(4, 5, 7, dtype=torch.float64)
        mapping = torch.tensor([0, 0, 1, 1])
        encoder = torch.randn(2, 5, 3, dtype=torch.float64)
        covariance = torch.einsum('nhi,nkj->hkij', z, z)
        objective = quadratic_from_target(covariance, weight)
        decoder, _ = solve_decoder(objective, encoder, mapping)
        target = torch.einsum('nhd,hdo->no', z, weight)
        latent = torch.einsum('nhd,hdr->nhr', z, encoder[mapping]).flatten(1)
        expected = torch.linalg.lstsq(latent, target).solution
        torch.testing.assert_close(decoder.flatten(0, 1), expected)
        direct_loss = (target - latent @ expected).square().sum()
        self.assertAlmostEqual(evaluate_loss(objective, encoder, decoder, mapping)[0], float(direct_loss), places=8)

    def test_shared_encoder_hessian_matches_autograd(self):
        z = torch.randn(20, 4, 5, dtype=torch.float64)
        decoder = torch.randn(4, 3, 7, dtype=torch.float64)
        covariance = torch.einsum('nhi,nkj->hkij', z, z)
        heads = torch.tensor([0, 1])
        e = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        delta = torch.randn_like(e)
        output = torch.einsum('nhi,ir,hro->no', z[:, heads], e, decoder[heads])
        gradient, = torch.autograd.grad(output.square().sum() / 2, e, create_graph=True)
        hessian, = torch.autograd.grad((gradient * delta).sum(), e)
        torch.testing.assert_close(encoder_hessian_vector_product(covariance, decoder, heads, delta), hessian)

    def test_rank_allocation_matches_exhaustive_search(self):
        curves = {layer: {2: 1.5 + layer, 4: 0.5 + layer, 6: 0.1 + layer} for layer in range(4)}
        probes = {layer: dict(delta_low=0.1 * (layer + 1), delta_high=-0.05 * (layer + 1)) for layer in curves}
        costs = predicted_costs(curves, sensitivities(curves, probes, anchor=4, low=2, high=6), 4)
        schedule, loss = allocate(costs, 16)
        brute = min(sum(costs[layer][rank] for layer, rank in enumerate(ranks))
                    for ranks in itertools.product((2, 4, 6), repeat=4) if sum(ranks) == 16)
        self.assertEqual(sum(schedule.values()), 16)
        self.assertAlmostEqual(loss, brute)

    def test_affine_base_recovers_low_rank_predictor(self):
        x = torch.randn(80, 6, dtype=torch.float64) + 2
        weight = torch.randn(6, 2, dtype=torch.float64) @ torch.randn(2, 5, dtype=torch.float64)
        bias = torch.randn(5, dtype=torch.float64)
        y = x @ weight + bias
        fitted = fit_affine_map(row_count=len(x), input_sum=x.sum(0), target_sum=y.sum(0),
                               input_gram=x.T @ x, input_target_gram=x.T @ y, rank=2)
        torch.testing.assert_close(x @ fitted.left @ fitted.right + fitted.bias, y)

    def test_orthogonal_basis_preserves_values_and_key_prediction(self):
        for dependent in (False, True):
            with self.subTest(dependent=dependent):
                left = torch.randn(2, 6, 2, dtype=torch.float64)
                if dependent:
                    left[..., 1] = left[..., 0]
                right = torch.randn(2, 2, 5, dtype=torch.float64)
                rotation, aligned_right = align_value_basis(left, right)
                torch.testing.assert_close(rotation.mT @ rotation, torch.eye(6, dtype=rotation.dtype).expand(2, 6, 6))
                codes = torch.randn(2, 11, 6, dtype=torch.float64)
                rotated = codes @ rotation
                torch.testing.assert_close(rotated[..., :2] @ aligned_right, codes @ left @ right)
                mapping = torch.tensor([0, 0, 1, 1])
                decoder = torch.randn(4, 6, 7, dtype=torch.float64)
                torch.testing.assert_close(rotated[mapping] @ (rotation[mapping].mT @ decoder), codes[mapping] @ decoder)

    def test_page_fisher_is_page_kl_curvature(self):
        dim, page_size = 4, 3
        q = torch.randn(1, dim, dtype=torch.float64)
        keys = torch.randn(12, dim, dtype=torch.float64)
        residual = torch.randn_like(keys)
        scale = dim ** -0.5
        direction = torch.randn(dim, dim, dtype=torch.float64)
        gram, _ = residual_fisher_gram(q, keys, residual, scaling=scale, page_size=page_size, excluded_prefix_pages=0)
        logits = (scale * q @ keys.T).reshape(-1, page_size).logsumexp(-1)
        teacher_log = logits.log_softmax(-1)
        perturbation = scale * q @ direction @ residual.T
        epsilon = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        changed = (scale * q @ keys.T + epsilon * perturbation).reshape(-1, page_size).logsumexp(-1)
        kl = (teacher_log.exp() * (teacher_log - changed.log_softmax(-1))).sum()
        slope, = torch.autograd.grad(kl, epsilon, create_graph=True)
        curvature, = torch.autograd.grad(slope, epsilon)
        expected = scale ** 2 * (q @ direction @ gram[0] @ direction.T @ q.T).squeeze()
        torch.testing.assert_close(curvature, expected)


if __name__ == '__main__':
    unittest.main()
