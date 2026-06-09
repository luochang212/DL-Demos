import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from dldemos.attention.main import AttentionModel
from dldemos.LogisticRegression.main import loss as binary_cross_entropy
from dldemos.MulticlassClassification.pt_main import MulticlassClassificationNet
from dldemos.nms.iou import iou
from dldemos.nms.nms import nms
from dldemos.StyleTransfer.style_transfer import (
    get_model_and_losses,
    gram,
    run_style_transfer,
)
from dldemos.Transformer.data_load import create_data, get_batch_indices
from dldemos.Transformer.translate import greedy_decode


class BasicNetworkTest(unittest.TestCase):
    def test_binary_cross_entropy_is_finite_at_probability_limits(self):
        result = binary_cross_entropy(np.array([[0.0, 1.0]]), np.array([[0, 1]]))
        self.assertTrue(np.isfinite(result))

    def test_multiclass_model_returns_logits(self):
        torch.manual_seed(0)
        model = MulticlassClassificationNet([2, 3])
        logits = model.forward(torch.ones(2, 4))

        self.assertFalse(torch.allclose(logits.sum(dim=0), torch.ones(4)))
        loss = model.loss(torch.tensor([0, 1, 2, 0]), logits)
        self.assertTrue(torch.isfinite(loss))


class AttentionTest(unittest.TestCase):
    def test_padding_does_not_change_attention_output(self):
        torch.manual_seed(0)
        model = AttentionModel(dropout_rate=0).eval()
        short = torch.tensor([[65, 66, 67]])
        padded_batch = torch.tensor([[65, 66, 67, 0, 0], [68, 69, 70, 71, 72]])

        with torch.no_grad():
            expected = model(short, torch.tensor([3]), n_output=2)
            actual = model(padded_batch, torch.tensor([3, 5]), n_output=2)[0:1]

        torch.testing.assert_close(actual, expected)


class EndTokenModel(nn.Module):
    def __init__(self, vocab_size, end_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.end_id = end_id

    def forward(self, x, y):
        logits = torch.zeros(x.shape[0], y.shape[1], self.vocab_size)
        logits[:, -1, self.end_id] = 1
        return logits


class TransformerTest(unittest.TestCase):
    def test_create_data_uses_english_source_and_chinese_target_vocabularies(self):
        en_vocab = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3, 'hello': 4}
        cn_vocab = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3, '你好': 5}
        with (
            patch(
                'dldemos.Transformer.data_load.load_en_vocab',
                return_value=(en_vocab, {}),
            ),
            patch(
                'dldemos.Transformer.data_load.load_cn_vocab',
                return_value=(cn_vocab, {}),
            ),
        ):
            source, target, _, _ = create_data(['hello'], ['你好'])

        np.testing.assert_array_equal(source[0, :3], [2, 4, 3])
        np.testing.assert_array_equal(target[0, :3], [2, 5, 3])

    def test_batch_indices_include_every_sample_once(self):
        torch.manual_seed(0)
        batches = [batch for batch, _ in get_batch_indices(10, 4)]
        flattened = [index for batch in batches for index in batch]

        self.assertEqual(len(batches), 3)
        self.assertEqual(sorted(flattened), list(range(10)))

    def test_greedy_decode_starts_and_stops_with_target_tokens(self):
        model = EndTokenModel(vocab_size=8, end_id=3)
        result = greedy_decode(
            model, torch.tensor([[4, 5]]), start_id=2, end_id=3, max_len=6
        )

        torch.testing.assert_close(result, torch.tensor([[2, 3]]))


class StyleTransferTest(unittest.TestCase):
    def test_gram_matrix_shape_and_symmetry(self):
        result = gram(torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4))

        self.assertEqual(result.shape, (3, 3))
        torch.testing.assert_close(result, result.T)

    def test_loss_model_can_use_a_small_injected_cnn(self):
        cnn = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        content = torch.rand(1, 3, 4, 4)
        style = torch.rand(1, 3, 4, 4)

        model, content_losses, style_losses = get_model_and_losses(
            content,
            style,
            content_layers=['conv_1'],
            style_layers=['conv_1'],
            cnn=cnn,
        )
        model(content)

        self.assertEqual(len(content_losses), 1)
        self.assertEqual(len(style_losses), 1)

    def test_short_style_transfer_flow_uses_explicit_entrypoint(self):
        cnn = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        content = torch.rand(1, 3, 4, 4)
        style = torch.rand(1, 3, 4, 4)
        losses = get_model_and_losses(
            content,
            style,
            content_layers=['conv_1'],
            style_layers=['conv_1'],
            cnn=cnn,
        )

        class OneStepOptimizer:
            def __init__(self, _):
                pass

            def zero_grad(self):
                pass

            def step(self, closure):
                closure()

        with (
            patch(
                'dldemos.StyleTransfer.style_transfer.read_image',
                side_effect=[content, style],
            ),
            patch(
                'dldemos.StyleTransfer.style_transfer.get_model_and_losses',
                return_value=losses,
            ),
            patch('dldemos.StyleTransfer.style_transfer.optim.LBFGS', OneStepOptimizer),
            patch('dldemos.StyleTransfer.style_transfer.save_image') as save_image,
        ):
            result = run_style_transfer('content', 'style', 'output', num_steps=1)

        self.assertEqual(result.shape, content.shape)
        save_image.assert_called_once()


class DetectionPostprocessingTest(unittest.TestCase):
    def test_iou_handles_overlap_disjoint_and_zero_area_boxes(self):
        self.assertEqual(iou((0, 0, 2, 2), (0, 0, 2, 2)), 1)
        self.assertEqual(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0)
        self.assertEqual(iou((0, 0, 0, 0), (0, 0, 0, 0)), 0)

    def test_nms_filters_low_scores_and_suppresses_overlapping_boxes(self):
        predicts = np.array(
            [
                [0.9, 0, 0, 10, 10],
                [0.8, 1, 1, 9, 9],
                [0.7, 20, 20, 30, 30],
                [0.2, 40, 40, 50, 50],
            ]
        )

        filtered, indices = nms(predicts, score_thresh=0.5, iou_thresh=0.5)

        self.assertEqual(indices, [0, 2])
        np.testing.assert_array_equal(filtered, predicts[[0, 2]])


if __name__ == '__main__':
    unittest.main()
