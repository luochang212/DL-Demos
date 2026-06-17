import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from chapters.cv.nms.iou import iou
from chapters.cv.nms.nms import nms
from chapters.cv.style_transfer.style_transfer import (
    get_model_and_losses,
    gram,
    run_style_transfer,
)
from chapters.fundamentals.logistic_regression.main import loss as binary_cross_entropy
from chapters.fundamentals.multiclass_classification.pt_main import (
    MulticlassClassificationNet,
)
from chapters.generative_models.vae.code.main import load_model as load_vae_model
from chapters.generative_models.vae.code.main import loss_fn as vae_loss
from chapters.generative_models.vae.code.model import VAE
from chapters.sequence_models.attention.main import AttentionModel, sequence_accuracy
from chapters.sequence_models.basic_rnn.main import load_model as load_rnn_model
from chapters.sequence_models.basic_rnn.models import RNN1, RNN2
from chapters.sequence_models.transformer.data_load import (
    create_data,
    encode_source,
    get_batch_indices,
)
from chapters.sequence_models.transformer.translate import greedy_decode
from chapters.training_tricks.advanced_optimizer.model import (
    DeepNetwork as OptimizerNetwork,
)
from chapters.training_tricks.advanced_optimizer.optimizer import (
    Adam,
    Momentum,
    RMSProp,
)
from chapters.training_tricks.initialization.main import (
    DeepNetwork as InitializationNetwork,
)
from chapters.training_tricks.regularization.main import (
    DeepNetwork as RegularizationNetwork,
)


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

    def test_numpy_tutorial_losses_are_finite_at_probability_limits(self):
        target = np.array([[0.0, 1.0]])
        prediction = np.array([[0.0, 1.0]])
        models = [
            InitializationNetwork([1, 1], []),
            RegularizationNetwork([1, 1], []),
            OptimizerNetwork([1, 1], ['sigmoid']),
        ]

        for model in models:
            with self.subTest(model=type(model).__module__):
                self.assertTrue(np.isfinite(model.loss(target, prediction)))


class AdvancedOptimizerTest(unittest.TestCase):
    def test_stateful_optimizers_work_with_default_constructor(self):
        for optimizer_type in [Momentum, RMSProp, Adam]:
            params = {'weight': np.array([1.0])}
            optimizer = optimizer_type(params, learning_rate=0.1)
            optimizer.add_grad({'weight': np.array([1.0])})

            optimizer.step()

            self.assertLess(params['weight'][0], 1.0)

    def test_stateful_optimizer_state_can_be_restored(self):
        params = {'weight': np.array([1.0])}
        optimizer = Adam(params, learning_rate=0.1)
        optimizer.add_grad({'weight': np.array([1.0])})
        optimizer.step()

        restored = Adam({'weight': params['weight'].copy()}, learning_rate=0.1)
        restored.load(optimizer.save())

        self.assertEqual(restored.epoch, optimizer.epoch)
        np.testing.assert_array_equal(
            restored.v_dict['weight'], optimizer.v_dict['weight']
        )
        np.testing.assert_array_equal(
            restored.s_dict['weight'], optimizer.s_dict['weight']
        )


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

    def test_sequence_accuracy_does_not_allow_errors_to_cancel(self):
        prediction = torch.tensor([[1, 3], [1, 2]])
        target = torch.tensor([[2, 2], [1, 2]])

        self.assertEqual(sequence_accuracy(prediction, target).item(), 0.5)


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
        cn_vocab = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3, '浣犲ソ': 5}
        with (
            patch(
                'chapters.sequence_models.transformer.data_load.load_en_vocab',
                return_value=(en_vocab, {}),
            ),
            patch(
                'chapters.sequence_models.transformer.data_load.load_cn_vocab',
                return_value=(cn_vocab, {}),
            ),
        ):
            source, target, _, _ = create_data(['hello'], ['浣犲ソ'])

        np.testing.assert_array_equal(source[0, :3], [2, 4, 3])
        np.testing.assert_array_equal(target[0, :3], [2, 5, 3])

    def test_encode_source_adds_boundaries_and_maps_unknown_words(self):
        vocab = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3, 'hello': 4}

        self.assertEqual(encode_source(['hello', 'missing'], vocab), [2, 4, 1, 3])

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


class SequenceAndGenerativeSmokeTest(unittest.TestCase):
    def test_rnn_models_forward_and_sample_on_cpu(self):
        with patch('numpy.random.choice', return_value=' '):
            rnn1 = RNN1(hidden_units=4)
            output1 = rnn1(torch.zeros(2, 3, 27))
            sample1 = rnn1.sample_word()

            rnn2 = RNN2(hidden_units=4, embeding_dim=4, dropout_rate=0)
            output2 = rnn2(torch.zeros(2, 3, dtype=torch.long))
            sample2 = rnn2.sample_word()

        self.assertEqual(output1.shape, (2, 3, 27))
        self.assertEqual(output2.shape, (2, 3, 27))
        self.assertEqual(sample1, ' ')
        self.assertEqual(sample2, ' ')

    def test_rnn_checkpoint_loads_on_cpu(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'rnn1.pth'
            torch.save(RNN1().state_dict(), checkpoint)

            model = load_rnn_model('rnn1', checkpoint, torch.device('cpu'))

        self.assertEqual(next(model.parameters()).device.type, 'cpu')

    def test_vae_missing_checkpoint_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Checkpoint not found'):
            load_vae_model('missing.pth', torch.device('cpu'))

    def test_vae_forward_loss_and_sample_on_cpu(self):
        model = VAE(hiddens=[2, 4], latent_dim=3).eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            output, mean, logvar = model(x)
            loss = vae_loss(x, output, mean, logvar)
            sample = model.sample()

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(sample.shape, x.shape)
        self.assertTrue(torch.isfinite(loss))


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
                'chapters.cv.style_transfer.style_transfer.read_image',
                side_effect=[content, style],
            ),
            patch(
                'chapters.cv.style_transfer.style_transfer.get_model_and_losses',
                return_value=losses,
            ),
            patch(
                'chapters.cv.style_transfer.style_transfer.optim.LBFGS',
                OneStepOptimizer,
            ),
            patch('chapters.cv.style_transfer.style_transfer.save_image') as save_image,
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
