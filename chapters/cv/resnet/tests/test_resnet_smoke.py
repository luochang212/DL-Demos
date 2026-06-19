import pytest

pytest.importorskip('tensorflow', reason='TensorFlow not installed')
tf = pytest.importorskip('tensorflow')


def test_identity_block_forward_on_cpu():
    from chapters.cv.resnet.code.tf_main import identity_block_2

    x = tf.random.normal((2, 56, 56, 64))
    out = identity_block_2(x, f=3, use_shortcut=True)
    assert out.shape == x.shape

    out_no_skip = identity_block_2(x, f=3, use_shortcut=False)
    assert out_no_skip.shape == x.shape


def test_convolution_block_forward_on_cpu():
    from chapters.cv.resnet.code.tf_main import convolution_block_2

    x = tf.random.normal((2, 56, 56, 64))
    out = convolution_block_2(x, f=3, filters=128, s=2, use_shortcut=True)
    assert out.shape == (2, 28, 28, 128)

    out_no_skip = convolution_block_2(x, f=3, filters=128, s=2, use_shortcut=False)
    assert out_no_skip.shape == (2, 28, 28, 128)


def test_init_model_resnet18_on_cpu():
    from chapters.cv.resnet.code.tf_main import init_model

    model = init_model(
        input_shape=(64, 64, 3), model_name='ResNet18', use_shortcut=True
    )
    x = tf.random.normal((1, 64, 64, 3))
    out = model(x, training=False)
    assert out.shape == (1, 1)


def test_init_model_resnet50_on_cpu():
    from chapters.cv.resnet.code.tf_main import init_model

    model = init_model(
        input_shape=(64, 64, 3), model_name='ResNet50', use_shortcut=True
    )
    x = tf.random.normal((1, 64, 64, 3))
    out = model(x, training=False)
    assert out.shape == (1, 1)
