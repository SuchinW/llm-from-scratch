import torch

from src.models.gpt import GPTModel, TransformerBlock


# ----- TransformerBlock -----

def test_block_output_shape_matches_input():
    B, N, d_model = 2, 8, 16
    block = TransformerBlock(d_model=d_model, n_head=4, context_len=N)
    x = torch.randn(B, N, d_model)

    out = block(x)

    assert out.shape == (B, N, d_model)


def test_block_is_residual_at_init_with_zero_init_weights():
    B, N, d_model = 2, 4, 8
    block = TransformerBlock(d_model=d_model, n_head=2, context_len=N)
    for p in block.parameters():
        torch.nn.init.zeros_(p)
    block.ln1.scale.data.fill_(1.0)
    block.ln2.scale.data.fill_(1.0)

    x = torch.randn(B, N, d_model)
    out = block(x)

    assert torch.allclose(out, x, atol=1e-5)


def test_block_gradients_flow():
    B, N, d_model = 2, 4, 8
    block = TransformerBlock(d_model=d_model, n_head=2, context_len=N)
    x = torch.randn(B, N, d_model, requires_grad=True)

    block(x).sum().backward()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} has no grad"


# ----- GPTModel -----

def _small_model(**overrides):
    cfg = dict(vocab_size=100, d_model=16, n_head=4, context_len=32, n_blocks=2)
    cfg.update(overrides)
    return GPTModel(**cfg)


def test_logits_shape():
    model = _small_model()
    idx = torch.randint(0, 100, (2, 8))

    logits = model(idx)

    assert logits.shape == (2, 8, 100)


def test_handles_single_token_sequence():
    model = _small_model()
    idx = torch.randint(0, 100, (1, 1))

    logits = model(idx)

    assert logits.shape == (1, 1, 100)


def test_handles_full_context_length():
    model = _small_model(context_len=16)
    idx = torch.randint(0, 100, (2, 16))

    logits = model(idx)

    assert logits.shape == (2, 16, 100)


def test_position_independent_at_token_position():
    """Same token at different positions should yield different logits
    (otherwise positional embedding is being ignored)."""
    torch.manual_seed(0)
    model = _small_model()
    model.eval()

    idx = torch.zeros((1, 4), dtype=torch.long)
    logits = model(idx)

    assert not torch.allclose(logits[0, 0], logits[0, 3])


def test_output_is_finite():
    model = _small_model()
    idx = torch.randint(0, 100, (2, 16))

    logits = model(idx)

    assert torch.all(torch.isfinite(logits))


def test_gradients_flow_to_all_parameters():
    model = _small_model()
    idx = torch.randint(0, 100, (2, 8))

    model(idx).sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"


def test_eval_mode_is_deterministic_with_dropout():
    model = _small_model(dropout=0.5)
    model.eval()
    idx = torch.randint(0, 100, (2, 8))

    a = model(idx)
    b = model(idx)

    assert torch.allclose(a, b)


def test_train_mode_dropout_introduces_variance():
    torch.manual_seed(0)
    model = _small_model(dropout=0.5)
    model.train()
    idx = torch.randint(0, 100, (2, 8))

    a = model(idx)
    b = model(idx)

    assert not torch.allclose(a, b)


def test_block_count_matches_n_blocks():
    model = _small_model(n_blocks=5)

    assert len(model.trn_blocks) == 5


def test_runs_on_cpu_device():
    model = _small_model().to("cpu")
    idx = torch.randint(0, 100, (1, 4), device="cpu")

    logits = model(idx)

    assert logits.device.type == "cpu"
