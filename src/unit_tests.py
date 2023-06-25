# Description: Unit tests for the data module.
import torch

def check_for_causal(dataloader):
    for batch_idx, (x, y) in enumerate(dataloader):
        try:
            assert x.shape == y.shape
        except AssertionError:
            print(f'Assertion failed on batch {batch_idx}:')
            print(f'x.shape = {x.shape}, y.shape = {y.shape}')            
            return

        try:
            assert torch.allclose(x[:, 1:], y[:, :-1])  # y should be a shifted version of x to the right
        except AssertionError:
            print(f'Assertion failed on batch {batch_idx}: x[:, 1:] and y[:, :-1] are not close.')
            print(f'x[:, 1:] = {x[:, 1:]}, y[:, :-1] = {y[:, :-1]}')
            return

    print("All assertions passed.")

def test_transformer_block():
    tb = TransformerBlock(d_model=64, num_heads=8)
    v = torch.randn(50, 10, 64)  # batch size=50, sequence length=10, d_model=64
    k = torch.randn(50, 10, 64)
    q = torch.randn(50, 10, 64)

    # Use the forward method
    output = tb(v, k, q, mask=None)

    # Check that the output has the expected shape
    assert output.shape == (50, 10, 64), "Output shape doesn't match"

    print("All tests passed.")

def test_multi_head_attention():
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    v = torch.randn(50, 10, 64)  # batch size=50, sequence length=10, d_model=64
    k = torch.randn(50, 10, 64)
    q = torch.randn(50, 10, 64)

    # Use the forward method
    output, weights = mha(v, k, q, mask=None)

    # Check that the output has the expected shape
    assert output.shape == (50, 10, 64), "Output shape doesn't match"
    assert weights.shape == (50, 8, 10, 10), "Weights shape doesn't match"

    print("All tests passed.")


def test_scaled_dot_product_attention():
    attention = ScaledDotProductAttention()
    query = torch.randn(50, 10, 64)  # batch size=50, sequence length=10, d_model=64
    key = torch.randn(50, 10, 64)
    value = torch.randn(50, 10, 64)

    # Use the forward method
    output, weights = attention(query, key, value)

    # Check that the output has the expected shape
    assert output.shape == (50, 10, 64), "Output shape doesn't match"
    assert weights.shape == (50, 10, 10), "Weights shape doesn't match"

    print("All tests passed.")
