import torch
from ops import ref_ops, opt_ops


def test(dtype, device):
    print(f"Testing dtype {dtype} and device {device}")
    a = torch.rand((100, 5), dtype=dtype, device=device)
    b = torch.rand((100, 30), dtype=dtype, device=device)
    indices = torch.sort(torch.randint(10, (100,), device=device))[0]
    out_ref = ref_ops(a, b, indices, 10)
    out_opt = opt_ops(a, b, indices, 10)
    print(out_opt.shape)
    assert torch.allclose(out_ref, out_opt)
    print("Assertion passed successfully!")


if __name__ == "__main__":
    test(torch.float32, "cpu")
    test(torch.float64, "cpu")
    #if torch.cuda.is_available():
    #    test(torch.float32, "cuda")
    #    test(torch.float64, "cuda")
