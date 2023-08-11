import torch
torch.manual_seed(0)
from ops import ref_ops, opt_ops


def test(dtype, device):
    print(f"Testing dtype {dtype} and device {device}")

    a_ref = torch.rand((100, 20), dtype=dtype, device=device, requires_grad=True)
    b_ref = torch.rand((100, 5), dtype=dtype, device=device, requires_grad=True)
    a_opt = a_ref.clone().detach().requires_grad_(True)
    b_opt = b_ref.clone().detach().requires_grad_(True)
    indices = torch.sort(torch.randint(10, (100,), device=device))[0]
    indices[torch.where(indices==1)[0]] = 2  # substitute all 1s by 2s so as to test the no-neighbor case
    out_ref = ref_ops(a_ref, b_ref, indices, 10)
    out_opt = opt_ops(a_opt, b_opt, indices, 10)
    assert torch.allclose(out_ref, out_opt)

    loss_ref = torch.sum(out_ref)
    loss_ref.backward()
    loss_opt = torch.sum(out_opt)
    loss_opt.backward()
    assert torch.allclose(a_ref.grad, a_opt.grad)
    assert torch.allclose(b_ref.grad, b_opt.grad)

    print("Assertions passed successfully!")


if __name__ == "__main__":
    test(torch.float32, "cpu")
    test(torch.float64, "cpu")
    #if torch.cuda.is_available():
    #    test(torch.float32, "cuda")
    #    test(torch.float64, "cuda")
