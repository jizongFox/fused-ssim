import torch

from fused_ssim import fused_ssim

torch.manual_seed(0)
image1 = torch.rand([1, 3, 1080, 1920], device="cuda", requires_grad=True)
image2 = torch.rand([1, 3, 1080, 1920], device="cuda", requires_grad=True)

loss = fused_ssim(image1, image2)
print(loss)
loss.backward()
image_1_grad = image1.grad.clone()
image_2_grad = image2.grad.clone()

image1.grad.zero_()
image2.grad.zero_()
loss = fused_ssim(image2, image1)
print(loss)
loss.backward()
image_1_grad2 = image1.grad.clone()
image_2_grad2 = image2.grad.clone()

print("image1 grad norm:", image_1_grad.norm())
print("image2 grad norm:", image_2_grad.norm())
print("image1 grad2 norm:", image_1_grad2.norm())
print("image2 grad2 norm:", image_2_grad2.norm())
