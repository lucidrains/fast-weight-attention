import torch
from einops import einsum, reduce
import torch.nn.functional as F

b, h, n, d = 2, 4, 10, 16

q = torch.randn(b, h, n, d, requires_grad=True)
k = torch.randn(b, h, n, d, requires_grad=True)
v = torch.randn(b, h, n, d, requires_grad=True)
gates = torch.rand(b, h, n, 1, requires_grad=True)
wo = torch.randn(h, d, d)

# Forward pass
scale = d ** -0.5
score = einsum(q, k, 'b h i dh, b h j dh -> b h i j') * scale
attn = score.softmax(dim=-1)

u = einsum(attn, v, 'b h i j, b h j dh -> b h i dh')
out = u * gates
pred_values = einsum(out, wo, 'b h n dh, h dh d -> b n d')

# Dummy loss
target = torch.randn(b, n, d)
loss = 0.5 * ((pred_values - target) ** 2).sum()

loss.backward()

# Fetch autograd gradients
grad_q_auto = q.grad
grad_k_auto = k.grad
grad_v_auto = v.grad

# Manual backward pass
# 1. Error
error = (pred_values - target) # shape (b, n, d)

# 2. dout
dout = einsum(error, wo, 'b n d, h dh d -> b h n dh')

# 3. du
du = dout * gates

# 4. delta
delta = reduce(dout * out, '... d -> ... 1', 'sum')

# 5. dattn, dv
dv = einsum(attn, du, 'b h i j, b h i dh -> b h j dh')
dattn = einsum(v, du, 'b h j dh, b h i dh -> b h i j')

# 6. dscore
dscore = scale * attn * (dattn - delta)

# 7. dq, dk
dq = einsum(k, dscore, 'b h j dh, b h i j -> b h i dh')
dk = einsum(q, dscore, 'b h i dh, b h i j -> b h j dh')


print(f"Max diff dv: {(dv - grad_v_auto).abs().max().item()}")
print(f"Max diff dq: {(dq - grad_q_auto).abs().max().item()}")
print(f"Max diff dk: {(dk - grad_k_auto).abs().max().item()}")
