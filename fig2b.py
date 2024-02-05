import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


seed = 0
rng = np.random.default_rng(seed)

mu = 0.01
batch = 100

ni = 30
nh = 20
no = 10

T = rng.uniform(-1, 1, [no,ni])

s_bp = s_fa = rng.uniform(-0.01, 0.01, [nh,ni])
w_bp = w_fa = rng.uniform(-0.01, 0.01, [no,nh])
b = rng.uniform(-0.5, 0.5, [nh,no])

params_bp = [s_bp, w_bp]
params_fa = [s_fa, w_fa]


@jax.jit
def predict(params, x):
    s, w = params
    h = jnp.matmul(s, x)
    z = jnp.matmul(w, h)
    return z

@jax.jit
def loss_fn(params, x, y):
    z = predict(params, x)
    tmp = 1/2 * jnp.sum((y-z)*(y-z), axis=1)
    loss = jnp.mean(tmp)
    return loss

@jax.jit
def calc_cos(bp, fa):
    bp = jnp.mean(bp, axis=0)
    fa = jnp.mean(fa, axis=0)
    tmp = jnp.matmul(bp.T, fa)
    bunshi = jnp.linalg.norm(tmp)
    tmp_a = jnp.linalg.norm(bp)
    tmp_b = jnp.linalg.norm(fa)
    bunbo = tmp_a * tmp_b
    cos = bunshi / bunbo
    return cos


grad_loss_fn = jax.grad(loss_fn, argnums=0)
grad_loss_fn_jit = jax.jit(grad_loss_fn)

cos_list = []

""" モデルの学習 """
epoch = 2000
for i in range(epoch+1):
    """ predict """
    x = rng.normal(0, 1, [batch, ni,1])
    y = jnp.matmul(T, x)

    z_bp = predict(params_bp, x)
    z_fa = predict(params_fa, x)

    """ calc grads """
    grads_bp = grad_loss_fn_jit(params_bp, x, y)
    grads_fa = grad_loss_fn_jit(params_fa, x, y)

    e = y - z_fa
    
    # backpropagation
    delta_bp = jnp.matmul(params_bp[1].T, e)
    
    # feedback alignment
    delta_fa = jnp.matmul(b, e)

    # calc cos theta
    cos = calc_cos(delta_bp, delta_fa)
    cos_list.append(cos)

    tmp_x = jnp.transpose(x, [0, 2, 1])
    tmp_ds = jnp.matmul(delta_fa, tmp_x)
    ds = jnp.mean(tmp_ds, axis=0)

    """ update params """    
    params_bp[0] = params_bp[0] - mu * grads_bp[0]
    params_fa[0] = params_fa[0] - mu * ds

    params_bp[1] = params_bp[1] - mu * grads_bp[1]
    params_fa[1] = params_fa[1] - mu * grads_fa[1]

plt.figure()
plt.ylim(0, 1)
plt.plot(cos_list)
plt.show()
