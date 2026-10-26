"""
torchlite: a small, NumPy-backed teaching subset of the PyTorch API.

Real PyTorch can't run in a browser, so the course installs this module under
the name `torch` inside the in-browser Python. It supports the parts interviews
use: tensors with autograd, nn.Module, nn.Linear, common activations and losses,
optim.SGD / optim.Adam, Dataset and DataLoader. Code you write against it is
ordinary PyTorch and runs unchanged on the real library.
"""
import math
import random as _random
import numpy as np

float32 = np.float32
int64 = np.int64
long = np.int64
_grad_enabled = [True]


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Size(tuple):
    def numel(self):
        n = 1
        for s in self:
            n *= s
        return n

    def __repr__(self):
        return "torch.Size(" + str(list(self)) + ")"


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _back=None, dtype=None):
        arr = np.array(data, dtype=dtype) if dtype is not None else np.asarray(data)
        if dtype is None and arr.dtype.kind == "f":
            arr = arr.astype(np.float32)
        if dtype is None and arr.dtype.kind in "iu":
            arr = arr.astype(np.int64)
        self.data = arr
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = _parents
        self._back = _back

    # ----- info -----
    @property
    def shape(self):
        return Size(self.data.shape)

    def size(self, dim=None):
        return self.shape if dim is None else self.data.shape[dim]

    @property
    def ndim(self):
        return self.data.ndim

    def dim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose(-2, -1) if self.ndim >= 2 else self

    def numel(self):
        return int(self.data.size)

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def numpy(self):
        return self.data

    def detach(self):
        return Tensor(self.data.copy())

    def float(self):
        return Tensor(self.data.astype(np.float32), self.requires_grad)

    def long(self):
        return Tensor(self.data.astype(np.int64))

    def requires_grad_(self, flag=True):
        self.requires_grad = flag
        return self

    def __repr__(self):
        g = ", requires_grad=True" if self.requires_grad else ""
        return "tensor(" + np.array2string(self.data, precision=4, separator=", ") + g + ")"

    def __len__(self):
        return self.data.shape[0]

    # ----- autograd -----
    def _make(self, data, parents, back):
        rg = _grad_enabled[0] and any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=rg, _parents=parents if rg else (), _back=back if rg else None)

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad = np.ones_like(self.data, dtype=np.float32)
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)
        grads = {id(self): np.asarray(grad, dtype=np.float32)}
        for t in reversed(order):
            g = grads.pop(id(t), None)
            if g is None:
                continue
            if not t._parents:
                t.grad = Tensor(g if t.grad is None else t.grad.data + g)
                continue
            for p, pg in zip(t._parents, t._back(g)):
                if p.requires_grad and pg is not None:
                    pg = _unbroadcast(np.asarray(pg, dtype=np.float32), p.data.shape)
                    grads[id(p)] = grads[id(p)] + pg if id(p) in grads else pg

    # ----- arithmetic -----
    def __add__(self, o):
        o = _t(o)
        return self._make(self.data + o.data, (self, o), lambda g: (g, g))

    __radd__ = __add__

    def __sub__(self, o):
        o = _t(o)
        return self._make(self.data - o.data, (self, o), lambda g: (g, -g))

    def __rsub__(self, o):
        return _t(o) - self

    def __mul__(self, o):
        o = _t(o)
        return self._make(self.data * o.data, (self, o), lambda g: (g * o.data, g * self.data))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _t(o)
        return self._make(self.data / o.data, (self, o),
                          lambda g: (g / o.data, -g * self.data / (o.data ** 2)))

    def __rtruediv__(self, o):
        return _t(o) / self

    def __neg__(self):
        return self * -1

    # in-place ops keep the same tensor object (used for manual weight updates)
    def _inplace(self, data):
        if _grad_enabled[0] and self.requires_grad and not self._parents:
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation. "
                               "Wrap the update in torch.no_grad().")
        self.data = np.asarray(data, dtype=self.data.dtype)
        return self

    def __iadd__(self, o): return self._inplace(self.data + _t(o).data)
    def __isub__(self, o): return self._inplace(self.data - _t(o).data)
    def __imul__(self, o): return self._inplace(self.data * _t(o).data)
    def __itruediv__(self, o): return self._inplace(self.data / _t(o).data)
    def add_(self, o, alpha=1): return self._inplace(self.data + alpha * _t(o).data)
    def sub_(self, o, alpha=1): return self._inplace(self.data - alpha * _t(o).data)
    def mul_(self, o): return self._inplace(self.data * _t(o).data)
    def zero_(self): return self._inplace(np.zeros_like(self.data))
    def fill_(self, v): return self._inplace(np.full_like(self.data, v))

    def __pow__(self, p):
        return self._make(self.data ** p, (self,), lambda g: (g * p * self.data ** (p - 1),))

    def __matmul__(self, o):
        o = _t(o)
        a, b = self.data, o.data

        def back(g):
            ga = g @ np.swapaxes(b, -1, -2) if b.ndim > 1 else np.outer(g, b) if a.ndim > 1 else g * b
            gb = np.swapaxes(a, -1, -2) @ g if a.ndim > 1 else np.outer(a, g) if b.ndim > 1 else g * a
            return ga, gb

        return self._make(a @ b, (self, o), back)

    def matmul(self, o):
        return self @ o

    # comparisons (no grad)
    def __gt__(self, o): return Tensor(self.data > _t(o).data)
    def __lt__(self, o): return Tensor(self.data < _t(o).data)
    def __ge__(self, o): return Tensor(self.data >= _t(o).data)
    def __le__(self, o): return Tensor(self.data <= _t(o).data)
    def __eq__(self, o): return Tensor(self.data == _t(o).data)
    def __ne__(self, o): return Tensor(self.data != _t(o).data)
    __hash__ = object.__hash__

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data
        elif isinstance(idx, tuple):
            idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)

        def back(g):
            full = np.zeros_like(self.data, dtype=np.float32)
            np.add.at(full, idx, g)
            return (full,)

        return self._make(self.data[idx], (self,), back)

    # ----- reductions -----
    def sum(self, dim=None, keepdim=False):
        out = self.data.sum(axis=dim, keepdims=keepdim)

        def back(g):
            if dim is not None and not keepdim:
                g = np.expand_dims(g, dim)
            return (np.broadcast_to(g, self.data.shape),)

        return self._make(out, (self,), back)

    def mean(self, dim=None, keepdim=False):
        n = self.data.size if dim is None else self.data.shape[dim]
        return self.sum(dim, keepdim) / n

    def max(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self.data.max())
        return Tensor(self.data.max(axis=dim, keepdims=keepdim)), Tensor(self.data.argmax(axis=dim))

    def argmax(self, dim=None, keepdim=False):
        r = self.data.argmax(axis=dim)
        return Tensor(np.expand_dims(r, dim) if keepdim and dim is not None else r)

    def var(self, dim=None, keepdim=False, unbiased=True):
        m = self.mean(dim, keepdim=True)
        n = self.data.size if dim is None else self.data.shape[dim]
        return ((self - m) ** 2).sum(dim, keepdim) / (n - 1 if unbiased else n)

    def std(self, dim=None, keepdim=False, unbiased=True):
        return self.var(dim, keepdim, unbiased) ** 0.5

    # ----- elementwise -----
    def exp(self):
        e = np.exp(self.data)
        return self._make(e, (self,), lambda g: (g * e,))

    def log(self):
        return self._make(np.log(self.data), (self,), lambda g: (g / self.data,))

    def sqrt(self):
        return self ** 0.5

    def abs(self):
        return self._make(np.abs(self.data), (self,), lambda g: (g * np.sign(self.data),))

    def relu(self):
        return self._make(np.maximum(self.data, 0), (self,), lambda g: (g * (self.data > 0),))

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        return self._make(s, (self,), lambda g: (g * s * (1 - s),))

    def tanh(self):
        t = np.tanh(self.data)
        return self._make(t, (self,), lambda g: (g * (1 - t * t),))

    def clamp(self, min=None, max=None):
        out = np.clip(self.data, min, max)
        mask = np.ones_like(self.data, dtype=bool)
        if min is not None: mask &= self.data >= min
        if max is not None: mask &= self.data <= max
        return self._make(out, (self,), lambda g: (g * mask,))

    def masked_fill(self, mask, value):
        m = _t(mask).data.astype(bool)
        out = np.where(m, value, self.data).astype(np.float32)
        return self._make(out, (self,), lambda g: (np.where(m, 0, g),))

    def softmax(self, dim=-1):
        x = self.data - self.data.max(axis=dim, keepdims=True)
        e = np.exp(x)
        s = e / e.sum(axis=dim, keepdims=True)
        return self._make(s, (self,), lambda g: (s * (g - (g * s).sum(axis=dim, keepdims=True)),))

    def log_softmax(self, dim=-1):
        x = self.data - self.data.max(axis=dim, keepdims=True)
        ls = x - np.log(np.exp(x).sum(axis=dim, keepdims=True))
        sm = np.exp(ls)
        return self._make(ls, (self,), lambda g: (g - sm * g.sum(axis=dim, keepdims=True),))

    # ----- shape -----
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._make(self.data.reshape(shape), (self,), lambda g: (g.reshape(self.data.shape),))

    view = reshape

    def flatten(self, start_dim=0):
        return self.reshape(self.data.shape[:start_dim] + (-1,))

    def transpose(self, d0, d1):
        return self._make(np.swapaxes(self.data, d0, d1), (self,), lambda g: (np.swapaxes(g, d0, d1),))

    def permute(self, *dims):
        inv = np.argsort(dims)
        return self._make(np.transpose(self.data, dims), (self,), lambda g: (np.transpose(g, inv),))

    def unsqueeze(self, dim):
        return self.reshape(np.expand_dims(self.data, dim).shape)

    def squeeze(self, dim=None):
        return self.reshape(np.squeeze(self.data, axis=dim).shape)

    def contiguous(self):
        return self

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float32))


# ----- constructors -----
def tensor(data, dtype=None, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def _shape(s):
    return tuple(s[0]) if len(s) == 1 and isinstance(s[0], (tuple, list)) else s


def zeros(*s, requires_grad=False): return Tensor(np.zeros(_shape(s), np.float32), requires_grad)
def ones(*s, requires_grad=False): return Tensor(np.ones(_shape(s), np.float32), requires_grad)
def randn(*s, requires_grad=False): return Tensor(np.random.randn(*_shape(s)).astype(np.float32), requires_grad)
def rand(*s, requires_grad=False): return Tensor(np.random.rand(*_shape(s)).astype(np.float32), requires_grad)
def arange(*a): return Tensor(np.arange(*a))
def zeros_like(t): return zeros(*t.shape)
def ones_like(t): return ones(*t.shape)
def eye(n): return Tensor(np.eye(n, dtype=np.float32))
def tril(t, diagonal=0): return Tensor(np.tril(_t(t).data, diagonal))
def triu(t, diagonal=0): return Tensor(np.triu(_t(t).data, diagonal))
def from_numpy(a): return Tensor(a)
def is_tensor(x): return isinstance(x, Tensor)


def manual_seed(s):
    np.random.seed(s)
    _random.seed(s)


def matmul(a, b): return _t(a) @ _t(b)
def exp(t): return t.exp()
def log(t): return t.log()
def sqrt(t): return t.sqrt()
def relu(t): return t.relu()
def sigmoid(t): return t.sigmoid()
def tanh(t): return t.tanh()
def softmax(t, dim=-1): return t.softmax(dim)
def argmax(t, dim=None, keepdim=False): return t.argmax(dim, keepdim)
def sum(t, dim=None, keepdim=False): return t.sum(dim, keepdim)
def mean(t, dim=None, keepdim=False): return t.mean(dim, keepdim)
def allclose(a, b, atol=1e-5, rtol=1e-5): return bool(np.allclose(_t(a).data, _t(b).data, atol=atol, rtol=rtol))


def cat(ts, dim=0):
    sizes = [t.data.shape[dim] for t in ts]
    out = np.concatenate([t.data for t in ts], axis=dim)

    def back(g):
        return tuple(np.split(g, np.cumsum(sizes)[:-1], axis=dim))

    return ts[0]._make(out, tuple(ts), back)


def stack(ts, dim=0):
    return cat([t.unsqueeze(dim) for t in ts], dim)


class no_grad:
    def __enter__(self):
        self.prev = _grad_enabled[0]
        _grad_enabled[0] = False

    def __exit__(self, *a):
        _grad_enabled[0] = self.prev

    def __call__(self, fn):
        def wrapped(*a, **k):
            with no_grad():
                return fn(*a, **k)
        return wrapped


# =================== torch.nn ===================
class _NN:
    pass


nn = _NN()


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        d = data.data if isinstance(data, Tensor) else data
        super().__init__(np.asarray(d, dtype=np.float32), requires_grad=requires_grad)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()


class Module:
    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if "_params" not in self.__dict__:
            raise AttributeError("call super().__init__() first, at the top of your __init__")
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def forward(self, *a):
        raise NotImplementedError("define forward(self, x) in your Module")

    def named_parameters(self, prefix=""):
        for n, p in self._params.items():
            yield prefix + n, p
        for n, m in self._modules.items():
            yield from m.named_parameters(prefix + n + ".")

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def children(self):
        return iter(self._modules.values())

    def train(self, mode=True):
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self):
        return {n: p.detach() for n, p in self.named_parameters()}

    def load_state_dict(self, sd):
        for n, p in self.named_parameters():
            p.data = np.array(sd[n].data, dtype=np.float32)

    def to(self, *a, **k):
        return self

    def __repr__(self):
        inner = "".join("\n  (" + n + "): " + repr(m).replace("\n", "\n  ") for n, m in self._modules.items())
        return type(self).__name__ + "(" + inner + ("\n" if inner else "") + ")"


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        k = 1 / math.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-k, k, (out_features, in_features)))
        self.bias = Parameter(np.random.uniform(-k, k, (out_features,))) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y

    def __repr__(self):
        return "Linear(in_features=%d, out_features=%d, bias=%s)" % (self.in_features, self.out_features, self.bias is not None)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim))

    def forward(self, idx):
        return self.weight[_t(idx).data.astype(np.int64)]


class LayerNorm(Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(d))
        self.bias = Parameter(np.zeros(d))

    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = ((x - m) ** 2).mean(-1, keepdim=True)
        return (x - m) / (v + self.eps) ** 0.5 * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32) / (1 - self.p)
        return x * Tensor(mask)


class _Act(Module):
    fn = None

    def forward(self, x):
        return type(self).fn(x)

    def __repr__(self):
        return type(self).__name__ + "()"


class ReLU(_Act): fn = staticmethod(lambda x: x.relu())
class Sigmoid(_Act): fn = staticmethod(lambda x: x.sigmoid())
class Tanh(_Act): fn = staticmethod(lambda x: x.tanh())


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.softmax(self.dim)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, l in enumerate(layers):
            setattr(self, str(i), l)
        object.__setattr__(self, "layers", list(layers))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)


class ModuleList(Module):
    def __init__(self, mods=()):
        super().__init__()
        object.__setattr__(self, "items", [])
        for m in mods:
            self.append(m)

    def append(self, m):
        setattr(self, str(len(self.items)), m)
        self.items.append(m)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


# ----- functional -----
class _F:
    pass


F = _F()
F.relu = lambda x: x.relu()
F.sigmoid = lambda x: x.sigmoid()
F.tanh = lambda x: x.tanh()
F.softmax = lambda x, dim=-1: x.softmax(dim)
F.log_softmax = lambda x, dim=-1: x.log_softmax(dim)


def _cross_entropy(logits, target, reduction="mean"):
    ls = logits.log_softmax(-1)
    t = _t(target).data.astype(np.int64)
    picked = ls[np.arange(len(t)), t]
    loss = -picked
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


def _mse(pred, target, reduction="mean"):
    d = (pred - _t(target)) ** 2
    return d.mean() if reduction == "mean" else d.sum() if reduction == "sum" else d


def _bce(p, y, reduction="mean"):
    y = _t(y)
    p = p.clamp(1e-7, 1 - 1e-7)
    l = -(y * p.log() + (1 - y) * (1 - p).log())
    return l.mean() if reduction == "mean" else l


def _bce_logits(z, y, reduction="mean"):
    return _bce(z.sigmoid(), y, reduction)


F.cross_entropy = _cross_entropy
F.mse_loss = _mse
F.binary_cross_entropy = _bce
F.binary_cross_entropy_with_logits = _bce_logits


class _Loss(Module):
    fn = None

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        return type(self).fn(a, b, self.reduction)


class CrossEntropyLoss(_Loss): fn = staticmethod(_cross_entropy)
class MSELoss(_Loss): fn = staticmethod(_mse)
class BCELoss(_Loss): fn = staticmethod(_bce)
class BCEWithLogitsLoss(_Loss): fn = staticmethod(_bce_logits)


for _n in ["Module", "Linear", "Embedding", "LayerNorm", "Dropout", "ReLU", "Sigmoid", "Tanh", "Softmax",
           "Sequential", "ModuleList", "Parameter", "CrossEntropyLoss", "MSELoss", "BCELoss", "BCEWithLogitsLoss"]:
    setattr(nn, _n, globals()[_n])
nn.functional = F


# =================== torch.optim ===================
class _Optim:
    pass


optim = _Optim()


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum, self.wd = momentum, weight_decay
        self.buf = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data + self.wd * p.data
            if self.momentum:
                self.buf[i] = self.momentum * self.buf[i] + g
                g = self.buf[i]
            p.data = (p.data - self.lr * g).astype(np.float32)


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.wd, self.t = eps, weight_decay, 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data = (p.data - self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


optim.SGD, optim.Adam, optim.Optimizer = SGD, Adam, Optimizer


# =================== torch.utils.data ===================
class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)


def _collate(items):
    first = items[0]
    if isinstance(first, (tuple, list)):
        return tuple(_collate([it[k] for it in items]) for k in range(len(first)))
    if isinstance(first, Tensor):
        return Tensor(np.stack([it.data for it in items]))
    return Tensor(np.array(items))


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last

    def __iter__(self):
        idx = list(range(len(self.dataset)))
        if self.shuffle:
            _random.shuffle(idx)
        for s in range(0, len(idx), self.batch_size):
            chunk = idx[s:s + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                return
            yield _collate([self.dataset[i] for i in chunk])

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


class _Utils:
    pass


utils = _Utils()
utils.data = _Utils()
utils.data.Dataset, utils.data.DataLoader, utils.data.TensorDataset = Dataset, DataLoader, TensorDataset


class _Cuda:
    @staticmethod
    def is_available():
        return False


cuda = _Cuda()


def device(name="cpu"):
    return name
