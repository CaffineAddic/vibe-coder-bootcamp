// Part 5: PyTorch. Runs on torchlite (course/torchlite.py), a NumPy-backed subset of the real API.
const PYTORCH_LESSONS = [
{stage:"16 · PyTorch basics", id:"pt_tensors", title:"Tensors", packages:["numpy","torch"],
learn:`<p>A PyTorch <b>tensor</b> is a NumPy array that can also track gradients and run on a GPU. Most NumPy habits carry over.</p>
<ul><li>Create: <code>torch.tensor([[1., 2.]])</code>, <code>torch.zeros(2, 3)</code>, <code>torch.randn(B, T, D)</code>, <code>torch.arange(5)</code>.</li><li>Shape: <code>x.shape</code> (a <code>torch.Size</code>) and <code>x.size(0)</code> for one dimension.</li><li>Reshape: <code>x.view(B, -1)</code> or <code>x.reshape(...)</code>, <code>x.unsqueeze(0)</code> to add a dimension, <code>x.squeeze()</code> to remove size-1 ones.</li><li>Reorder dimensions: <code>x.transpose(1, 2)</code> swaps two, and <code>x.permute(0, 2, 1, 3)</code> reorders all of them.</li><li>Reductions take <code>dim=</code> (where NumPy says <code>axis=</code>): <code>x.sum(dim=1)</code>, <code>x.mean(dim=-1, keepdim=True)</code>.</li><li><code>@</code> or <code>torch.matmul</code> does batched matrix multiply. <code>.item()</code> turns a one-element tensor into a Python number.</li></ul>
<p>These lessons run on <b>torchlite</b>, a small NumPy-backed copy of the PyTorch API built for this course, because real PyTorch doesn't run in a browser. The code you write is real PyTorch and runs unchanged on the actual library.</p>`,
example:`import torch
x = torch.arange(6).float().view(2, 3)
print(x, x.shape)
print(x.sum(dim=0), x.mean(dim=1, keepdim=True).shape)
print(x.unsqueeze(0).shape, x.T.shape)
B = torch.randn(4, 2, 3)
print((B @ B.transpose(1, 2)).shape)`,
why:"Interviews that allow PyTorch expect fluent shape manipulation. view, transpose and dim= come up in every attention answer.",
task:"Write <code>row_dot(a, b)</code> where a and b are (B, D) tensors. Return a (B,) tensor holding the dot product of each pair of rows. Then write <code>to_heads(x, h)</code> that reshapes (B, T, D) into (B, h, T, D/h), the multi-head layout.",
pseudo:`row_dot: (a * b).sum(dim=1)
to_heads:
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)`,
hints:["Elementwise multiply, then sum over the feature dimension.","Split D into (h, D/h) first, then swap the T and h dimensions."],
starter:`import torch

def row_dot(a, b):
    pass

def to_heads(x, h):
    pass
`,
solution:`import torch

def row_dot(a, b):
    return (a * b).sum(dim=1)

def to_heads(x, h):
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)
`,
tests:`import torch
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
r = row_dot(a, b)
check(tuple(r.shape) == (2,), "row_dot returns shape (B,)")
check(r.tolist() == [17.0, 53.0], "1*5+2*6 = 17 and 3*7+4*8 = 53")
x = torch.arange(2 * 3 * 8).float().view(2, 3, 8)
y = to_heads(x, 4)
check(tuple(y.shape) == (2, 4, 3, 2), "(2, 3, 8) with 4 heads becomes (2, 4, 3, 2)")
check(y[0, 1, 2].tolist() == x[0, 2, 2:4].tolist(), "head 1 of token 2 holds features 2..3 of that token")`},

{stage:"16 · PyTorch basics", id:"pt_autograd", title:"Autograd", packages:["numpy","torch"],
learn:`<p>PyTorch records the operations you do on tensors that have <code>requires_grad=True</code>. Calling <code>.backward()</code> on a scalar result fills in <code>.grad</code> for each of them.</p>
<pre>w = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (w ** 2).sum()
loss.backward()
w.grad   # tensor([2., 4.])</pre>
<p>Three rules interviewers check:</p>
<ul><li>Gradients <b>accumulate</b>. Calling backward twice adds to <code>.grad</code>. That's why training loops call <code>optimizer.zero_grad()</code> every step.</li><li><code>backward()</code> without an argument only works on a scalar. Reduce the loss with <code>.mean()</code> or <code>.sum()</code> first.</li><li>Wrap evaluation in <code>with torch.no_grad():</code> so no graph is built. Update weights manually inside it too.</li></ul>`,
example:`import torch
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
print("dy/dx at 3:", x.grad)   # 2x + 2 = 8
y = x ** 2 + 2 * x
y.backward()
print("after a second backward (accumulated):", x.grad)`,
why:"'What does zero_grad do and why?' and 'why no_grad at eval?' are standard questions.",
task:"Write <code>fit(x, y, steps=200, lr=0.1)</code> that learns a scalar <code>w</code> and <code>b</code> (both starting at 0, with <code>requires_grad=True</code>) to minimize the mean of (w·x + b − y)². Use autograd for the gradients and update by hand inside <code>torch.no_grad()</code>. Zero the grads every step. Return <code>(w.item(), b.item())</code>.",
pseudo:`w = torch.tensor(0.0, requires_grad=True); b = same
repeat steps:
    loss = ((w * x + b - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad; b -= lr * b.grad
    w.grad = None; b.grad = None`,
hints:["Without no_grad, the update itself would be recorded in the graph.","Setting .grad = None (or .grad.zero_()) stops the gradients from piling up."],
starter:`import torch

def fit(x, y, steps=200, lr=0.1):
    pass
`,
solution:`import torch

def fit(x, y, steps=200, lr=0.1):
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    for _ in range(steps):
        loss = ((w * x + b - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None
        b.grad = None
    return w.item(), b.item()
`,
tests:`import torch
x = torch.tensor([0.0, 1.0, 2.0, 3.0])
w, b = fit(x, 3 * x - 1)
check(abs(w - 3) < 0.02, "w should be close to 3 (got " + str(round(w, 3)) + ")")
check(abs(b + 1) < 0.05, "b should be close to -1 (got " + str(round(b, 3)) + ")")`},

{stage:"16 · PyTorch basics", id:"pt_module", title:"nn.Module", packages:["numpy","torch"],
learn:`<p>Every PyTorch model is a subclass of <code>nn.Module</code>, written the same way every time:</p>
<pre>import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                 # always first
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))</pre>
<p>Layers assigned to <code>self</code> in <code>__init__</code> are registered automatically, so <code>model.parameters()</code> finds their weights. You call the model as <code>model(x)</code>, never <code>model.forward(x)</code>.</p>
<p><code>nn.Linear(in, out)</code> stores a weight of shape <b>(out, in)</b> and computes <code>x @ W.T + b</code>. Its parameter count is <code>in·out + out</code>.</p>
<p><code>nn.Sequential(nn.Linear(...), nn.ReLU(), ...)</code> chains layers when you don't need custom logic.</p>`,
example:`import torch
import torch.nn as nn
layer = nn.Linear(3, 2)
print(layer, layer.weight.shape, layer.bias.shape)
model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
print(model)
print(sum(p.numel() for p in model.parameters()), "parameters")
print(model(torch.randn(5, 3)).shape)`,
why:"This is the exact pattern behind 'implement X as a class'. Write the skeleton first and you'll never face a blank page.",
task:"Write <code>class MLP(nn.Module)</code> with <code>__init__(self, d_in, d_hidden, d_out)</code> and <code>forward(self, x)</code>: Linear → ReLU → Linear. Then write <code>count_params(model)</code> that returns the total number of parameters.",
pseudo:`class MLP(nn.Module):
    __init__: super().__init__(); self.fc1 = Linear(d_in, d_hidden); self.fc2 = Linear(d_hidden, d_out)
    forward:  return self.fc2(relu(self.fc1(x)))

count_params: sum(p.numel() for p in model.parameters())`,
hints:["super().__init__() must come before you assign any layer.","For 4 → 8 → 3: (4·8 + 8) + (8·3 + 3) = 67."],
starter:`import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        pass

    def forward(self, x):
        pass

def count_params(model):
    pass
`,
solution:`import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def count_params(model):
    return sum(p.numel() for p in model.parameters())
`,
tests:`import torch
m = MLP(4, 8, 3)
check(tuple(m(torch.randn(5, 4)).shape) == (5, 3), "output shape is (batch, d_out)")
check(count_params(m) == 67, "4->8->3 has 67 parameters, got " + str(count_params(m)))
check(len(list(m.parameters())) == 4, "two weights and two biases are registered")`},

{stage:"16 · PyTorch basics", id:"pt_train", title:"The training loop", packages:["numpy","torch"],
learn:`<p>Every PyTorch training loop has the same five lines in the same order. Memorize them:</p>
<pre>for epoch in range(epochs):
    logits = model(X)                 # 1. forward
    loss = loss_fn(logits, y)         # 2. loss
    optimizer.zero_grad()             # 3. clear old gradients
    loss.backward()                   # 4. backprop
    optimizer.step()                  # 5. update weights</pre>
<p><code>nn.CrossEntropyLoss</code> takes raw <b>logits</b> of shape (N, C) and integer class labels of shape (N,). It applies log-softmax itself, so don't put a softmax in your model's forward. That's a classic bug.</p>
<p>The optimizer is created from the parameters: <code>torch.optim.Adam(model.parameters(), lr=1e-2)</code>.</p>`,
example:`import torch
import torch.nn as nn
logits = torch.tensor([[2.0, 0.5], [0.1, 1.5]])
labels = torch.tensor([0, 1])
print(nn.CrossEntropyLoss()(logits, labels))`,
why:"'Write a training loop' is asked in nearly every PyTorch interview, and the order of those five lines is what they check.",
task:"Write <code>train(model, X, y, epochs=100, lr=0.05)</code> for classification. Use <code>nn.CrossEntropyLoss</code> and <code>torch.optim.Adam</code>, train full-batch, and return a list of the loss value (a float) from every epoch.",
pseudo:`loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)
history = []
for each epoch:
    loss = loss_fn(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
    history.append(loss.item())
return history`,
hints:["Use .item() so you store floats, not tensors that keep the graph alive.","zero_grad goes before backward, every single step."],
starter:`import torch
import torch.nn as nn

def train(model, X, y, epochs=100, lr=0.05):
    pass
`,
solution:`import torch
import torch.nn as nn

def train(model, X, y, epochs=100, lr=0.05):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        loss = loss_fn(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history
`,
tests:`import torch
import torch.nn as nn
torch.manual_seed(0)
X = torch.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).long()
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
h = train(model, X, y)
check(isinstance(h, list) and len(h) == 100, "returns one loss per epoch")
check(isinstance(h[0], float), "losses are floats (use .item())")
check(h[-1] < h[0] * 0.5, "loss should drop by more than half")
with torch.no_grad():
    acc = (model(X).argmax(dim=1) == y).float().mean().item()
check(acc > 0.95, "training accuracy above 95% (got " + str(round(acc, 3)) + ")")`},

{stage:"16 · PyTorch basics", id:"pt_data", title:"Dataset and DataLoader", packages:["numpy","torch"],
learn:`<p>A <code>Dataset</code> answers two questions: how many examples (<code>__len__</code>) and what example <i>i</i> is (<code>__getitem__</code>). A <code>DataLoader</code> wraps it to produce shuffled mini-batches.</p>
<pre>from torch.utils.data import Dataset, DataLoader

class MyData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

loader = DataLoader(MyData(X, y), batch_size=32, shuffle=True)
for xb, yb in loader:
    ...</pre>
<p>Shuffle the training loader but not the validation one. Do preprocessing such as tokenizing or normalizing in <code>__getitem__</code> or before building the dataset.</p>`,
example:`import torch
from torch.utils.data import TensorDataset, DataLoader
ds = TensorDataset(torch.arange(10).float(), torch.arange(10))
for xb, yb in DataLoader(ds, batch_size=4):
    print(xb.shape, yb.tolist())`,
why:"Writing a custom Dataset is a common warm-up, especially for text, where __getitem__ does the tokenizing.",
task:"Write <code>class TextDataset(Dataset)</code> built from a list of sentences, a list of int labels, a <code>vocab</code> dict and a <code>max_len</code>. <code>__getitem__</code> lowercases and splits the sentence, maps each word to its id (unknown words use <code>vocab[\"&lt;unk&gt;\"]</code>), truncates or pads with 0 to exactly max_len, and returns <code>(torch.tensor(ids), torch.tensor(label))</code>.",
pseudo:`ids = [vocab.get(w, vocab["<unk>"]) for w in text.lower().split()]
ids = ids[:max_len] + [0] * (max_len - len(ids))
return torch.tensor(ids), torch.tensor(label)`,
hints:["[0] * negative number is an empty list, so the padding line also works when the sentence is too long.","Store everything in __init__ and do the work in __getitem__."],
starter:`import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        pass

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass
`,
solution:`import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts, self.labels = texts, labels
        self.vocab, self.max_len = vocab, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        unk = self.vocab["<unk>"]
        ids = [self.vocab.get(w, unk) for w in self.texts[i].lower().split()]
        ids = ids[:self.max_len] + [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[i])
`,
tests:`import torch
from torch.utils.data import DataLoader
vocab = {"<pad>": 0, "<unk>": 1, "good": 2, "movie": 3, "bad": 4}
ds = TextDataset(["Good movie", "bad bad bad bad bad", "great movie"], [1, 0, 1], vocab, 4)
check(len(ds) == 3, "__len__ is the number of sentences")
x, y = ds[0]
check(x.tolist() == [2, 3, 0, 0] and y.item() == 1, "'Good movie' -> [2, 3, 0, 0], padded")
check(ds[1][0].tolist() == [4, 4, 4, 4], "long sentences are truncated to max_len")
check(ds[2][0].tolist() == [1, 3, 0, 0], "unknown word 'great' maps to <unk>")
xb, yb = next(iter(DataLoader(ds, batch_size=2)))
check(tuple(xb.shape) == (2, 4) and yb.tolist() == [1, 0], "DataLoader stacks items into (batch, max_len)")`},

{stage:"16 · PyTorch basics", id:"pt_eval", title:"Evaluation mode", packages:["numpy","torch"],
learn:`<p>Two switches matter when you evaluate a model or serve predictions:</p>
<ul><li><code>model.eval()</code> changes layer <b>behavior</b>: Dropout stops dropping and BatchNorm uses its running statistics. Call <code>model.train()</code> to switch back before training again.</li><li><code>torch.no_grad()</code> stops PyTorch from <b>recording</b> operations for backprop, which saves memory and time.</li></ul>
<p>They do different things, and you need both. A common interview question is "what's the difference between model.eval() and torch.no_grad()?"</p>`,
example:`import torch
import torch.nn as nn
drop = nn.Dropout(0.5)
x = torch.ones(1, 8)
print("train mode:", drop(x))
drop.eval()
print("eval mode: ", drop(x))`,
why:"Forgetting eval() gives noisy, wrong validation numbers, and interviewers ask about it directly.",
task:"Write <code>predict(model, X)</code>. Put the model in eval mode, run it under <code>torch.no_grad()</code>, and return the predicted class ids as a Python list. Put the model back in train mode before returning, since the caller may keep training.",
pseudo:`model.eval()
with torch.no_grad():
    preds = model(X).argmax(dim=1)
model.train()
return preds.tolist()`,
hints:["argmax(dim=1) picks the highest logit per row.","Restoring train mode is a nice touch interviewers notice."],
starter:`import torch

def predict(model, X):
    pass
`,
solution:`import torch

def predict(model, X):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    model.train()
    return preds.tolist()
`,
tests:`import torch
import torch.nn as nn
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Dropout(0.9), nn.Linear(32, 3))
X = torch.randn(10, 4)
a, b = predict(m, X), predict(m, X)
check(isinstance(a, list) and len(a) == 10, "returns a list of 10 class ids")
check(a == b, "predictions are deterministic, so dropout was off")
check(m.training, "the model is back in train mode afterwards")`},

{stage:"17 · Attention in PyTorch", id:"pt_selfattn", title:"Self-attention module", packages:["numpy","torch"],
learn:`<p>This is the interview question you froze on, now as a real <code>nn.Module</code>. Write the skeleton first, then fill in <code>forward</code> while saying the shapes out loud:</p>
<pre>x:        (B, T, D)
Q, K, V:  (B, T, D)   from three nn.Linear(D, D)
scores:   (B, T, T)   Q @ K.transpose(-2, -1) / sqrt(D)
mask:     (T, T)      torch.tril(torch.ones(T, T))
weights:  (B, T, T)   softmax over the last dim
out:      (B, T, D)   weights @ V, then an output nn.Linear(D, D)</pre>
<p><code>scores.masked_fill(mask == 0, float("-inf"))</code> blocks the future positions.</p>`,
example:`import torch
T = 4
print(torch.tril(torch.ones(T, T)))
s = torch.zeros(T, T).masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
print(torch.softmax(s, dim=-1))`,
why:"Single-head causal self-attention is the most common transformer coding question.",
task:"Write <code>class SelfAttention(nn.Module)</code> with <code>__init__(self, d_model, causal=False)</code> and <code>forward(self, x)</code>. Use linear layers named <code>q</code>, <code>k</code>, <code>v</code> and <code>out</code>, all d_model → d_model. Return a tensor of shape (B, T, D).",
pseudo:`__init__: super().__init__(); self.q/k/v/out = nn.Linear(d, d); self.causal = causal
forward(x):
    B, T, D = x.shape
    Q, K, V = self.q(x), self.k(x), self.v(x)
    s = Q @ K.transpose(-2, -1) / math.sqrt(D)
    if self.causal: s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    return self.out(torch.softmax(s, dim=-1) @ V)`,
hints:["import math for math.sqrt.","Store causal on self in __init__ so forward can read it."],
starter:`import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, causal=False):
        pass

    def forward(self, x):
        pass
`,
solution:`import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, causal=False):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.causal = causal

    def forward(self, x):
        B, T, D = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        s = Q @ K.transpose(-2, -1) / math.sqrt(D)
        if self.causal:
            s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        return self.out(torch.softmax(s, dim=-1) @ V)
`,
tests:`import torch
torch.manual_seed(0)
att = SelfAttention(8, causal=True)
x = torch.randn(2, 5, 8)
y = att(x)
check(tuple(y.shape) == (2, 5, 8), "output shape is (B, T, D)")
check(sum(p.numel() for p in att.parameters()) == 4 * (8 * 8 + 8), "four D x D linear layers with biases")
x2 = x.detach()
x2 = torch.cat([x2[:, :3], x2[:, 3:] + 5.0], dim=1)
check(torch.allclose(att(x2)[:, :3], y[:, :3], atol=1e-5), "causal: changing tokens 3-4 doesn't change outputs 0-2")
y.sum().backward()
check(att.q.weight.grad is not None, "gradients flow back to the q projection")`},

{stage:"17 · Attention in PyTorch", id:"pt_mha", title:"Multi-head attention", packages:["numpy","torch"],
learn:`<p>Multi-head attention runs h smaller attentions in parallel. The usual follow-up after single-head attention is "now make it multi-head". The only new part is reshaping:</p>
<pre>Q: (B, T, D) → view(B, T, h, D/h) → transpose(1, 2) → (B, h, T, d_head)
scores: (B, h, T, T)   divide by sqrt(d_head), not sqrt(D)
out:    (B, h, T, d_head) → transpose(1, 2) → reshape(B, T, D) → Linear(D, D)</pre>
<p>Check <code>D % h == 0</code> in <code>__init__</code>. The parameter count doesn't depend on h: it's 4·(D² + D) either way.</p>`,
example:`import torch
B, T, D, h = 2, 3, 8, 4
x = torch.randn(B, T, D)
heads = x.view(B, T, h, D // h).transpose(1, 2)
print(heads.shape)
back = heads.transpose(1, 2).reshape(B, T, D)
print(torch.allclose(back, x))`,
why:"This is the standard 'level up' question once single-head is done.",
task:"Write <code>class MultiHeadAttention(nn.Module)</code> with <code>__init__(self, d_model, n_heads, causal=False)</code>. Raise <code>ValueError</code> if d_model isn't divisible by n_heads. Use layers named <code>q</code>, <code>k</code>, <code>v</code>, <code>out</code>, all D → D. forward takes (B, T, D) and returns (B, T, D).",
pseudo:`split(t): t.view(B, T, h, dh).transpose(1, 2)
Q, K, V = split(self.q(x)), split(self.k(x)), split(self.v(x))
s = Q @ K.transpose(-2, -1) / sqrt(dh)        # (B, h, T, T)
causal mask as before (broadcasts over B and h)
o = softmax(s, -1) @ V                         # (B, h, T, dh)
o = o.transpose(1, 2).reshape(B, T, D)
return self.out(o)`,
hints:["dh = D // h. Store h and dh on self.","After transpose, use reshape (not view) to merge the heads back."],
starter:`import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        pass

    def forward(self, x):
        pass
`,
solution:`import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh, self.causal = n_heads, d_model // n_heads, causal
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        Q, K, V = split(self.q(x)), split(self.k(x)), split(self.v(x))
        s = Q @ K.transpose(-2, -1) / math.sqrt(self.dh)
        if self.causal:
            s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        o = torch.softmax(s, dim=-1) @ V
        return self.out(o.transpose(1, 2).reshape(B, T, D))
`,
tests:`import torch
torch.manual_seed(0)
m = MultiHeadAttention(8, 2, causal=True)
x = torch.randn(3, 4, 8)
y = m(x)
check(tuple(y.shape) == (3, 4, 8), "output shape is (B, T, D)")
check(sum(p.numel() for p in m.parameters()) == 4 * (64 + 8), "parameter count is 4 * (D*D + D)")
x2 = torch.cat([x[:, :2], x[:, 2:] * 3.0], dim=1)
check(torch.allclose(m(x2)[:, :2], y[:, :2], atol=1e-5), "causal mask blocks future tokens")
raised = False
try:
    MultiHeadAttention(10, 3)
except ValueError:
    raised = True
check(raised, "d_model=10 with 3 heads raises ValueError")`},

{stage:"17 · Attention in PyTorch", id:"pt_block", title:"A transformer block", packages:["numpy","torch"],
learn:`<p>A transformer block wraps attention with an MLP, residual connections and LayerNorm. Modern models use the <b>pre-norm</b> layout:</p>
<pre>x = x + attn(ln1(x))
x = x + mlp(ln2(x))      # mlp: Linear(D, 4D) → ReLU (or GELU) → Linear(4D, D)</pre>
<p>Residuals give gradients a direct path through deep stacks. Pre-norm (LayerNorm before each sub-layer) trains more stably than the original post-norm layout. The MLP is usually 4× wider than D.</p>
<p>If you can write this block from memory, you can write a GPT-style model: stack N blocks, add token and position embeddings at the start and a final LayerNorm and Linear to vocabulary logits at the end.</p>`,
example:`import torch
import torch.nn as nn
ln = nn.LayerNorm(4)
x = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
print(ln(x))`,
why:"'Implement a transformer block' is the natural end point of attention questions.",
task:"Write <code>class Block(nn.Module)</code> with <code>__init__(self, d_model, n_heads)</code>: a causal <code>MultiHeadAttention</code> (provided below) named <code>attn</code>, <code>nn.LayerNorm</code>s named <code>ln1</code> and <code>ln2</code>, and an <code>nn.Sequential</code> MLP named <code>mlp</code> (D → 4D → ReLU → D). forward applies the pre-norm layout.",
pseudo:`__init__:
    self.ln1 = nn.LayerNorm(d); self.attn = MultiHeadAttention(d, h, causal=True)
    self.ln2 = nn.LayerNorm(d); self.mlp = Sequential(Linear(d, 4d), ReLU(), Linear(4d, d))
forward:
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x`,
hints:["Residual means adding the sub-layer's output back to its input.","Don't reassign x before computing the residual. Write x = x + f(norm(x))."],
starter:`import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        self.h, self.dh, self.causal = n_heads, d_model // n_heads, causal
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        Q, K, V = split(self.q(x)), split(self.k(x)), split(self.v(x))
        s = Q @ K.transpose(-2, -1) / math.sqrt(self.dh)
        if self.causal:
            s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        o = torch.softmax(s, dim=-1) @ V
        return self.out(o.transpose(1, 2).reshape(B, T, D))

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        pass

    def forward(self, x):
        pass
`,
solution:`import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        self.h, self.dh, self.causal = n_heads, d_model // n_heads, causal
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        Q, K, V = split(self.q(x)), split(self.k(x)), split(self.v(x))
        s = Q @ K.transpose(-2, -1) / math.sqrt(self.dh)
        if self.causal:
            s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        o = torch.softmax(s, dim=-1) @ V
        return self.out(o.transpose(1, 2).reshape(B, T, D))

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
`,
tests:`import torch
torch.manual_seed(0)
b = Block(8, 2)
x = torch.randn(2, 5, 8)
y = b(x)
check(tuple(y.shape) == (2, 5, 8), "output shape is (B, T, D)")
n = sum(p.numel() for p in b.parameters())
check(n == 4 * (64 + 8) + 2 * 16 + (8 * 32 + 32) + (32 * 8 + 8), "attention + 2 LayerNorms + MLP parameters, got " + str(n))
x2 = torch.cat([x[:, :2], x[:, 2:] - 4.0], dim=1)
check(torch.allclose(b(x2)[:, :2], y[:, :2], atol=1e-4), "still causal after the MLP and residuals")
for p in b.parameters():
    p.grad = None
b(x).sum().backward()
check(b.ln1.weight.grad is not None and b.mlp[0].weight.grad is not None, "gradients reach every sub-layer")`},
];
