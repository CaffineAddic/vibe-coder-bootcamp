// Part 3: NumPy. Vectorized versions of what you wrote with loops in Part 2.
const NUMPY_LESSONS = [
{stage:"13 · NumPy", id:"np_arrays", title:"Arrays, shapes and axes", packages:["numpy"],
learn:`<p>A NumPy <b>array</b> is a grid of numbers that all share one type. Its <code>.shape</code> is a tuple: a batch of 32 examples with 10 features is <code>(32, 10)</code>.</p>
<p>Make arrays with <code>np.array(list)</code>, <code>np.zeros((r, c))</code>, <code>np.ones</code>, <code>np.arange(n)</code>, <code>np.random.randn(r, c)</code>. Change shape with <code>.reshape(r, c)</code>. One dimension can be <code>-1</code>, meaning "work it out".</p>
<p>The idea that matters most is <b>axis</b>. <code>X.sum(axis=0)</code> collapses the rows and gives one number per column. <code>X.sum(axis=1)</code> collapses the columns and gives one number per row. Read <code>axis=0</code> as "down the rows". <code>keepdims=True</code> keeps the collapsed dimension as size 1, which you'll need for broadcasting.</p>
<p>Math on arrays happens elementwise, with no loops: <code>X * 2</code>, <code>X ** 2</code>, <code>np.exp(X)</code>. <code>@</code> is matrix multiply.</p>`,
example:`import numpy as np
X = np.arange(6).reshape(2, 3)
print(X)
print("shape", X.shape, "dtype", X.dtype)
print("sum axis=0 (per column):", X.sum(axis=0))
print("sum axis=1 (per row):   ", X.sum(axis=1))
print("keepdims:", X.sum(axis=1, keepdims=True).shape)
print(X @ X.T)`,
why:"Interviewers check whether you can reason about shapes and axes without guessing. Most ML bugs are shape bugs.",
task:"Write <code>col_stats(X)</code> that returns a tuple <code>(means, maxes)</code>: the mean and the max of each <b>column</b> of a 2-D array. No loops.",
pseudo:`means = X.mean(axis=0)
maxes = X.max(axis=0)
return means, maxes`,
hints:["Per column means collapse the rows, which is axis=0.","Both results should have shape (n_columns,)."],
starter:`import numpy as np

def col_stats(X):
    pass
`,
solution:`import numpy as np

def col_stats(X):
    return X.mean(axis=0), X.max(axis=0)
`,
tests:`import numpy as np
X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 0.0]])
m, mx = col_stats(X)
check(m.shape == (2,), "means should have shape (2,), got " + str(np.shape(m)))
check(np.allclose(m, [3.0, 10.0]), "column means are [3, 10]")
check(np.allclose(mx, [5.0, 20.0]), "column maxes are [5, 20]")`},

{stage:"13 · NumPy", id:"np_masks", title:"Indexing and boolean masks", packages:["numpy"],
learn:`<p>Index a 2-D array with <code>X[row, col]</code>. Slices work per dimension: <code>X[:, 0]</code> is the first column, <code>X[:5]</code> the first five rows, <code>X[:, -2:]</code> the last two columns.</p>
<p>A <b>boolean mask</b> picks the elements where a condition is true: <code>x[x &gt; 0]</code>. For rows, build a mask with one value per row: <code>X[X[:, 0] &gt; 0]</code> keeps rows whose first column is positive. Combine masks with <code>&amp;</code>, <code>|</code> and <code>~</code> (not <code>and</code>/<code>or</code>), and wrap each condition in parentheses.</p>
<p><code>np.where(cond, a, b)</code> picks from a where cond is true and from b elsewhere. It's a vectorized if/else.</p>
<p><b>Fancy indexing</b> uses an array of indexes: <code>probs[np.arange(n), labels]</code> picks one entry per row. This is how you grab the probability of the correct class.</p>`,
example:`import numpy as np
X = np.array([[1, -2], [-3, 4], [5, 6]])
print("first column:", X[:, 0])
print("rows with positive col 0:\\n", X[X[:, 0] > 0])
print("negatives to 0:\\n", np.where(X < 0, 0, X))
probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
labels = np.array([1, 0, 1])
print("prob of correct class:", probs[np.arange(3), labels])`,
why:"Filtering rows, masking padding tokens and picking out the correct-class probability all use these tricks.",
task:"Write <code>nll(probs, labels)</code>: the mean negative log-likelihood. <code>probs</code> is (N, C), <code>labels</code> is (N,) ints. Pick each row's probability of its label with fancy indexing, clip to at least 1e-12, and return the mean of −log. No loops.",
pseudo:`p = probs[np.arange(N), labels]
p = np.clip(p, 1e-12, 1.0)
return -np.log(p).mean()`,
hints:["np.arange(len(labels)) gives the row indexes.","np.clip(a, lo, hi) bounds the values."],
starter:`import numpy as np

def nll(probs, labels):
    pass
`,
solution:`import numpy as np

def nll(probs, labels):
    p = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())
`,
tests:`import numpy as np
P = np.array([[0.1, 0.9], [0.8, 0.2]])
check(close(nll(P, np.array([1, 0])), -(np.log(0.9) + np.log(0.8)) / 2), "mean of -log(0.9) and -log(0.8)")
check(nll(np.array([[1.0, 0.0]]), np.array([1])) > 20, "p = 0 is clipped, not infinite")`},

{stage:"13 · NumPy", id:"np_broadcast", title:"Broadcasting", packages:["numpy"],
learn:`<p><b>Broadcasting</b> lets arrays of different shapes combine without copying. NumPy lines the shapes up <b>from the right</b>. Two dimensions are compatible if they're equal or one of them is 1. A size-1 dimension is stretched to match.</p>
<ul><li><code>(32, 10) - (10,)</code> works: subtract a per-column value from every row.</li><li><code>(32, 10) - (32,)</code> fails, because 10 and 32 don't match.</li><li><code>(32, 10) - (32, 1)</code> works: subtract a per-row value. That's why you use <code>keepdims=True</code>.</li></ul>
<p>When unsure, print the shapes and line them up from the right on paper.</p>`,
example:`import numpy as np
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
col_mean = X.mean(axis=0)                 # (3,)
row_mean = X.mean(axis=1, keepdims=True)  # (2, 1)
print(X - col_mean)
print(X - row_mean)
try:
    X - X.mean(axis=1)                    # (2,3) - (2,) fails
except ValueError as e:
    print("error:", e)`,
why:"Feature scaling, LayerNorm and adding biases all rely on broadcasting. Getting it wrong silently produces nonsense.",
task:"Write <code>layer_norm(X, eps=1e-5)</code> that normalizes each <b>row</b> of X to mean 0 and variance 1: (x − row mean) / sqrt(row variance + eps). No loops.",
pseudo:`mu  = X.mean(axis=1, keepdims=True)   # (N, 1)
var = X.var(axis=1, keepdims=True)    # (N, 1)
return (X - mu) / np.sqrt(var + eps)`,
hints:["Per row means axis=1.","keepdims=True gives (N, 1), which broadcasts against (N, D)."],
starter:`import numpy as np

def layer_norm(X, eps=1e-5):
    pass
`,
solution:`import numpy as np

def layer_norm(X, eps=1e-5):
    mu = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)
`,
tests:`import numpy as np
X = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 40.0]])
Y = layer_norm(X)
check(Y.shape == X.shape, "output keeps the input shape")
check(np.allclose(Y.mean(axis=1), 0, atol=1e-6), "each row has mean 0")
check(np.allclose(Y.var(axis=1), 1, atol=1e-3), "each row has variance 1")`},

{stage:"13 · NumPy", id:"np_softmax", title:"Vectorized softmax", packages:["numpy"],
learn:`<p>In Part 2 you wrote softmax for one list. In NumPy you do a whole batch at once by choosing the axis and keeping dimensions so broadcasting lines up.</p>
<p>The stability trick is the same: subtract the row max before <code>np.exp</code>.</p>`,
example:`import numpy as np
z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
print(np.exp(z[0]) / np.exp(z[0]).sum())
print("naive on big values:", np.exp(z[1]))  # overflow to inf`,
why:"This is one of the most common ML coding questions, and you're expected to write it vectorized.",
task:"Write <code>softmax(Z, axis=-1)</code> that works on any shape, along any axis, and is numerically stable.",
pseudo:`Z = Z - Z.max(axis=axis, keepdims=True)
E = np.exp(Z)
return E / E.sum(axis=axis, keepdims=True)`,
hints:["Both the max and the sum need keepdims=True.","Test it mentally: every slice along axis should sum to 1."],
starter:`import numpy as np

def softmax(Z, axis=-1):
    pass
`,
solution:`import numpy as np

def softmax(Z, axis=-1):
    Z = Z - Z.max(axis=axis, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=axis, keepdims=True)
`,
tests:`import numpy as np
P = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]))
check(np.allclose(P.sum(axis=-1), 1), "each row sums to 1")
check(np.allclose(P[0], P[1]), "shifting a row by a constant doesn't change it, and big values don't overflow")
Q = softmax(np.ones((2, 3, 4)), axis=1)
check(np.allclose(Q, 1 / 3), "works along axis=1 of a 3-D array")`},

{stage:"13 · NumPy", id:"np_onehot", title:"One-hot encoding and accuracy", packages:["numpy"],
learn:`<p><b>One-hot</b> encoding turns a class id into a vector with a single 1: with 3 classes, label 2 becomes <code>[0, 0, 1]</code>. The fast way is to index into an identity matrix: <code>np.eye(C)[labels]</code>.</p>
<p><code>np.argmax(logits, axis=1)</code> turns a batch of scores into predicted classes. Comparing with <code>==</code> gives booleans, and the mean of booleans is the fraction that are True.</p>`,
example:`import numpy as np
labels = np.array([2, 0, 1])
print(np.eye(3)[labels])
logits = np.array([[0.1, 0.2, 3.0], [2.0, 0.1, 0.0], [0.0, 0.1, 0.0]])
preds = logits.argmax(axis=1)
print(preds, (preds == labels).mean())`,
why:"One-hot targets, argmax predictions and accuracy show up in nearly every classification question.",
task:"Write <code>one_hot(labels, num_classes)</code> returning an (N, C) float array, and <code>accuracy(logits, labels)</code> returning a float between 0 and 1.",
pseudo:`one_hot: np.eye(num_classes)[labels]
accuracy: (logits.argmax(axis=1) == labels).mean()`,
hints:["np.eye(C) is the C×C identity matrix. Row i is the one-hot vector for class i.","Wrap the accuracy in float() so you return a plain number."],
starter:`import numpy as np

def one_hot(labels, num_classes):
    pass

def accuracy(logits, labels):
    pass
`,
solution:`import numpy as np

def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

def accuracy(logits, labels):
    return float((logits.argmax(axis=1) == labels).mean())
`,
tests:`import numpy as np
O = one_hot(np.array([2, 0]), 3)
check(O.shape == (2, 3), "shape is (N, C)")
check(np.array_equal(O, [[0, 0, 1], [1, 0, 0]]), "label 2 -> [0,0,1], label 0 -> [1,0,0]")
L = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
check(close(accuracy(L, np.array([1, 0, 0, 0])), 0.75), "3 of 4 correct is 0.75")`},

{stage:"13 · NumPy", id:"np_pairwise", title:"Pairwise distances without loops", packages:["numpy"],
learn:`<p>A classic "vectorize this" question: compute the distance between every row of A (n × d) and every row of B (m × d), giving an (n × m) matrix.</p>
<p>Two ways to do it:</p>
<ul><li><b>Broadcasting:</b> <code>A[:, None, :] - B[None, :, :]</code> has shape (n, m, d). Square, sum over the last axis, take the square root. Easy to write, but it uses n·m·d memory.</li><li><b>The algebra trick:</b> ‖a − b‖² = ‖a‖² + ‖b‖² − 2·a·b. That's <code>(A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T</code>, which only needs n·m memory. Clip at 0 before the square root, since rounding can make tiny values negative.</li></ul>
<p><code>None</code> in an index (also written <code>np.newaxis</code>) inserts a size-1 dimension.</p>`,
example:`import numpy as np
A = np.array([[0.0, 0.0], [1.0, 1.0]])
B = np.array([[1.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
print((A[:, None, :] - B[None, :, :]).shape)
print(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)))`,
why:"kNN, k-means, embedding search and contrastive losses all need it, and it's a favorite 'remove the loops' question.",
task:"Write <code>pairwise_dist(A, B)</code> using the algebra trick (no (n, m, d) intermediate). Return the (n, m) Euclidean distances.",
pseudo:`a2 = (A**2).sum(axis=1)[:, None]   # (n, 1)
b2 = (B**2).sum(axis=1)[None, :]   # (1, m)
d2 = a2 + b2 - 2 * A @ B.T
return np.sqrt(np.maximum(d2, 0))`,
hints:["Shapes: (n,1) + (1,m) broadcasts to (n,m), and A @ B.T is (n,m).","np.maximum(d2, 0) fixes tiny negative values from rounding."],
starter:`import numpy as np

def pairwise_dist(A, B):
    pass
`,
solution:`import numpy as np

def pairwise_dist(A, B):
    a2 = (A ** 2).sum(axis=1)[:, None]
    b2 = (B ** 2).sum(axis=1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2 * A @ B.T, 0))
`,
tests:`import numpy as np
A = np.array([[0.0, 0.0], [1.0, 1.0]])
B = np.array([[1.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
D = pairwise_dist(A, B)
check(D.shape == (2, 3), "shape is (n, m)")
check(np.allclose(D[0], [1, 0, 5]), "distances from the origin are 1, 0, 5")
rng = np.random.default_rng(0)
X, Y = rng.normal(size=(20, 5)), rng.normal(size=(7, 5))
check(np.allclose(pairwise_dist(X, Y), np.sqrt(((X[:, None] - Y[None]) ** 2).sum(-1))), "matches the broadcasting version")
check(np.all(np.diag(pairwise_dist(X, X)) >= 0), "no NaNs from tiny negative values")`},

{stage:"13 · NumPy", id:"np_linreg", title:"Vectorized gradient descent", packages:["numpy"],
learn:`<p>With matrices, the linear-regression gradient you wrote with loops collapses to one line. For predictions <code>ŷ = Xw + b</code> and MSE loss:</p>
<ul><li><code>err = X @ w + b - y</code> has shape (N,)</li><li><code>dw = 2 / N * X.T @ err</code> has shape (D,)</li><li><code>db = 2 / N * err.sum()</code></li></ul>
<p>Shape-check everything: X is (N, D), w is (D,), y is (N,). If <code>X.T @ err</code> doesn't come out as (D,), something is transposed wrong.</p>`,
example:`import numpy as np
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
w = np.zeros(2)
print((X @ w).shape, X.T.shape)`,
why:"'Implement linear regression with NumPy' is a staple, and the follow-up is always the gradient's shape.",
task:"Write <code>fit_linreg(X, y, lr=0.1, steps=500)</code> using vectorized gradient descent from w = zeros, b = 0. Return <code>(w, b)</code>.",
pseudo:`N, D = X.shape
w = np.zeros(D); b = 0.0
repeat steps:
    err = X @ w + b - y
    w -= lr * 2/N * X.T @ err
    b -= lr * 2/N * err.sum()
return w, b`,
hints:["X.T @ err is (D, N) @ (N,), which gives (D,).","Update w and b only after computing both gradients from the same err."],
starter:`import numpy as np

def fit_linreg(X, y, lr=0.1, steps=500):
    pass
`,
solution:`import numpy as np

def fit_linreg(X, y, lr=0.1, steps=500):
    N, D = X.shape
    w, b = np.zeros(D), 0.0
    for _ in range(steps):
        err = X @ w + b - y
        dw = 2 / N * X.T @ err
        db = 2 / N * err.sum()
        w -= lr * dw
        b -= lr * db
    return w, b
`,
tests:`import numpy as np
rng = np.random.default_rng(1)
X = rng.normal(size=(200, 3))
y = X @ np.array([2.0, -1.0, 0.5]) + 3.0
w, b = fit_linreg(X, y)
check(np.shape(w) == (3,), "w has shape (3,)")
check(np.allclose(w, [2, -1, 0.5], atol=0.02), "recovers w = [2, -1, 0.5], got " + str(np.round(w, 3)))
check(abs(b - 3) < 0.02, "recovers b = 3")`},

{stage:"13 · NumPy", id:"np_attention", title:"Batched attention with a mask", packages:["numpy"],
learn:`<p>Real attention runs on a batch. With Q, K and V all of shape <code>(B, T, d)</code>:</p>
<ul><li><code>K.transpose(0, 2, 1)</code> is (B, d, T). Or use <code>np.swapaxes(K, -1, -2)</code>.</li><li><code>Q @ Kᵀ</code> is (B, T, T). The <code>@</code> operator multiplies the last two dimensions and treats the leading ones as a batch.</li><li>A causal mask is <code>np.tril(np.ones((T, T)))</code>. Put <code>-np.inf</code> where it's 0, using <code>np.where</code>.</li></ul>
<p>Saying these shapes out loud is half of a good interview answer.</p>`,
example:`import numpy as np
B, T, d = 2, 4, 8
Q = np.random.randn(B, T, d)
print((Q @ np.swapaxes(Q, -1, -2)).shape)
print(np.tril(np.ones((T, T))))`,
why:"The vectorized version is what interviewers expect once you've shown you know the equation.",
task:"Write <code>attention(Q, K, V, causal=False)</code> for inputs of shape (B, T, d). Return softmax(QKᵀ/√d)V with shape (B, T, d). If <code>causal</code>, block attention to future positions.",
pseudo:`d = Q.shape[-1]
S = Q @ swapaxes(K, -1, -2) / sqrt(d)      # (B, T, T)
if causal:
    mask = tril(ones((T, T)))
    S = where(mask == 0, -inf, S)
W = softmax over last axis
return W @ V`,
hints:["Write a stable softmax inline: subtract S.max(-1, keepdims=True).","The (T, T) mask broadcasts across the batch dimension automatically."],
starter:`import numpy as np

def attention(Q, K, V, causal=False):
    pass
`,
solution:`import numpy as np

def attention(Q, K, V, causal=False):
    d = Q.shape[-1]
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if causal:
        T = Q.shape[-2]
        S = np.where(np.tril(np.ones((T, T))) == 0, -np.inf, S)
    S = S - S.max(axis=-1, keepdims=True)
    W = np.exp(S)
    W = W / W.sum(axis=-1, keepdims=True)
    return W @ V
`,
tests:`import numpy as np
rng = np.random.default_rng(0)
Q, K, V = rng.normal(size=(2, 5, 4)), rng.normal(size=(2, 5, 4)), rng.normal(size=(2, 5, 4))
out = attention(Q, K, V)
check(out.shape == (2, 5, 4), "output shape is (B, T, d)")
o2 = attention(np.zeros((1, 3, 2)), np.zeros((1, 3, 2)), np.array([[[1.0, 0], [2, 0], [3, 0]]]))
check(np.allclose(o2[0, :, 0], 2.0), "equal scores average V evenly")
c = attention(Q, K, V, causal=True)
check(np.allclose(c[:, 0], V[:, 0]), "with causal=True, token 0 only sees itself")
V2 = V.copy(); V2[:, 3:] += 100
check(np.allclose(attention(Q, K, V2, causal=True)[:, :3], c[:, :3]), "changing future tokens doesn't affect earlier outputs")`},
];
