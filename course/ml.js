// Part 2: ML coding. The from-scratch implementations ML interviews ask for, in plain Python.
const ML_LESSONS = [
{stage:"5 · ML warm-up", id:"argmax", title:"Argmax",
why:"A classifier outputs a score per class. The predicted class is the index of the largest score.",
task:"Write <code>argmax(xs)</code> that returns the <b>index</b> of the largest value. If there's a tie, return the first one.",
pseudo:`best_i = 0
for i from 1 to len(xs)-1:
    if xs[i] > xs[best_i]:
        best_i = i
return best_i`,
hints:["Track the index, not the value.","range(1, len(xs)) gives you every index after the first.","Use > (not >=) so ties keep the earlier index."],
starter:`def argmax(xs):
    pass
`,
solution:`def argmax(xs):
    best_i = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best_i]:
            best_i = i
    return best_i
`,
tests:`check(argmax([0.1, 0.7, 0.2]) == 1, "argmax([0.1,0.7,0.2]) should be 1")
check(argmax([5]) == 0, "single element")
check(argmax([3, 9, 9]) == 1, "ties return the first index")
check(argmax([-5, -1, -3]) == 1, "works with negatives")`},

{stage:"5 · ML warm-up", id:"word_counts", title:"Count tokens",
why:"Building a vocabulary for a text model starts with counting how often each token appears.",
task:"Write <code>count_tokens(text)</code> that lowercases the text, splits on whitespace, and returns a dict of token → count.",
pseudo:`counts = empty dict
for each word in text.lower().split():
    counts[word] = counts.get(word, 0) + 1
return counts`,
hints:["text.lower().split() gives you a list of words.","dict.get(key, 0) returns 0 when the key isn't there yet."],
starter:`def count_tokens(text):
    pass
`,
solution:`def count_tokens(text):
    counts = {}
    for w in text.lower().split():
        counts[w] = counts.get(w, 0) + 1
    return counts
`,
tests:`r = count_tokens("The cat saw the dog")
check(r["the"] == 2, "'the' appears twice (case-insensitive)")
check(r["cat"] == 1, "'cat' appears once")
check(len(r) == 4, "there should be 4 distinct tokens")
check(count_tokens("") == {}, "empty text gives an empty dict")`},

{stage:"5 · ML warm-up", id:"vocab", title:"Token ids",
why:"Models eat numbers, not strings. You map each token to an integer id, then encode sentences as lists of ids.",
task:"Write <code>build_vocab(tokens)</code> that gives each <b>new</b> token the next id, starting at 0, in order of first appearance. Then write <code>encode(tokens, vocab)</code> that returns the list of ids.",
pseudo:`vocab = {}
for t in tokens:
    if t not in vocab:
        vocab[t] = len(vocab)

encode: [vocab[t] for t in tokens]`,
hints:["len(vocab) is exactly the next unused id.","A list comprehension makes encode one line."],
starter:`def build_vocab(tokens):
    pass

def encode(tokens, vocab):
    pass
`,
solution:`def build_vocab(tokens):
    vocab = {}
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab

def encode(tokens, vocab):
    return [vocab[t] for t in tokens]
`,
tests:`v = build_vocab(["a", "b", "a", "c"])
check(v == {"a": 0, "b": 1, "c": 2}, "ids follow first appearance")
check(encode(["c", "a", "a"], v) == [2, 0, 0], "encode maps tokens to ids")`},

{stage:"6 · Vectors", id:"dot", title:"Dot product",
why:"The most important operation in ML. Attention scores, linear layers and similarity are all dot products.",
task:"Write <code>dot(a, b)</code> for two equal-length lists. Raise a <code>ValueError</code> if the lengths differ.",
pseudo:`if len(a) != len(b): raise ValueError
s = 0
for i in range(len(a)):
    s += a[i] * b[i]
return s`,
hints:["Check the lengths first, before the loop.","zip(a, b) pairs up items so you can write: for x, y in zip(a, b)."],
starter:`def dot(a, b):
    pass
`,
solution:`def dot(a, b):
    if len(a) != len(b):
        raise ValueError("length mismatch")
    s = 0
    for x, y in zip(a, b):
        s += x * y
    return s
`,
tests:`check(dot([1, 2, 3], [4, 5, 6]) == 32, "1*4 + 2*5 + 3*6 = 32")
check(dot([], []) == 0, "empty vectors give 0")
raised = False
try:
    dot([1, 2], [1])
except ValueError:
    raised = True
check(raised, "mismatched lengths should raise ValueError")`},

{stage:"6 · Vectors", id:"mean_std", title:"Mean and standard deviation",
why:"You use these for feature scaling, BatchNorm and LayerNorm, and for reading your training metrics.",
task:"Write <code>mean(xs)</code> and <code>std(xs)</code>. Use the population formula: variance = average of (x − mean)².",
pseudo:`mean = sum(xs) / len(xs)
var  = sum((x - mean)^2 for x in xs) / len(xs)
std  = sqrt(var)`,
hints:["import math, then math.sqrt.","Compute the mean once and reuse it inside std."],
starter:`import math

def mean(xs):
    pass

def std(xs):
    pass
`,
solution:`import math

def mean(xs):
    return sum(xs) / len(xs)

def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
`,
tests:`check(close(mean([1, 2, 3, 4]), 2.5), "mean of [1,2,3,4] is 2.5")
check(close(std([2, 4, 4, 4, 5, 5, 7, 9]), 2.0), "std of the classic example is 2.0")
check(close(std([3, 3, 3]), 0.0), "constant list has std 0")`},

{stage:"6 · Vectors", id:"zscore", title:"Standardize features",
why:"Gradient descent behaves much better when features have mean 0 and std 1. This is also the core of LayerNorm.",
task:"Write <code>standardize(xs, eps=1e-8)</code> that returns <code>(x − mean) / (std + eps)</code> for each x. Write mean and std inline or copy them from the last exercise.",
pseudo:`m = mean(xs); s = std(xs)
return [(x - m) / (s + eps) for x in xs]`,
hints:["eps stops a divide-by-zero when every value is the same.","This is LayerNorm without the learned scale and shift."],
starter:`import math

def standardize(xs, eps=1e-8):
    pass
`,
solution:`import math

def standardize(xs, eps=1e-8):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return [(x - m) / (s + eps) for x in xs]
`,
tests:`z = standardize([1.0, 2.0, 3.0])
check(close(sum(z) / 3, 0.0), "result should have mean 0")
check(close(z[2], 1.2247448), "z[2] should be about 1.2247")
c = standardize([5.0, 5.0])
check(close(c[0], 0.0), "constant input gives zeros, not a crash")`},

{stage:"7 · Matrices", id:"transpose", title:"Transpose",
why:"K transposed in QKᵀ. You'll flip matrices constantly, and you need to know the shapes cold.",
task:"Write <code>transpose(M)</code> for a matrix stored as a list of rows. A 2×3 matrix becomes 3×2.",
pseudo:`rows = len(M); cols = len(M[0])
result[j][i] = M[i][j]
for j in range(cols): build row j from column j of M`,
hints:["The output has len(M[0]) rows.","Row j of the output is [M[i][j] for i in range(len(M))]."],
starter:`def transpose(M):
    pass
`,
solution:`def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]
`,
tests:`check(transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]], "2x3 becomes 3x2")
check(transpose([[7]]) == [[7]], "1x1 is unchanged")`},

{stage:"7 · Matrices", id:"matvec", title:"Matrix times vector",
why:"A linear layer is y = Wx + b. This is the Wx part.",
task:"Write <code>matvec(W, x)</code>. W has shape (out, in) and x has length in. Return a list of length out.",
pseudo:`for each row in W:
    output.append(dot(row, x))`,
hints:["Each output number is the dot product of one row of W with x.","Shape check: len(W[0]) must equal len(x)."],
starter:`def matvec(W, x):
    pass
`,
solution:`def matvec(W, x):
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]
`,
tests:`check(matvec([[1, 0], [0, 1]], [3, 4]) == [3, 4], "identity leaves x unchanged")
check(matvec([[1, 2], [3, 4], [5, 6]], [1, 1]) == [3, 7, 11], "(3,2) times (2,) gives (3,)")`},

{stage:"7 · Matrices", id:"matmul", title:"Matrix multiply",
why:"Interviewers ask for this all the time. It's also how a batch goes through a layer, and how attention computes QKᵀ.",
task:"Write <code>matmul(A, B)</code>. A is (n, k), B is (k, m), and the result is (n, m). Raise <code>ValueError</code> if the inner sizes don't match.",
pseudo:`check len(A[0]) == len(B)
C = n x m zeros
for i in range(n):
  for j in range(m):
    for p in range(k):
      C[i][j] += A[i][p] * B[p][j]`,
hints:["Three nested loops: rows of A, columns of B, then the shared dimension.","Build C with [[0]*m for _ in range(n)]. Don't use [[0]*m]*n, because every row would be the same list.","C[i][j] is the dot product of row i of A and column j of B."],
starter:`def matmul(A, B):
    pass
`,
solution:`def matmul(A, B):
    n, k, m = len(A), len(A[0]), len(B[0])
    if k != len(B):
        raise ValueError("shape mismatch")
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C
`,
tests:`check(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]], "2x2 @ 2x2")
check(matmul([[1, 2, 3]], [[1], [1], [1]]) == [[6]], "1x3 @ 3x1 gives 1x1")
raised = False
try:
    matmul([[1, 2]], [[1, 2]])
except ValueError:
    raised = True
check(raised, "(1,2) @ (1,2) should raise ValueError")`},

{stage:"8 · Core ML functions", id:"activations", title:"ReLU and sigmoid",
why:"These are the two activations you're most likely to be asked about. Sigmoid also turns a logit into a probability.",
task:"Write <code>relu(x)</code> and <code>sigmoid(x)</code> for a single number. sigmoid(x) = 1 / (1 + e<sup>−x</sup>).",
pseudo:`relu(x)    = max(0, x)
sigmoid(x) = 1 / (1 + exp(-x))`,
hints:["math.exp(-x).","Advanced: for very negative x, exp(-x) overflows. You can use exp(x) / (1 + exp(x)) when x < 0."],
starter:`import math

def relu(x):
    pass

def sigmoid(x):
    pass
`,
solution:`import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)
`,
tests:`check(relu(-3) == 0 and relu(2.5) == 2.5, "relu clips negatives to 0")
check(close(sigmoid(0), 0.5), "sigmoid(0) = 0.5")
check(close(sigmoid(2), 0.8807971), "sigmoid(2) is about 0.8808")
check(sigmoid(-1000) >= 0, "sigmoid(-1000) should not crash")`},

{stage:"8 · Core ML functions", id:"softmax", title:"Stable softmax",
why:"The single most-asked ML coding question. The trick that matters is subtracting the max.",
task:"Write <code>softmax(xs)</code> that turns scores into probabilities that sum to 1. Make it <b>numerically stable</b> so it works for huge inputs like [1000, 1001].",
pseudo:`m = max(xs)
exps = [exp(x - m) for x in xs]
total = sum(exps)
return [e / total for e in exps]`,
hints:["exp(1000) overflows. Subtracting the max first gives the same answer without overflowing.","Why it's the same: e^(x−m) / Σe^(x−m) = e^x / Σe^x, because the e^(−m) cancels.","Say this out loud in the interview. They're listening for it."],
starter:`import math

def softmax(xs):
    pass
`,
solution:`import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
`,
tests:`p = softmax([1.0, 2.0, 3.0])
check(close(sum(p), 1.0), "probabilities sum to 1")
check(close(p[2], 0.6652409), "p[2] is about 0.6652")
q = softmax([1000.0, 1001.0])
check(close(q[1], 0.7310586), "stable for large inputs")`},

{stage:"8 · Core ML functions", id:"cross_entropy", title:"Cross-entropy loss",
why:"This is the classification loss. Expect questions like: what is it, why the log, what happens at p = 0?",
task:"Write <code>cross_entropy(probs, label)</code>, the loss for one example: −log(probs[label]). Clip the probability to at least 1e-12 so log(0) doesn't crash.",
pseudo:`p = max(probs[label], 1e-12)
return -log(p)`,
hints:["Only the probability of the correct class matters.","math.log is the natural log."],
starter:`import math

def cross_entropy(probs, label):
    pass
`,
solution:`import math

def cross_entropy(probs, label):
    return -math.log(max(probs[label], 1e-12))
`,
tests:`check(close(cross_entropy([0.1, 0.7, 0.2], 1), 0.3566749), "-log(0.7) is about 0.3567")
check(close(cross_entropy([0.0, 1.0], 1), 0.0), "a perfect prediction gives 0 loss")
check(cross_entropy([1.0, 0.0], 1) > 20, "p = 0 gives a large finite loss, not a crash")`},

{stage:"8 · Core ML functions", id:"metrics", title:"Accuracy, precision, recall",
why:"Interviewers love asking you to compute these by hand, then asking when accuracy is misleading.",
task:"Write <code>metrics(y_true, y_pred)</code> for binary labels (0/1). Return a tuple <code>(accuracy, precision, recall)</code>. Return 0.0 for precision or recall if the denominator is 0.",
pseudo:`count tp, fp, fn, tn in one loop
accuracy  = (tp + tn) / n
precision = tp / (tp + fp)   # of predicted positives, how many were right
recall    = tp / (tp + fn)   # of actual positives, how many we found`,
hints:["tp: true 1, predicted 1. fp: true 0, predicted 1. fn: true 1, predicted 0.","Guard both divisions."],
starter:`def metrics(y_true, y_pred):
    pass
`,
solution:`def metrics(y_true, y_pred):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
        else: tn += 1
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return acc, prec, rec
`,
tests:`a, p, r = metrics([1, 0, 1, 1, 0], [1, 1, 0, 1, 0])
check(close(a, 0.6), "accuracy should be 3/5")
check(close(p, 2/3), "precision should be 2/3")
check(close(r, 2/3), "recall should be 2/3")
a, p, r = metrics([0, 0], [0, 0])
check(p == 0.0 and r == 0.0, "no positives gives 0.0, not a crash")`},

{stage:"9 · Layers as classes", id:"linear", title:"A Linear layer class",
why:"This is exactly the shape of the question you froze on: a class with __init__ and forward. Learn the skeleton.",
task:"Write a class <code>Linear</code>. <code>__init__(self, W, b)</code> stores weights of shape (out, in) and a bias of length out. <code>forward(self, x)</code> returns Wx + b for a single vector x.",
pseudo:`class Linear:
    __init__: self.W = W; self.b = b
    forward(x):
        out = []
        for each row, bias in zip(W, b):
            out.append(dot(row, x) + bias)
        return out`,
hints:["Always write the skeleton first: class line, __init__, forward. Then fill it in.","Everything you store needs self. in front of it.","This is your matvec plus a bias."],
starter:`class Linear:
    def __init__(self, W, b):
        pass

    def forward(self, x):
        pass
`,
solution:`class Linear:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return [sum(w * xi for w, xi in zip(row, x)) + bi
                for row, bi in zip(self.W, self.b)]
`,
tests:`layer = Linear([[1, 2], [3, 4]], [0.5, -1])
check(layer.forward([1, 1]) == [3.5, 6], "Wx + b with x = [1,1]")
check(layer.forward([0, 0]) == [0.5, -1], "zero input returns the bias")`},

{stage:"9 · Layers as classes", id:"scaler", title:"A fit/transform scaler",
why:"The scikit-learn fit/transform pattern. It also tests whether you know to fit on training data only.",
task:"Write a class <code>StandardScaler</code> with <code>fit(xs)</code> (stores mean and std, returns self) and <code>transform(xs)</code> (applies the stored values). Use eps = 1e-8 in the division.",
pseudo:`fit(xs):
    self.mean = mean(xs); self.std = std(xs); return self
transform(xs):
    return [(x - self.mean) / (self.std + 1e-8) for x in xs]`,
hints:["transform must use the stored stats, not recompute them from new data.","Returning self from fit lets you chain: StandardScaler().fit(train).transform(test)."],
starter:`import math

class StandardScaler:
    def fit(self, xs):
        pass

    def transform(self, xs):
        pass
`,
solution:`import math

class StandardScaler:
    def fit(self, xs):
        self.mean = sum(xs) / len(xs)
        self.std = math.sqrt(sum((x - self.mean) ** 2 for x in xs) / len(xs))
        return self

    def transform(self, xs):
        return [(x - self.mean) / (self.std + 1e-8) for x in xs]
`,
tests:`s = StandardScaler().fit([0.0, 2.0])
check(close(s.mean, 1.0) and close(s.std, 1.0), "fit stores mean 1 and std 1")
check(close(s.transform([3.0])[0], 2.0), "transform uses the stored stats: (3-1)/1 = 2")`},

{stage:"10 · Training loops", id:"mse", title:"MSE loss and its gradient",
why:"Regression loss, and the first gradient you should be able to derive on a whiteboard.",
task:"Write <code>mse(y, y_hat)</code> = mean of (y_hat − y)². Then write <code>mse_grad(y, y_hat)</code> that returns the derivative with respect to each y_hat: 2·(y_hat − y)/n.",
pseudo:`n = len(y)
mse  = sum((yh - yt)^2) / n
grad = [2 * (yh - yt) / n for each pair]`,
hints:["Derivative of (u)² is 2u, and the 1/n from the mean stays."],
starter:`def mse(y, y_hat):
    pass

def mse_grad(y, y_hat):
    pass
`,
solution:`def mse(y, y_hat):
    return sum((p - t) ** 2 for t, p in zip(y, y_hat)) / len(y)

def mse_grad(y, y_hat):
    n = len(y)
    return [2 * (p - t) / n for t, p in zip(y, y_hat)]
`,
tests:`check(close(mse([1, 2], [1, 4]), 2.0), "((0)^2 + (2)^2) / 2 = 2")
check(mse_grad([1, 2], [1, 4]) == [0.0, 2.0], "gradient is 2*(y_hat - y)/n")`},

{stage:"10 · Training loops", id:"gd", title:"Gradient descent: linear regression",
why:"You will be asked to write a training loop. Here is the whole loop, with no libraries, in about 10 lines.",
task:"Write <code>fit_line(xs, ys, lr=0.05, steps=2000)</code> that learns w and b for y ≈ w·x + b by gradient descent on MSE. Return <code>(w, b)</code>.",
pseudo:`w, b = 0, 0
repeat steps times:
    preds = [w*x + b for x in xs]
    errors = [p - y for p, y in zip(preds, ys)]
    dw = 2/n * sum(e * x)
    db = 2/n * sum(e)
    w -= lr * dw
    b -= lr * db
return w, b`,
hints:["Every training loop has the same four steps: forward, loss/error, gradient, update.","dw uses error times x. db uses the error alone.","Data generated from y = 2x + 1 should give w ≈ 2 and b ≈ 1."],
starter:`def fit_line(xs, ys, lr=0.05, steps=2000):
    pass
`,
solution:`def fit_line(xs, ys, lr=0.05, steps=2000):
    w, b = 0.0, 0.0
    n = len(xs)
    for _ in range(steps):
        errs = [(w * x + b) - y for x, y in zip(xs, ys)]
        dw = 2 / n * sum(e * x for e, x in zip(errs, xs))
        db = 2 / n * sum(errs)
        w -= lr * dw
        b -= lr * db
    return w, b
`,
tests:`xs = [0, 1, 2, 3, 4]
ys = [2 * x + 1 for x in xs]
w, b = fit_line(xs, ys)
check(abs(w - 2) < 0.01, "w should be close to 2 (got " + str(round(w, 3)) + ")")
check(abs(b - 1) < 0.01, "b should be close to 1 (got " + str(round(b, 3)) + ")")`},

{stage:"10 · Training loops", id:"logreg", title:"Logistic regression step",
why:"Binary classification from scratch. The gradient turns out to be neat: (p − y)·x.",
task:"Write <code>logreg_step(w, b, X, y, lr)</code> that does one gradient step for logistic regression. X is a list of feature vectors, y is 0/1 labels. Return the new <code>(w, b)</code>.",
pseudo:`for each example i:
    p = sigmoid(dot(w, X[i]) + b)
    err = p - y[i]
    dw[j] += err * X[i][j]   for each feature j
    db    += err
average dw and db by n, then
w = w - lr * dw;  b = b - lr * db`,
hints:["With sigmoid + cross-entropy, the gradient wrt the logit is just p − y.","Keep dw as a list the same length as w."],
starter:`import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def logreg_step(w, b, X, y, lr):
    pass
`,
solution:`import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def logreg_step(w, b, X, y, lr):
    n = len(X)
    dw = [0.0] * len(w)
    db = 0.0
    for xi, yi in zip(X, y):
        p = sigmoid(sum(wj * xj for wj, xj in zip(w, xi)) + b)
        err = p - yi
        for j in range(len(w)):
            dw[j] += err * xi[j]
        db += err
    w = [wj - lr * d / n for wj, d in zip(w, dw)]
    b = b - lr * db / n
    return w, b
`,
tests:`w, b = logreg_step([0.0], 0.0, [[1.0], [-1.0]], [1, 0], 1.0)
check(close(w[0], 0.5), "first step from zeros gives w = 0.5")
check(close(b, 0.0), "symmetric data keeps b = 0")
w, b = [0.0, 0.0], 0.0
X = [[2, 1], [1, 3], [-1, -2], [-2, -1]]
Y = [1, 1, 0, 0]
for _ in range(200):
    w, b = logreg_step(w, b, X, Y, 0.5)
preds = [1 if sigmoid(w[0]*x[0] + w[1]*x[1] + b) > 0.5 else 0 for x in X]
check(preds == Y, "after 200 steps it should separate the toy data")`},

{stage:"11 · Classic algorithms", id:"knn", title:"k-nearest neighbors",
why:"A common 'implement from scratch' question. It tests distances, sorting with a key, and voting.",
task:"Write <code>knn_predict(X_train, y_train, x, k)</code>. Use Euclidean distance, take the k closest training points, and return the most common label among them.",
pseudo:`dists = [(distance(x, xi), yi) for each training pair]
sort dists by distance
take first k labels
return most common label`,
hints:["sorted(list_of_pairs, key=lambda t: t[0]).","You can skip the sqrt. It doesn't change the ordering.","Count votes with a dict, then pick the label with the highest count."],
starter:`def knn_predict(X_train, y_train, x, k):
    pass
`,
solution:`def knn_predict(X_train, y_train, x, k):
    d = []
    for xi, yi in zip(X_train, y_train):
        dist = sum((a - c) ** 2 for a, c in zip(xi, x))
        d.append((dist, yi))
    d = sorted(d, key=lambda t: t[0])[:k]
    votes = {}
    for _, label in d:
        votes[label] = votes.get(label, 0) + 1
    return max(votes, key=lambda lab: votes[lab])
`,
tests:`X = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]]
Y = ["a", "a", "a", "b", "b", "b"]
check(knn_predict(X, Y, [0.2, 0.2], 3) == "a", "point near the origin is 'a'")
check(knn_predict(X, Y, [5.5, 5.5], 3) == "b", "point near (5,5) is 'b'")
check(knn_predict(X, Y, [4, 4], 1) == "b", "k=1 uses the single nearest point")`},

{stage:"11 · Classic algorithms", id:"kmeans", title:"k-means (one iteration)",
why:"Shows you can write the assign-then-update loop that a lot of algorithms share.",
task:"Write <code>kmeans_step(points, centroids)</code>. Assign each point to its nearest centroid, then move each centroid to the mean of its points. Return the new centroids. If a centroid gets no points, keep it where it was.",
pseudo:`groups = [[] for each centroid]
for p in points:
    c = index of nearest centroid
    groups[c].append(p)
new = []
for i, g in enumerate(groups):
    if g empty: keep centroids[i]
    else: average of g, coordinate by coordinate`,
hints:["The nearest centroid is an argmin over distances.","Coordinate mean: [sum(p[d] for p in g) / len(g) for d in range(dims)]."],
starter:`def kmeans_step(points, centroids):
    pass
`,
solution:`def kmeans_step(points, centroids):
    groups = [[] for _ in centroids]
    for p in points:
        best, best_d = 0, None
        for i, c in enumerate(centroids):
            d = sum((a - b) ** 2 for a, b in zip(p, c))
            if best_d is None or d < best_d:
                best, best_d = i, d
        groups[best].append(p)
    new = []
    for i, g in enumerate(groups):
        if not g:
            new.append(centroids[i])
        else:
            new.append([sum(p[d] for p in g) / len(g) for d in range(len(g[0]))])
    return new
`,
tests:`pts = [[0, 0], [0, 2], [10, 10], [10, 12]]
c = kmeans_step(pts, [[1, 1], [9, 9]])
check(c == [[0.0, 1.0], [10.0, 11.0]], "centroids move to their cluster means")
c = kmeans_step([[0, 0]], [[0, 0], [100, 100]])
check(c[1] == [100, 100], "an empty cluster keeps its centroid")`},

{stage:"11 · Classic algorithms", id:"batches", title:"Mini-batches",
why:"Every training loop iterates over batches. Interviewers check that you handle the last, smaller batch.",
task:"Write <code>batches(data, batch_size)</code> that returns a list of chunks. The last chunk may be smaller.",
pseudo:`out = []
for start in range(0, len(data), batch_size):
    out.append(data[start : start + batch_size])
return out`,
hints:["range(0, n, step) jumps by step.","Slicing past the end of a list is safe in Python."],
starter:`def batches(data, batch_size):
    pass
`,
solution:`def batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
`,
tests:`check(batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]], "the last batch is smaller")
check(batches([], 3) == [], "no data gives no batches")`},

{stage:"12 · Attention", id:"attention", title:"Scaled dot-product attention",
why:"The question you froze on. You've now written every piece: matmul, transpose, softmax. This exercise just puts them together.",
task:"Write <code>attention(Q, K, V)</code> for one sequence. Q and K are (T, d), V is (T, d_v). Return softmax(QKᵀ / √d) · V, shape (T, d_v). Helpers are already written for you.",
pseudo:`d = len(Q[0])
scores = matmul(Q, transpose(K))        # (T, T)
scale every score by 1 / sqrt(d)
weights = [softmax(row) for row in scores]  # each row sums to 1
return matmul(weights, V)               # (T, d_v)`,
hints:["Say the shapes out loud first: (T,d) @ (d,T) → (T,T), then (T,T) @ (T,d_v) → (T,d_v).","Softmax goes across each row, so each query's weights over all keys sum to 1.","Scale before softmax, not after."],
starter:`import math

def transpose(M):
    return [list(r) for r in zip(*M)]

def matmul(A, B):
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]

def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]

def attention(Q, K, V):
    pass
`,
solution:`import math

def transpose(M):
    return [list(r) for r in zip(*M)]

def matmul(A, B):
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]

def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]

def attention(Q, K, V):
    d = len(Q[0])
    scores = matmul(Q, transpose(K))
    scores = [[s / math.sqrt(d) for s in row] for row in scores]
    weights = [softmax(row) for row in scores]
    return matmul(weights, V)
`,
tests:`out = attention([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]])
check(len(out) == 2 and len(out[0]) == 2, "output shape should be (2, 2)")
check(close(out[0][0], 1.6604769), "out[0][0] is about 1.6605")
check(close(out[1][1], 3.3395231), "out[1][1] is about 3.3395")
same = attention([[0, 0]], [[1, 1], [2, 2]], [[10], [20]])
check(close(same[0][0], 15.0), "equal scores average V evenly")`},

{stage:"12 · Attention", id:"causal", title:"Causal mask",
why:"GPT-style models can't look at future tokens. This is the follow-up question that comes right after attention.",
task:"Write <code>causal_attention(Q, K, V)</code>. Same as before, but before the softmax set every score where key index j > query index i to −infinity.",
pseudo:`scores as before (scaled)
for i in range(T):
    for j in range(T):
        if j > i: scores[i][j] = -inf
softmax rows, then multiply by V`,
hints:["float('-inf') works, and exp(-inf - m) is 0.","Row 0 can only attend to itself, so out[0] must equal V[0] exactly."],
starter:`import math

def transpose(M):
    return [list(r) for r in zip(*M)]

def matmul(A, B):
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]

def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]

def causal_attention(Q, K, V):
    pass
`,
solution:`import math

def transpose(M):
    return [list(r) for r in zip(*M)]

def matmul(A, B):
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]

def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]

def causal_attention(Q, K, V):
    d = len(Q[0])
    T = len(Q)
    scores = matmul(Q, transpose(K))
    for i in range(T):
        for j in range(T):
            scores[i][j] = scores[i][j] / math.sqrt(d) if j <= i else float("-inf")
    weights = [softmax(row) for row in scores]
    return matmul(weights, V)
`,
tests:`Q = [[1, 0], [0, 1], [1, 1]]
V = [[1, 0], [0, 1], [5, 5]]
out = causal_attention(Q, Q, V)
check(close(out[0][0], 1.0) and close(out[0][1], 0.0), "token 0 only sees itself")
check(out[1][0] < 1 and out[1][1] > 0, "token 1 mixes tokens 0 and 1")
check(out[1][0] + out[1][1] < 1.0001, "token 1 cannot see token 2 (no 5s leaking in)")`},
];
