// Part 1: Foundations. Each lesson teaches one idea, shows a runnable example, then asks you to write code.
// Fields: stage, id, title, learn (HTML), example (runnable Python), why, task, pseudo, hints[], starter, solution, tests
const FOUNDATION_LESSONS = [
{stage:"1 · Values and names", id:"variables", title:"Variables and types",
learn:`<p>A <b>variable</b> is a name that points at a value. You create one with <code>=</code>. Python works out the type on its own.</p>
<p>The four types you'll use constantly:</p>
<ul><li><code>int</code>: whole numbers, like <code>3</code></li><li><code>float</code>: decimals, like <code>0.01</code> (a learning rate)</li><li><code>str</code>: text, like <code>"cat"</code></li><li><code>bool</code>: <code>True</code> or <code>False</code></li></ul>
<p><code>/</code> always gives a float (<code>7 / 2 == 3.5</code>). <code>//</code> divides and drops the remainder (<code>7 // 2 == 3</code>). <code>%</code> gives the remainder (<code>7 % 2 == 1</code>). <code>**</code> is power (<code>2 ** 3 == 8</code>).</p>
<p><code>print()</code> shows a value. <code>type()</code> tells you what kind of value it is. When you're confused, print things.</p>`,
example:`lr = 0.01
epochs = 10
name = "model_v1"
print(type(lr), type(epochs), type(name))
print(7 / 2, 7 // 2, 7 % 2, 2 ** 3)
print("steps:", epochs * 100)`,
why:"Nothing works until you can store a value, do math on it and print it to check yourself.",
task:"Write <code>average3(a, b, c)</code> that returns the average of three numbers.",
pseudo:`return (a + b + c) / 3`,
hints:["Add first, then divide. Parentheses control the order.","Use /, not //, so you keep the decimals."],
starter:`def average3(a, b, c):
    # replace pass with your code
    pass
`,
solution:`def average3(a, b, c):
    return (a + b + c) / 3
`,
tests:`check(average3(1, 2, 3) == 2, "average3(1, 2, 3) should be 2")
check(close(average3(1, 2, 2), 1.6666667), "average3(1, 2, 2) should be about 1.667")`},

{stage:"1 · Values and names", id:"functions", title:"Functions and return",
learn:`<p>A <b>function</b> is a named, reusable piece of code. <code>def</code> starts it, the names in brackets are its <b>parameters</b>, and the indented block is its body.</p>
<p><code>return</code> sends a value back and ends the function. A function with no <code>return</code> gives back <code>None</code>. Printing a value is <b>not</b> the same as returning it. This is the most common beginner bug.</p>
<p>Parameters can have defaults: <code>def step(w, lr=0.1)</code> lets you call <code>step(w)</code> or <code>step(w, 0.5)</code>.</p>
<p>Indentation is part of the syntax. Use 4 spaces. Everything indented under the <code>def</code> belongs to it.</p>`,
example:`def double(x):
    return x * 2

def say_double(x):
    print(x * 2)   # prints, but returns None

a = double(5)
b = say_double(5)
print("a =", a, "| b =", b)`,
why:"Every interview answer is a function. You need to write def, parameters and return without thinking.",
task:"Write <code>sgd_update(w, grad, lr=0.1)</code> that returns the new weight: <code>w − lr × grad</code>.",
pseudo:`return w - lr * grad`,
hints:["The default value goes in the def line: lr=0.1.","Return the value. Don't print it."],
starter:`def sgd_update(w, grad, lr=0.1):
    pass
`,
solution:`def sgd_update(w, grad, lr=0.1):
    return w - lr * grad
`,
tests:`check(close(sgd_update(1.0, 2.0), 0.8), "1.0 - 0.1*2.0 = 0.8")
check(close(sgd_update(1.0, 2.0, lr=0.5), 0.0), "custom lr works")
check(sgd_update(1.0, 2.0) is not None, "the function must return, not print")`},

{stage:"2 · Control flow", id:"conditions", title:"if, elif, else",
learn:`<p><code>if</code> runs a block only when its condition is <code>True</code>. <code>elif</code> checks another condition if the first failed. <code>else</code> catches everything left.</p>
<p>Comparisons: <code>==</code> (equal), <code>!=</code> (not equal), <code>&lt;</code>, <code>&lt;=</code>, <code>&gt;</code>, <code>&gt;=</code>. Combine them with <code>and</code>, <code>or</code>, <code>not</code>.</p>
<p>Python checks the branches top to bottom and runs only the first one that matches, so order matters.</p>`,
example:`loss = 0.42
if loss < 0.1:
    print("great")
elif loss < 0.5:
    print("ok")
else:
    print("keep training")`,
why:"Clipping, early stopping, masks and edge cases are all if-statements.",
task:"Write <code>clip(x, lo, hi)</code> that returns <code>lo</code> if x is below lo, <code>hi</code> if x is above hi, and x otherwise. (Gradient clipping works like this.)",
pseudo:`if x < lo: return lo
elif x > hi: return hi
else: return x`,
hints:["Three cases, so if / elif / else.","You can return inside each branch."],
starter:`def clip(x, lo, hi):
    pass
`,
solution:`def clip(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    return x
`,
tests:`check(clip(5, 0, 1) == 1, "5 clipped to [0,1] is 1")
check(clip(-2, 0, 1) == 0, "-2 clipped to [0,1] is 0")
check(clip(0.3, 0, 1) == 0.3, "values inside the range are unchanged")`},

{stage:"2 · Control flow", id:"loops", title:"for and while loops",
learn:`<p>A <code>for</code> loop runs once per item: <code>for x in xs:</code>. To loop over numbers, use <code>range</code>: <code>range(5)</code> is 0,1,2,3,4 and <code>range(2, 10, 2)</code> is 2,4,6,8.</p>
<p>The <b>accumulator pattern</b> shows up everywhere: make a variable before the loop, update it inside, use it after.</p>
<p><code>enumerate(xs)</code> gives you the index and the item together. <code>zip(a, b)</code> walks two lists side by side.</p>
<p>A <code>while</code> loop runs as long as a condition is true. Make sure something inside changes the condition, or it never stops.</p>`,
example:`losses = [0.9, 0.6, 0.4]
total = 0
for i, loss in enumerate(losses):
    total += loss
    print("epoch", i, "loss", loss)
print("sum", total)

for a, b in zip([1, 2], [10, 20]):
    print(a + b)`,
why:"Sums, means, dot products and training loops are all loops with an accumulator.",
task:"Write <code>total(xs)</code> that returns the sum of a list without using the built-in <code>sum()</code>.",
pseudo:`running = 0
for each x in xs:
    running = running + x
return running`,
hints:["Start a variable at 0 before the loop.","Return after the loop, not inside it. Watch your indentation."],
starter:`def total(xs):
    pass
`,
solution:`def total(xs):
    running = 0
    for x in xs:
        running += x
    return running
`,
tests:`check(total([1, 2, 3]) == 6, "total([1,2,3]) should be 6")
check(total([]) == 0, "an empty list sums to 0")
check(close(total([0.5, 0.25]), 0.75), "works with floats")`},

{stage:"3 · Collections", id:"lists", title:"Lists, indexing and slicing",
learn:`<p>A <b>list</b> holds items in order: <code>xs = [3, 1, 4]</code>. Indexes start at 0. <code>xs[-1]</code> is the last item.</p>
<p><b>Slicing</b> takes a piece: <code>xs[a:b]</code> runs from index a up to <i>but not including</i> b. <code>xs[:2]</code> is the first two, <code>xs[-2:]</code> the last two, <code>xs[::-1]</code> reversed.</p>
<p>Useful tools: <code>len(xs)</code>, <code>xs.append(v)</code>, <code>max</code>, <code>min</code>, <code>sorted(xs)</code> (new list) and <code>xs.sort()</code> (in place).</p>
<p>Lists are <b>mutable</b>. If two names point at the same list, changing it through one changes it for both. Use <code>xs.copy()</code> or <code>xs[:]</code> when you need a separate copy.</p>`,
example:`xs = [10, 20, 30, 40, 50]
print(xs[0], xs[-1], xs[1:3], xs[::-1])
a = [1, 2]
b = a          # same list
c = a.copy()   # separate list
a.append(3)
print(b, c)`,
why:"Train/test splits, windows of tokens and batches are all slicing.",
task:"Write <code>split(data, frac)</code> that returns <code>(train, test)</code>: the first <code>int(len(data) * frac)</code> items, then the rest.",
pseudo:`cut = int(len(data) * frac)
return data[:cut], data[cut:]`,
hints:["int() rounds down to a whole number.","You can return two values separated by a comma. That makes a tuple."],
starter:`def split(data, frac):
    pass
`,
solution:`def split(data, frac):
    cut = int(len(data) * frac)
    return data[:cut], data[cut:]
`,
tests:`tr, te = split([1, 2, 3, 4, 5], 0.8)
check(tr == [1, 2, 3, 4] and te == [5], "80/20 split of 5 items")
tr, te = split([1, 2, 3], 0.5)
check(tr == [1] and te == [2, 3], "int() rounds 1.5 down to 1")`},

{stage:"3 · Collections", id:"strings", title:"Strings",
learn:`<p>Strings are text. They behave a lot like lists of characters: <code>s[0]</code>, <code>len(s)</code> and slicing all work. Unlike lists, they can't be changed in place. Methods return new strings.</p>
<p>Methods you'll use: <code>.lower()</code>, <code>.strip()</code> (trim spaces), <code>.split()</code> (string to list of words), <code>" ".join(words)</code> (list back to string), <code>.replace(a, b)</code>, <code>.startswith()</code>.</p>
<p>f-strings put values into text: <code>f"loss={loss:.3f}"</code>.</p>`,
example:`s = "  The Cat SAT  "
clean = s.strip().lower()
words = clean.split()
print(clean, words, len(words))
print("-".join(words))
loss = 0.123456
print(f"loss={loss:.3f}")`,
why:"Text preprocessing and tokenization are string work, and they come up in NLP-flavored interviews.",
task:"Write <code>normalize(text)</code> that lowercases the text, removes the characters <code>. , ! ?</code>, and collapses any run of spaces to a single space with no leading or trailing spaces.",
pseudo:`t = text.lower()
for ch in ".,!?":
    t = t.replace(ch, "")
return " ".join(t.split())`,
hints:["Loop over the string \".,!?\" and replace each character with \"\".","split() with no argument splits on any run of whitespace, and join puts single spaces back."],
starter:`def normalize(text):
    pass
`,
solution:`def normalize(text):
    t = text.lower()
    for ch in ".,!?":
        t = t.replace(ch, "")
    return " ".join(t.split())
`,
tests:`check(normalize("Hello, World!") == "hello world", "lowercase and drop punctuation")
check(normalize("  a   b  ") == "a b", "collapse spaces")
check(normalize("") == "", "empty stays empty")`},

{stage:"3 · Collections", id:"dicts", title:"Dictionaries and sets",
learn:`<p>A <b>dict</b> maps keys to values: <code>vocab = {"cat": 0, "dog": 1}</code>. Look up with <code>vocab["cat"]</code>. A missing key raises <code>KeyError</code>, so use <code>vocab.get(k, default)</code> when it might not be there.</p>
<p>Loop with <code>for k, v in d.items():</code>. Check membership with <code>k in d</code>.</p>
<p>A <b>set</b> holds unique items with no order: <code>set([1, 1, 2])</code> is <code>{1, 2}</code>.</p>
<p>The big idea: looking something up in a dict or set takes about the same time however large it is (O(1)). Searching a list means checking every item (O(n)). Interviewers often want you to swap a list search for a dict or set.</p>`,
example:`counts = {}
for w in "a b a c a".split():
    counts[w] = counts.get(w, 0) + 1
print(counts)
for k, v in counts.items():
    print(k, v)
print(set([3, 1, 3, 2]))`,
why:"Vocabularies, label maps, counting and caching are all dicts.",
task:"Write <code>label_map(labels)</code> that returns a dict from each distinct label to an id, with ids assigned in <b>sorted</b> label order starting at 0.",
pseudo:`unique = sorted(set(labels))
return {label: i for i, label in enumerate(unique)}`,
hints:["set() removes duplicates, sorted() orders them and gives back a list.","enumerate gives you (index, item) pairs."],
starter:`def label_map(labels):
    pass
`,
solution:`def label_map(labels):
    unique = sorted(set(labels))
    return {lab: i for i, lab in enumerate(unique)}
`,
tests:`check(label_map(["dog", "cat", "dog", "bird"]) == {"bird": 0, "cat": 1, "dog": 2}, "sorted labels get ids 0,1,2")
check(label_map([]) == {}, "no labels gives an empty dict")`},

{stage:"3 · Collections", id:"comprehensions", title:"List comprehensions",
learn:`<p>A <b>comprehension</b> builds a list in one line: <code>[x * 2 for x in xs]</code> is the same as a loop that appends <code>x * 2</code> each time.</p>
<p>Add a filter at the end: <code>[x for x in xs if x &gt; 0]</code>.</p>
<p>They nest, which is how you build a matrix: <code>[[0] * cols for _ in range(rows)]</code>. Don't write <code>[[0] * cols] * rows</code>. That repeats the same inner list, so changing one row changes all of them.</p>
<p>Dict comprehensions work the same way: <code>{w: i for i, w in enumerate(words)}</code>.</p>`,
example:`xs = [-2, -1, 0, 1, 2]
print([x * x for x in xs])
print([x for x in xs if x > 0])
grid = [[0] * 3 for _ in range(2)]
grid[0][0] = 9
print(grid)
bad = [[0] * 3] * 2
bad[0][0] = 9
print(bad, "<- both rows changed")`,
why:"Interviewers read comprehensions as a sign of fluency, and ML code is full of them.",
task:"Write <code>relu_all(xs)</code> that returns a new list with every negative value replaced by 0. Use one comprehension.",
pseudo:`return [x if x > 0 else 0 for x in xs]`,
hints:["An if/else in the value part goes before the for: [a if cond else b for x in xs].","max(0, x) also works."],
starter:`def relu_all(xs):
    pass
`,
solution:`def relu_all(xs):
    return [max(0, x) for x in xs]
`,
tests:`check(relu_all([-1, 2, -3, 4]) == [0, 2, 0, 4], "negatives become 0")
check(relu_all([]) == [], "empty in, empty out")`},

{stage:"4 · Writing solid code", id:"errors", title:"Errors and exceptions",
learn:`<p>When something goes wrong, Python <b>raises</b> an exception: <code>ZeroDivisionError</code>, <code>KeyError</code>, <code>IndexError</code>, <code>TypeError</code>, <code>ValueError</code>. The last line of the error message tells you which one and roughly why. Read it before anything else.</p>
<p>Catch errors you expect with <code>try</code> / <code>except</code>. Raise your own with <code>raise ValueError("message")</code> when input is invalid. Interviewers like seeing you check shapes and raise a clear error.</p>
<p><code>assert cond, "message"</code> is a quick sanity check that raises if cond is false.</p>`,
example:`def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0

print(safe_div(1, 2), safe_div(1, 0))
try:
    [1, 2][5]
except IndexError as e:
    print("caught:", e)`,
why:"Handling bad input and shape mismatches is how you show you write real code, not just happy-path code.",
task:"Write <code>check_shape(M, rows, cols)</code> that raises <code>ValueError</code> if matrix M (a list of rows) isn't exactly rows × cols, and returns <code>True</code> if it is.",
pseudo:`if len(M) != rows: raise ValueError
for each row in M:
    if len(row) != cols: raise ValueError
return True`,
hints:["Check the number of rows first, then each row's length.","raise ValueError(\"expected ...\") with a helpful message."],
starter:`def check_shape(M, rows, cols):
    pass
`,
solution:`def check_shape(M, rows, cols):
    if len(M) != rows:
        raise ValueError("expected %d rows, got %d" % (rows, len(M)))
    for r in M:
        if len(r) != cols:
            raise ValueError("expected %d cols" % cols)
    return True
`,
tests:`check(check_shape([[1, 2], [3, 4]], 2, 2) == True, "a 2x2 matrix passes")
bad = 0
for M, r, c in [([[1, 2]], 2, 2), ([[1, 2], [3]], 2, 2)]:
    try:
        check_shape(M, r, c)
    except ValueError:
        bad += 1
check(bad == 2, "wrong row count and ragged rows both raise ValueError")`},

{stage:"4 · Writing solid code", id:"classes", title:"Classes and objects",
learn:`<p>A <b>class</b> bundles data and the functions that work on it. It's a blueprint. Each <b>object</b> made from it has its own data.</p>
<p><code>__init__</code> runs when you create an object. It stores values on <code>self</code>. Every method takes <code>self</code> as its first parameter, and you read stored values as <code>self.name</code>.</p>
<p>This is the exact shape of PyTorch modules: store parameters in <code>__init__</code>, compute in <code>forward</code>. When you freeze, write this skeleton first:</p>
<pre>class Thing:
    def __init__(self, ...):
        self.x = ...
    def method(self, ...):
        return ...</pre>`,
example:`class Counter:
    def __init__(self):
        self.n = 0
    def add(self, k=1):
        self.n += k
        return self.n

a = Counter()
b = Counter()
a.add(); a.add(5)
print(a.n, b.n)  # separate objects, separate data`,
why:"The question you froze on was 'implement this in a class'. This is the skeleton.",
task:"Write a class <code>RunningMean</code> with <code>update(x)</code>, which records a new value, and <code>value()</code>, which returns the mean of everything seen so far (0.0 if nothing yet). Store only a total and a count, not the whole list.",
pseudo:`__init__: self.total = 0; self.count = 0
update(x): total += x; count += 1
value(): return total / count if count else 0.0`,
hints:["Two attributes in __init__: self.total and self.count.","Guard the division when count is 0.","Training loops track the average loss exactly like this."],
starter:`class RunningMean:
    def __init__(self):
        pass

    def update(self, x):
        pass

    def value(self):
        pass
`,
solution:`class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, x):
        self.total += x
        self.count += 1

    def value(self):
        return self.total / self.count if self.count else 0.0
`,
tests:`m = RunningMean()
check(m.value() == 0.0, "empty mean is 0.0")
for x in [1, 2, 3, 4]:
    m.update(x)
check(close(m.value(), 2.5), "mean of 1..4 is 2.5")
m2 = RunningMean()
check(m2.value() == 0.0, "a second object has its own state")`},

{stage:"4 · Writing solid code", id:"recursion", title:"Recursion",
learn:`<p>A <b>recursive</b> function calls itself on a smaller version of the problem. Every recursive function needs two parts:</p>
<ul><li>a <b>base case</b> that answers directly and stops the recursion</li><li>a <b>recursive case</b> that shrinks the problem and calls itself</li></ul>
<p>Without a base case, it never stops and you get a <code>RecursionError</code>. Trees, nested lists and divide-and-conquer algorithms are natural fits.</p>`,
example:`def factorial(n):
    if n <= 1:          # base case
        return 1
    return n * factorial(n - 1)  # smaller problem

print(factorial(5))`,
why:"Nested data (JSON configs, model parameter trees) and tree questions both use recursion.",
task:"Write <code>flatten(xs)</code> that turns an arbitrarily nested list into a flat list, keeping order. <code>[1, [2, [3]], 4]</code> becomes <code>[1, 2, 3, 4]</code>.",
pseudo:`out = []
for item in xs:
    if item is a list: out.extend(flatten(item))
    else: out.append(item)
return out`,
hints:["isinstance(item, list) tells you whether something is a list.","extend adds every element of a list. append adds one item."],
starter:`def flatten(xs):
    pass
`,
solution:`def flatten(xs):
    out = []
    for item in xs:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
`,
tests:`check(flatten([1, [2, [3]], 4]) == [1, 2, 3, 4], "nested list flattens in order")
check(flatten([[[]]]) == [], "nested empties give []")
check(flatten([1, 2]) == [1, 2], "an already-flat list is unchanged")`},

{stage:"4 · Writing solid code", id:"bigo", title:"Big-O and choosing data structures",
learn:`<p><b>Big-O</b> describes how running time grows with input size n. You'll be asked for it after almost every answer.</p>
<ul><li><b>O(1)</b>: constant. Dict or set lookup, list index.</li><li><b>O(log n)</b>: halve each step. Binary search.</li><li><b>O(n)</b>: one pass. A loop over the list.</li><li><b>O(n log n)</b>: sorting.</li><li><b>O(n²)</b>: a loop inside a loop. Comparing all pairs. Attention over a sequence of length n.</li></ul>
<p>Rule of thumb: count nested loops over the input. To speed up an O(n²) "have I seen this?" search, remember what you've seen in a set, and it becomes O(n).</p>`,
example:`def has_dup_slow(xs):          # O(n^2)
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] == xs[j]:
                return True
    return False

print(has_dup_slow([3, 1, 4, 1]))`,
why:"\"What's the complexity?\" follows nearly every coding answer, including attention (O(n²) in sequence length).",
task:"Write <code>has_duplicates(xs)</code> in <b>O(n)</b> using a set.",
pseudo:`seen = empty set
for x in xs:
    if x in seen: return True
    seen.add(x)
return False`,
hints:["Checking x in a set is O(1).","One loop, so O(n) overall."],
starter:`def has_duplicates(xs):
    pass
`,
solution:`def has_duplicates(xs):
    seen = set()
    for x in xs:
        if x in seen:
            return True
        seen.add(x)
    return False
`,
tests:`check(has_duplicates([3, 1, 4, 1]) == True, "finds the repeated 1")
check(has_duplicates([1, 2, 3]) == False, "no duplicates")
big = list(range(200000))
check(has_duplicates(big) == False, "fast on 200,000 items")`},

{stage:"4 · Writing solid code", id:"binsearch", title:"Binary search",
learn:`<p>Binary search finds a value in a <b>sorted</b> list in O(log n) by repeatedly checking the middle and throwing away the half that can't contain the answer.</p>
<p>Keep two pointers, <code>lo</code> and <code>hi</code>. Loop while <code>lo &lt;= hi</code>. The classic bugs are off-by-one errors in the pointer updates, so set <code>lo = mid + 1</code> or <code>hi = mid - 1</code>, never just <code>mid</code>.</p>`,
example:`xs = [1, 3, 5, 7, 9, 11]
lo, hi = 0, len(xs) - 1
while lo <= hi:
    mid = (lo + hi) // 2
    print("checking index", mid, "value", xs[mid])
    if xs[mid] == 9: print("found"); break
    if xs[mid] < 9: lo = mid + 1
    else: hi = mid - 1`,
why:"A classic warm-up question. It also shows up in threshold tuning and searching sorted scores.",
task:"Write <code>bsearch(xs, target)</code> that returns the index of target in sorted list xs, or -1 if it isn't there.",
pseudo:`lo, hi = 0, len(xs) - 1
while lo <= hi:
    mid = (lo + hi) // 2
    if xs[mid] == target: return mid
    if xs[mid] < target: lo = mid + 1
    else: hi = mid - 1
return -1`,
hints:["Use // for the middle so it's an integer.","If the middle is too small, the answer must be to the right."],
starter:`def bsearch(xs, target):
    pass
`,
solution:`def bsearch(xs, target):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
`,
tests:`xs = [1, 3, 5, 7, 9, 11]
check(bsearch(xs, 9) == 4, "9 is at index 4")
check(bsearch(xs, 1) == 0 and bsearch(xs, 11) == 5, "finds both ends")
check(bsearch(xs, 4) == -1, "missing value gives -1")
check(bsearch([], 1) == -1, "empty list gives -1")`},
];
