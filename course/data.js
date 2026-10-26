// Part 4: pandas and data processing for ML. The "here's a messy table, get it ready to train on" questions.
const DATA_LESSONS = [
{stage:"14 · pandas basics", id:"pd_frames", title:"DataFrames: select, filter, sort", packages:["numpy","pandas"],
learn:`<p>A <b>DataFrame</b> is a table: named columns, each holding one type, plus a row index. A single column is a <b>Series</b>.</p>
<ul><li>Read data: <code>pd.read_csv(path)</code>. In this course we read from a string with <code>io.StringIO</code>.</li><li>Look at it: <code>df.head()</code>, <code>df.shape</code>, <code>df.dtypes</code>, <code>df.describe()</code>.</li><li>Select columns: <code>df["price"]</code> gives a Series, and <code>df[["price", "qty"]]</code> gives a DataFrame.</li><li>Filter rows with a boolean mask: <code>df[df["price"] &gt; 10]</code>. Combine conditions with <code>&amp;</code> and <code>|</code>, each in parentheses.</li><li>Sort: <code>df.sort_values("price", ascending=False)</code>.</li><li>New column: <code>df["total"] = df["price"] * df["qty"]</code>.</li></ul>
<p><code>.loc[rows, cols]</code> selects by label and <code>.iloc[rows, cols]</code> by position.</p>`,
example:`import io
import pandas as pd
csv = """item,price,qty
apple,1.2,10
pear,3.5,2
melon,4.0,5"""
df = pd.read_csv(io.StringIO(csv))
print(df.shape)
print(df.dtypes)
df["total"] = df["price"] * df["qty"]
print(df[df["total"] > 8].sort_values("total", ascending=False))`,
why:"Almost every data-processing question starts here. Speed with these basics leaves time for the real problem.",
task:"Write <code>top_orders(df, n)</code>. Add a column <code>revenue = price × qty</code>, keep only rows with <code>qty &gt; 0</code>, and return the <code>n</code> rows with the highest revenue, sorted highest first. Reset the index so it runs 0..n-1.",
pseudo:`df = df.copy()
df["revenue"] = df["price"] * df["qty"]
df = df[df["qty"] > 0]
return df.sort_values("revenue", ascending=False).head(n).reset_index(drop=True)`,
hints:["Copy first so you don't change the caller's DataFrame.","reset_index(drop=True) throws away the old index instead of adding it as a column."],
starter:`import pandas as pd

def top_orders(df, n):
    pass
`,
solution:`import pandas as pd

def top_orders(df, n):
    df = df.copy()
    df["revenue"] = df["price"] * df["qty"]
    df = df[df["qty"] > 0]
    return df.sort_values("revenue", ascending=False).head(n).reset_index(drop=True)
`,
tests:`import pandas as pd
df = pd.DataFrame({"item": ["a", "b", "c", "d"], "price": [2.0, 10.0, 1.0, 50.0], "qty": [5, 3, 100, 0]})
r = top_orders(df, 2)
check(list(r["item"]) == ["c", "b"], "top 2 by revenue are c (100) then b (30); d has qty 0")
check(list(r.index) == [0, 1], "index is reset to 0..n-1")
check("revenue" not in df.columns, "the original DataFrame is not modified")`},

{stage:"14 · pandas basics", id:"pd_missing", title:"Missing values", packages:["numpy","pandas"],
learn:`<p>Real data has holes. pandas marks them as <code>NaN</code>.</p>
<ul><li>Find them: <code>df.isna().sum()</code> counts the missing values per column.</li><li>Drop them: <code>df.dropna(subset=["label"])</code> drops rows missing a label.</li><li>Fill them: <code>df["age"].fillna(df["age"].median())</code>. The median is robust to outliers. Use the most frequent value for categories.</li></ul>
<p>Never drop rows or fill values without thinking. Rows without a label can't be used for training, so drop those. For features, filling is usually better than throwing data away. Adding a "was missing" flag column can also help the model.</p>
<p>In a real pipeline, compute fill values on the <b>training set only</b> and reuse them on the test set. Lesson 6 covers why.</p>`,
example:`import numpy as np
import pandas as pd
df = pd.DataFrame({"age": [25, np.nan, 40, 31], "city": ["NY", "SF", None, "NY"], "label": [1, 0, np.nan, 1]})
print(df.isna().sum())
print(df["city"].mode()[0], df["age"].median())`,
why:"'This column has missing values, what do you do?' is close to guaranteed in a data question.",
task:"Write <code>clean(df)</code>. Drop rows where <code>label</code> is missing. Then fill missing values in every <b>numeric</b> column (other than label) with that column's median, and in every <b>text</b> column with its most frequent value. Return the new DataFrame.",
pseudo:`df = df.dropna(subset=["label"]).copy()
for col in df.columns:
    if col == "label": continue
    if numeric: fill with median
    else: fill with mode()[0]
return df`,
hints:["pd.api.types.is_numeric_dtype(df[col]) tells you whether a column is numeric.","Drop the unlabeled rows first, then compute the medians from the rows you keep."],
starter:`import pandas as pd

def clean(df):
    pass
`,
solution:`import pandas as pd

def clean(df):
    df = df.dropna(subset=["label"]).copy()
    for col in df.columns:
        if col == "label":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
`,
tests:`import numpy as np
import pandas as pd
df = pd.DataFrame({"age": [20.0, np.nan, 40.0, 30.0, np.nan], "city": ["NY", "SF", None, "NY", "SF"], "label": [1, 0, np.nan, 1, 1]})
r = clean(df)
check(len(r) == 4, "the row with a missing label is dropped")
check(r.isna().sum().sum() == 0, "no missing values remain")
check(close(r["age"].iloc[1], 25.0), "age is filled with the median of the labeled rows (25)")
check(r["city"].iloc[2] == "NY", "city keeps its value where present")`},

{stage:"14 · pandas basics", id:"pd_groupby", title:"groupby and aggregation", packages:["numpy","pandas"],
learn:`<p><code>groupby</code> splits the rows into groups, computes something per group, and combines the results. It's the pandas version of SQL's <code>GROUP BY</code>.</p>
<pre>df.groupby("user")["amount"].mean()
df.groupby("user").agg(n=("amount", "size"), avg=("amount", "mean"))</pre>
<p>The second form, <b>named aggregation</b>, lets you choose the output column names. Common functions: <code>"size"</code>, <code>"count"</code> (ignores NaN), <code>"sum"</code>, <code>"mean"</code>, <code>"max"</code>, <code>"nunique"</code>.</p>
<p>The grouped column becomes the index. Use <code>.reset_index()</code> to turn it back into a normal column.</p>
<p>To attach a group statistic to every original row, use <code>transform</code>: <code>df["user_avg"] = df.groupby("user")["amount"].transform("mean")</code>. This is a very common feature-engineering move.</p>`,
example:`import pandas as pd
df = pd.DataFrame({"user": ["a", "b", "a", "a", "b"], "amount": [10, 5, 30, 20, 15]})
print(df.groupby("user").agg(n=("amount", "size"), total=("amount", "sum")).reset_index())
df["user_avg"] = df.groupby("user")["amount"].transform("mean")
print(df)`,
why:"Per-user and per-item aggregate features are the bread and butter of tabular ML, and of the questions about it.",
task:"Write <code>user_features(df)</code> that returns one row per <code>user</code>, sorted by user, with columns <code>user</code>, <code>n_orders</code>, <code>total</code>, <code>avg</code> and <code>max</code> computed from <code>amount</code>.",
pseudo:`return (df.groupby("user")
          .agg(n_orders=("amount", "size"), total=("amount", "sum"),
               avg=("amount", "mean"), max=("amount", "max"))
          .reset_index()
          .sort_values("user"))`,
hints:["Named aggregation: new_name=(\"column\", \"function\").","reset_index() turns the user index back into a column."],
starter:`import pandas as pd

def user_features(df):
    pass
`,
solution:`import pandas as pd

def user_features(df):
    out = df.groupby("user").agg(
        n_orders=("amount", "size"),
        total=("amount", "sum"),
        avg=("amount", "mean"),
        max=("amount", "max"),
    ).reset_index()
    return out.sort_values("user").reset_index(drop=True)
`,
tests:`import pandas as pd
df = pd.DataFrame({"user": ["b", "a", "a", "b", "a"], "amount": [5, 10, 30, 15, 20]})
r = user_features(df)
check(list(r.columns) == ["user", "n_orders", "total", "avg", "max"], "columns are user, n_orders, total, avg, max")
check(list(r["user"]) == ["a", "b"], "one row per user, sorted")
check(list(r["n_orders"]) == [3, 2] and list(r["total"]) == [60, 20], "counts and totals are right")
check(close(r["avg"].iloc[0], 20.0) and r["max"].iloc[1] == 15, "avg and max are right")`},

{stage:"14 · pandas basics", id:"pd_merge", title:"Joining tables", packages:["numpy","pandas"],
learn:`<p><code>pd.merge(left, right, on="key", how=...)</code> combines tables like a SQL join.</p>
<ul><li><code>how="inner"</code> keeps keys present in both tables (the default).</li><li><code>how="left"</code> keeps every row of the left table and fills NaN where the right table has no match. Use it to attach features to a table of examples without losing any examples.</li><li><code>how="outer"</code> keeps everything from both.</li></ul>
<p>Watch for <b>duplicate keys</b>. If a key appears 3 times on the right, each matching left row turns into 3 rows. Check <code>len()</code> before and after a merge. That's a habit interviewers notice.</p>`,
example:`import pandas as pd
users = pd.DataFrame({"user": ["a", "b", "c"], "country": ["US", "IN", "US"]})
orders = pd.DataFrame({"user": ["a", "a", "b"], "amount": [10, 20, 5]})
print(pd.merge(users, orders, on="user", how="inner"))
print(pd.merge(users, orders, on="user", how="left"))`,
why:"Building a training table usually means joining events, users and labels. Losing or duplicating rows is a classic bug.",
task:"Write <code>build_table(users, orders)</code>. Compute each user's total <code>amount</code> from orders, left-join it onto <code>users</code> as a column <code>total_spent</code>, fill users with no orders with 0, and keep exactly one row per user in the original order.",
pseudo:`totals = orders.groupby("user")["amount"].sum().reset_index(name="total_spent")
out = users.merge(totals, on="user", how="left")
out["total_spent"] = out["total_spent"].fillna(0)
return out`,
hints:["Aggregate the orders first so there's one row per user before merging. That avoids duplicates.","reset_index(name=...) names the column when turning a Series into a DataFrame."],
starter:`import pandas as pd

def build_table(users, orders):
    pass
`,
solution:`import pandas as pd

def build_table(users, orders):
    totals = orders.groupby("user")["amount"].sum().reset_index(name="total_spent")
    out = users.merge(totals, on="user", how="left")
    out["total_spent"] = out["total_spent"].fillna(0)
    return out
`,
tests:`import pandas as pd
users = pd.DataFrame({"user": ["c", "a", "b"], "age": [30, 20, 40]})
orders = pd.DataFrame({"user": ["a", "a", "b", "z"], "amount": [10, 20, 5, 99]})
r = build_table(users, orders)
check(len(r) == 3, "exactly one row per user (no duplicates, user z ignored)")
check(list(r["user"]) == ["c", "a", "b"], "keeps the original user order")
check(list(r["total_spent"]) == [0, 30, 5], "totals are 0, 30, 5")`},

{stage:"15 · Data prep for ML", id:"pd_encode", title:"Encoding categorical features", packages:["numpy","pandas"],
learn:`<p>Models need numbers, so text categories have to be encoded.</p>
<ul><li><b>One-hot</b>: one 0/1 column per category. <code>pd.get_dummies(df, columns=["city"])</code>. Good for a small number of categories with no order.</li><li><b>Ordinal</b>: map categories that have a natural order to integers, like <code>{"low": 0, "mid": 1, "high": 2}</code> with <code>.map()</code>.</li><li><b>Frequency or target encoding</b>: replace a high-cardinality category (like a zip code) with a statistic. Target encoding must be fit on training data only, or it leaks the label.</li></ul>
<p>A test set can contain categories that the training set never had. Your encoder needs a plan for those. A common one is mapping them to an "unknown" bucket, or to all zeros for one-hot.</p>`,
example:`import pandas as pd
df = pd.DataFrame({"city": ["NY", "SF", "NY"], "size": ["low", "high", "mid"]})
print(pd.get_dummies(df, columns=["city"], dtype=int))
print(df["size"].map({"low": 0, "mid": 1, "high": 2}))`,
why:"You'll be asked how to handle categorical features, high-cardinality ones, and categories that only appear at test time.",
task:"Write <code>one_hot_fit(train_col)</code> that returns the sorted list of categories seen in training, and <code>one_hot_apply(col, categories)</code> that returns a DataFrame with one int 0/1 column per training category, named <code>f\"{col.name}_{cat}\"</code>. Categories not seen in training get all zeros.",
pseudo:`fit: return sorted(train_col.dropna().unique())
apply:
    out = DataFrame(index=col.index)
    for cat in categories:
        out[f"{col.name}_{cat}"] = (col == cat).astype(int)
    return out`,
hints:["(col == cat) gives a boolean Series. .astype(int) turns it into 0/1.","Looping over the training categories means unseen categories automatically get zeros everywhere."],
starter:`import pandas as pd

def one_hot_fit(train_col):
    pass

def one_hot_apply(col, categories):
    pass
`,
solution:`import pandas as pd

def one_hot_fit(train_col):
    return sorted(train_col.dropna().unique())

def one_hot_apply(col, categories):
    out = pd.DataFrame(index=col.index)
    for cat in categories:
        out[f"{col.name}_{cat}"] = (col == cat).astype(int)
    return out
`,
tests:`import pandas as pd
train = pd.Series(["SF", "NY", "SF"], name="city")
cats = one_hot_fit(train)
check(list(cats) == ["NY", "SF"], "categories are sorted: NY, SF")
test = pd.Series(["NY", "LA"], name="city")
r = one_hot_apply(test, cats)
check(list(r.columns) == ["city_NY", "city_SF"], "columns come from training categories")
check(r.values.tolist() == [[1, 0], [0, 0]], "NY -> [1,0]; unseen LA -> all zeros")`},

{stage:"15 · Data prep for ML", id:"pd_leak", title:"Split, then scale: avoiding leakage", packages:["numpy","pandas"],
learn:`<p><b>Data leakage</b> means information from the test set, or from the future, sneaks into training. Scores look great, then fall apart in production. Interviewers ask about it all the time.</p>
<p>The rule: <b>split first</b>. Then compute every statistic (means, stds, medians for filling, category lists, target encodings) on the <b>training set only</b>, and apply those same numbers to validation and test.</p>
<p>Wrong: <code>X = (X - X.mean()) / X.std()</code>, then split. The test rows influenced the mean.<br>Right: split, then <code>mu = train.mean()</code>, and use <code>mu</code> for both train and test.</p>
<p>Shuffle before a random split, with a fixed seed so results are reproducible: <code>df.sample(frac=1, random_state=42)</code>.</p>`,
example:`import pandas as pd
df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
train, test = df.iloc[:3], df.iloc[3:]
print("leaky mean:", df["x"].mean(), " correct (train-only) mean:", train["x"].mean())`,
why:"\"How do you avoid leakage?\" is one of the most common ML interview questions, and this is the code behind the answer.",
task:"Write <code>split_scale(df, features, test_frac=0.25, seed=0)</code>. Shuffle with <code>df.sample(frac=1, random_state=seed)</code>, put the last <code>int(len(df) * test_frac)</code> rows in test and the rest in train, then standardize <code>features</code> using the <b>train</b> mean and std (pandas' default std). Return <code>(train, test)</code>.",
pseudo:`df = df.sample(frac=1, random_state=seed)
n_test = int(len(df) * test_frac)
train, test = df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()
mu, sd = train[features].mean(), train[features].std()
train[features] = (train[features] - mu) / sd
test[features]  = (test[features]  - mu) / sd`,
hints:["Compute mu and sd once, from train only, and reuse them for test.","After scaling, train has mean ≈ 0, but test usually won't. That's expected and correct."],
starter:`import pandas as pd

def split_scale(df, features, test_frac=0.25, seed=0):
    pass
`,
solution:`import pandas as pd

def split_scale(df, features, test_frac=0.25, seed=0):
    df = df.sample(frac=1, random_state=seed)
    n_test = int(len(df) * test_frac)
    train, test = df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()
    mu, sd = train[features].mean(), train[features].std()
    train[features] = (train[features] - mu) / sd
    test[features] = (test[features] - mu) / sd
    return train, test
`,
tests:`import numpy as np
import pandas as pd
df = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float) * 3, "label": [0, 1] * 10})
tr, te = split_scale(df, ["x", "y"])
check(len(tr) == 15 and len(te) == 5, "15 train rows, 5 test rows")
check(np.allclose(tr[["x", "y"]].mean(), 0, atol=1e-9), "train features have mean 0")
raw = df.sample(frac=1, random_state=0)
mu, sd = raw.iloc[:15]["x"].mean(), raw.iloc[:15]["x"].std()
check(np.allclose(te["x"].values, (raw.iloc[15:]["x"].values - mu) / sd), "test is scaled with train statistics, not its own")
check(set(tr.index).isdisjoint(te.index), "no row is in both sets")`},

{stage:"15 · Data prep for ML", id:"pd_group_split", title:"Group and time splits", packages:["numpy","pandas"],
learn:`<p>A random row split is wrong whenever rows are related:</p>
<ul><li><b>Same entity in many rows</b> (a user, a patient, a document). If the same user is in train and test, the model can memorize that user. Split by <b>group</b> so every user lands entirely on one side.</li><li><b>Time series or anything forecast-like.</b> Train on the past and test on the future. A random split lets the model peek ahead.</li></ul>
<p>Saying "I'd split by user to avoid leakage" or "I'd use a time-based split" when you see the data is a strong interview signal.</p>`,
example:`import pandas as pd
df = pd.DataFrame({"user": ["a", "a", "b", "c", "c", "c"], "x": range(6)})
print(df["user"].unique())`,
why:"Follow-up questions on leakage almost always come back to group and time splits.",
task:"Write <code>group_split(df, group_col, test_frac, seed=0)</code>. Take the sorted unique groups, shuffle them with <code>np.random.default_rng(seed).permutation</code>, put the first <code>max(1, int(n_groups * test_frac))</code> groups in test and the rest in train. Return <code>(train_df, test_df)</code>.",
pseudo:`groups = np.array(sorted(df[group_col].unique()))
groups = rng.permutation(groups)
n_test = max(1, int(len(groups) * test_frac))
test_groups = set(groups[:n_test])
mask = df[group_col].isin(test_groups)
return df[~mask], df[mask]`,
hints:["df[col].isin(collection) gives a boolean mask.","~mask flips the mask."],
starter:`import numpy as np
import pandas as pd

def group_split(df, group_col, test_frac, seed=0):
    pass
`,
solution:`import numpy as np
import pandas as pd

def group_split(df, group_col, test_frac, seed=0):
    groups = np.array(sorted(df[group_col].unique()))
    groups = np.random.default_rng(seed).permutation(groups)
    n_test = max(1, int(len(groups) * test_frac))
    mask = df[group_col].isin(set(groups[:n_test]))
    return df[~mask], df[mask]
`,
tests:`import pandas as pd
df = pd.DataFrame({"user": list("aabbbccddddeeff"), "x": range(15)})
tr, te = group_split(df, "user", 0.34)
check(len(tr) + len(te) == len(df), "every row ends up in exactly one set")
check(set(tr["user"]).isdisjoint(set(te["user"])), "no user appears in both train and test")
check(te["user"].nunique() == 2, "34% of 6 users rounds down to 2 test users")`},

{stage:"15 · Data prep for ML", id:"pd_time", title:"Datetime features and lags", packages:["numpy","pandas"],
learn:`<p>Timestamps hold a lot of signal once you unpack them.</p>
<ul><li>Parse: <code>pd.to_datetime(df["ts"])</code>.</li><li>Extract parts with <code>.dt</code>: <code>.dt.hour</code>, <code>.dt.dayofweek</code> (Monday = 0), <code>.dt.month</code>.</li><li><b>Lag features</b> use the previous value: <code>df.groupby("user")["amount"].shift(1)</code>. Sort by time first.</li><li><b>Rolling features</b>: <code>.rolling(3).mean()</code>.</li></ul>
<p>Leakage check for time features: every feature for a row may only use information available <b>before</b> that row's time. <code>shift(1)</code> is safe. A group mean over all rows, including future ones, is not.</p>`,
example:`import pandas as pd
df = pd.DataFrame({"ts": ["2026-01-05 09:30", "2026-01-06 18:00"], "amount": [10, 20]})
df["ts"] = pd.to_datetime(df["ts"])
print(df["ts"].dt.dayofweek, df["ts"].dt.hour)`,
why:"Event logs and transactions are the most common raw data in ML take-home and live data questions.",
task:"Write <code>add_features(df)</code> for columns <code>user</code>, <code>ts</code> (string) and <code>amount</code>. Parse ts, sort by user then ts, and add <code>hour</code>, <code>is_weekend</code> (1 if Saturday or Sunday, else 0), and <code>prev_amount</code>: the user's previous amount, or 0 for their first event. Reset the index.",
pseudo:`df = df.copy(); df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values(["user", "ts"]).reset_index(drop=True)
df["hour"] = df["ts"].dt.hour
df["is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(int)
df["prev_amount"] = df.groupby("user")["amount"].shift(1).fillna(0)`,
hints:["dayofweek is 5 for Saturday and 6 for Sunday.","shift(1) inside groupby only looks at earlier rows of the same user."],
starter:`import pandas as pd

def add_features(df):
    pass
`,
solution:`import pandas as pd

def add_features(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["user", "ts"]).reset_index(drop=True)
    df["hour"] = df["ts"].dt.hour
    df["is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(int)
    df["prev_amount"] = df.groupby("user")["amount"].shift(1).fillna(0)
    return df
`,
tests:`import pandas as pd
df = pd.DataFrame({"user": ["b", "a", "a", "b"],
                   "ts": ["2026-01-03 10:00", "2026-01-05 08:00", "2026-01-04 23:00", "2026-01-02 12:00"],
                   "amount": [7, 30, 20, 5]})
r = add_features(df)
check(list(r["user"]) == ["a", "a", "b", "b"], "sorted by user")
check(list(r["amount"]) == [20, 30, 5, 7], "and by time within each user")
check(list(r["hour"]) == [23, 8, 12, 10], "hour extracted")
check(list(r["is_weekend"]) == [1, 0, 0, 1], "2026-01-04 is a Sunday and 2026-01-03 a Saturday")
check(list(r["prev_amount"]) == [0, 20, 0, 5], "prev_amount uses only that user's earlier event")`},

{stage:"15 · Data prep for ML", id:"pd_pipeline", title:"End to end: raw table to training arrays", packages:["numpy","pandas"],
learn:`<p>This is the shape of a live data-processing interview: messy CSV in, model-ready arrays out. Talk through the steps in this order:</p>
<ol><li>Look at the data: shape, dtypes, missing values, label balance.</li><li>Drop rows that can't be used (no label).</li><li><b>Split</b> before computing any statistic.</li><li>Fit the preprocessing on train: fill values, category lists, scaling stats.</li><li>Apply it to train and test.</li><li>Return NumPy arrays: <code>X</code> as float (N, D), <code>y</code> as int (N,).</li></ol>
<p>Keep the fitted statistics in one place (a dict or a class) so the exact same transform can run at prediction time.</p>`,
example:`import io
import pandas as pd
raw = """age,plan,spend,churned
34,pro,120.5,0
,free,0,1
51,pro,,0
29,free,10,
45,team,300,0"""
df = pd.read_csv(io.StringIO(raw))
print(df.isna().sum())
print(df["churned"].value_counts(dropna=False))`,
why:"This pulls the whole part together and mirrors real take-home tasks.",
task:"Write a class <code>Prep</code>. <code>fit(df)</code> learns from a training DataFrame with columns <code>age</code>, <code>spend</code> (numeric, may be missing) and <code>plan</code> (text): the median of each numeric column, the mean and std of each after filling, and the sorted plan categories. <code>transform(df)</code> returns a float NumPy array whose columns are the scaled age, the scaled spend, then one 0/1 column per training plan. Return <code>self</code> from fit.",
pseudo:`fit:
    self.med = {c: df[c].median() for c in ["age","spend"]}
    filled = df[num].fillna(self.med)
    self.mu, self.sd = filled.mean(), filled.std()
    self.plans = sorted(df["plan"].dropna().unique())
transform:
    num = (df[num].fillna(self.med) - self.mu) / self.sd
    hot = [(df["plan"] == p).astype(float) for p in self.plans]
    return np.column_stack([num.values] + hot)`,
hints:["fillna accepts a dict of column -> value.","np.column_stack glues 1-D and 2-D arrays together side by side."],
starter:`import numpy as np
import pandas as pd

class Prep:
    NUM = ["age", "spend"]

    def fit(self, df):
        pass

    def transform(self, df):
        pass
`,
solution:`import numpy as np
import pandas as pd

class Prep:
    NUM = ["age", "spend"]

    def fit(self, df):
        self.med = {c: df[c].median() for c in self.NUM}
        filled = df[self.NUM].fillna(self.med)
        self.mu, self.sd = filled.mean(), filled.std()
        self.plans = sorted(df["plan"].dropna().unique())
        return self

    def transform(self, df):
        num = (df[self.NUM].fillna(self.med) - self.mu) / self.sd
        hot = [(df["plan"] == p).astype(float).values for p in self.plans]
        return np.column_stack([num.values] + hot).astype(float)
`,
tests:`import numpy as np
import pandas as pd
train = pd.DataFrame({"age": [20.0, 40.0, np.nan], "spend": [10.0, np.nan, 30.0], "plan": ["pro", "free", "pro"]})
p = Prep().fit(train)
X = p.transform(train)
check(X.shape == (3, 4), "columns: age, spend, plan_free, plan_pro")
check(not np.isnan(X).any(), "no NaNs after transform")
check(np.allclose(X[:, :2].mean(axis=0), 0), "numeric columns are centered on train")
T = p.transform(pd.DataFrame({"age": [30.0], "spend": [20.0], "plan": ["team"]}))
check(np.allclose(T[0], [0, 0, 0, 0]), "median values scale to 0, and an unseen plan gets all-zero one-hot")`},
];
