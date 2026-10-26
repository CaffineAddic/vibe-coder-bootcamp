# Contributing

Pull requests for new lessons, exercises, concept cards, and fixes are welcome.

## Adding an exercise

Add an object to `course/foundations.js` (Part 1) or `course/ml.js` (Part 2):

```js
{stage:"8 · Core ML functions", id:"unique_id", title:"Short title",
 learn:`<p>Optional. Part 1 lessons teach the idea here.</p>`,
 example:`print("optional runnable example, shown with learn")`,
 why:"One sentence on why interviewers care.",
 task:"What to write. Use <code>name(args)</code> for the function.",
 pseudo:`plain-language steps`,
 hints:["gentle", "more specific", "nearly the answer"],
 starter:`def name(args):\n    pass\n`,
 solution:`def name(args):\n    ...\n`,
 tests:`check(name(1) == 2, "message shown to the learner")`}
```

Guidelines:

- Tests use `check(condition, message)` and `close(a, b)` for floats. Write messages that say what should happen, so a failing test teaches something.
- Include at least one edge case (empty input, ties, zero, a big value).
- Stick to the standard library parts Skulpt supports (`math` and builtins). No NumPy.
- Keep `id` unique. Learners' saved progress is keyed by it.
- Run `npm test` before opening a PR. Every solution must pass its own tests.

## Adding a concept card

Add `{topic, q, a}` to `course/concepts.js`. Answers should be two to four sentences that someone could say out loud in an interview.
