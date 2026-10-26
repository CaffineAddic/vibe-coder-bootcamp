# Contributing

Pull requests for new lessons, exercises, concept cards, and fixes are welcome.

## Adding an exercise

Add an object to the file for its part: `course/foundations.js`, `ml.js`, `numpy.js`, `data.js` or `pytorch.js`:

```js
{stage:"8 · Core ML functions", id:"unique_id", title:"Short title",
 learn:`<p>Optional. Part 1 lessons teach the idea here.</p>`,
 example:`print("optional runnable example, shown with learn")`,
 packages:["numpy", "pandas"],   // optional: libraries the lesson needs ("torch" for Part 5)
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
- Lessons without `packages` must run in Skulpt too (builtins and `math` only).
- PyTorch lessons may only use what `course/torchlite.py` implements. If you add to torchlite, run `npm run build` and `npm run test:real-torch`.
- Keep `id` unique. Learners' saved progress is keyed by it.
- Run `npm test` before opening a PR. Every solution must pass its own tests.

## Adding a concept card

Add `{topic, q, a}` to `course/concepts.js`. Answers should be two to four sentences that someone could say out loud in an interview.
