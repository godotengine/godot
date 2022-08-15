Analysis Opportunities:

//===---------------------------------------------------------------------===//

In test/Transforms/LoopStrengthReduce/quadradic-exit-value.ll, the
ScalarEvolution expression for %r is this:

  {1,+,3,+,2}<loop>

Outside the loop, this could be evaluated simply as (%n * %n), however
ScalarEvolution currently evaluates it as

  (-2 + (2 * (trunc i65 (((zext i64 (-2 + %n) to i65) * (zext i64 (-1 + %n) to i65)) /u 2) to i64)) + (3 * %n))

In addition to being much more complicated, it involves i65 arithmetic,
which is very inefficient when expanded into code.

//===---------------------------------------------------------------------===//

In formatValue in test/CodeGen/X86/lsr-delayed-fold.ll,

ScalarEvolution is forming this expression:

((trunc i64 (-1 * %arg5) to i32) + (trunc i64 %arg5 to i32) + (-1 * (trunc i64 undef to i32)))

This could be folded to

(-1 * (trunc i64 undef to i32))

//===---------------------------------------------------------------------===//
