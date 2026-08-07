func test():
	# The following statements should all be reported as standalone expressions:
	1234
	0.0 + 0.0
	Color(1, 1, 1)
	Vector3.ZERO
	[true, false]
	float(125)
	# The following statements should not produce `STANDALONE_EXPRESSION`:
	var _a = 1
	_a = 2 # Assignment is a local (or global) side effect.
	@warning_ignore("redundant_await")
	await 3 # The `await` operand is usually a coroutine or a signal.
	absi(4) # A call (in general) can have side effects.
	@warning_ignore("return_value_discarded")
	preload("../../utils.notest.gd") # A static initializer may have side effects.
	"""
	Python-like "comment".
	"""
	@warning_ignore("standalone_ternary")
	1 if 2 else 3 # Produces `STANDALONE_TERNARY` instead.
	# Logical `and`/`or` short-circuit can conditionally execute a call, which is a
	# valid effect, so these should not produce `STANDALONE_EXPRESSION`:
	_a and absi(5)
	_a or absi(6)
	absi(7) and _a
	_a and (_a or absi(8))
	# A logical operator with no side effect is still reported:
	_a and _a
	# The effect can be nested in any sub-expression, so these should not produce
	# `STANDALONE_EXPRESSION` either:
	_a and (absi(9) + 2)
	_a and not absi(10)
	_a and (absi(11) if _a else 0)
	@warning_ignore("redundant_await")
	_a and await 12
	[absi(13)]
	{ "key": absi(14) }
	# Without a call, the same expressions are still reported:
	not _a
	[_a]
	{ "key": _a }
