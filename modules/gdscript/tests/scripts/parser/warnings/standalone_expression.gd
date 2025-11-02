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
