func test():
	# `_` can be used as a separator for numbers in GDScript.
	# It can be placed anywhere in the number, except at the beginning.
	# Currently, GDScript in the `master` branch only allows using one separator
	# per number.
	# Results are assigned to variables to avoid warnings.
	var __ = 1_23
	__ = 123_  # Trailing number separators are OK.
	__ = 12_3
	__ = 123_456
	__ = 0x1234_5678
	__ = 0b1001_0101
