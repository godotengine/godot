# Do not fix code style here!

func test():
	# The following lines are equivalent:
	var _integer: int = 1
	var _integer2 : int = 1
	var _inferred := 1
	var _inferred2 : = 1

	# Type inference is automatic for constants.
	const _INTEGER = 1
	const _INTEGER_REDUNDANT_TYPED : int = 1
	const _INTEGER_REDUNDANT_TYPED2 : int = 1
	const _INTEGER_REDUNDANT_INFERRED := 1
	const _INTEGER_REDUNDANT_INFERRED2 : = 1
