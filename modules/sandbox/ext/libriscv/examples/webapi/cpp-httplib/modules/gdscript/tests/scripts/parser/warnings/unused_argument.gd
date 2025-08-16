# This should emit a warning since the unused argument is not prefixed with an underscore.
func function_with_unused_argument(p_arg1, p_arg2):
	print(p_arg1)


# This shouldn't emit a warning since the unused argument is prefixed with an underscore.
func function_with_ignored_unused_argument(p_arg1, _p_arg2):
	print(p_arg1)


func test():
	pass
