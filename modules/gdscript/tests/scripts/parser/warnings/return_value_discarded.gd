func i_return_int() -> int:
	return 4


func test():
	i_return_int()
	preload("../../utils.notest.gd") # `preload` is a function-like keyword.
