var _typed_array: Array[int]

func weak_param_func(weak_param = _typed_array):
	weak_param = [11] # Don't treat the literal as typed!
	return weak_param

func hard_param_func(hard_param := _typed_array):
	hard_param = [12]
	return hard_param

func test():
	var weak_var = _typed_array
	print(weak_var.is_typed())
	weak_var = [21] # Don't treat the literal as typed!
	print(weak_var.is_typed())
	print(weak_param_func().is_typed())

	var hard_var := _typed_array
	print(hard_var.is_typed())
	hard_var = [22]
	print(hard_var.is_typed())
	print(hard_param_func().is_typed())
