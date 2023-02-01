func test():
	var left_hard_int := 1
	var right_hard_int := 2
	var result_hard_int := left_hard_int if true else right_hard_int
	assert(result_hard_int == 1)

	var left_hard_variant := 1 as Variant
	var right_hard_variant := 2.0 as Variant
	var result_hard_variant := left_hard_variant if true else right_hard_variant
	assert(result_hard_variant == 1)

	print('ok')
