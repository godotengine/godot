func test():
	var left_hard_int := 1
	var right_weak_int = 2
	var result_hm_int := left_hard_int if true else right_weak_int

	print('not ok')
