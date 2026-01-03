func test():
	var differently: Dictionary[float, float] = { 1.0: 0.0 }
	var typed: Dictionary[int, int] = differently
	print('not ok')
