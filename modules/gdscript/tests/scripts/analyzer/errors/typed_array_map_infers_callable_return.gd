func test():
	var mapped := [1, 2, 3].map(func(a: int) -> float: return float(a) / 2.0)
	var typed: Array[int] = mapped
	print('not ok')
