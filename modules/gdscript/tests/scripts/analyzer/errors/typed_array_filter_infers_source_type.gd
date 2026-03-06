func test():
	var source: Array[int] = [1, 2, 3]
	var filtered := source.filter(func(a): return a > 1)
	var typed: Array[String] = filtered
	print('not ok')
