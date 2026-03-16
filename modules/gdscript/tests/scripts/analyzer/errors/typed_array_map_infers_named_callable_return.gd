func half(a: int) -> float:
	return float(a) / 2.0

func test():
	var callable = half
	var mapped := [1, 2, 3].map(callable)
	var typed: Array[int] = mapped
	print('not ok')
