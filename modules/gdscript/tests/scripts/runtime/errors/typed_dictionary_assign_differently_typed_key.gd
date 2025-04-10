func get_key() -> Variant:
	return "key"

func test():
	var typed: Dictionary[int, int]
	typed[get_key()] = 0
	print('not ok')
