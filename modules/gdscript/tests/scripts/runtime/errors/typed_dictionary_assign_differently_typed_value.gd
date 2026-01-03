func get_value() -> Variant:
	return "value"

func test():
	var typed: Dictionary[int, int]
	typed[0] = get_value()
	print("not ok")
