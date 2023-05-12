class Iterator:
	func _iter_init(_count):
		return true
	func _iter_next(_count):
		return false
	func _iter_get(_count) -> StringName:
		return &'custom'

func test():
	var hard_iterator := Iterator.new()

	for x in hard_iterator:
		if x is int:
			pass
