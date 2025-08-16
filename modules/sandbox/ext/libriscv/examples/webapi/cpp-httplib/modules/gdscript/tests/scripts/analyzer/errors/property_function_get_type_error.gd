var _prop: int

# Getter function has wrong return type.
var prop: String:
	get = get_prop

func get_prop():
	return _prop

func test():
	pass
