var _prop : int

# Setter function has wrong argument type.
var prop : String:
	set = set_prop

func set_prop(value : int):
	_prop = value

func test():
	pass
