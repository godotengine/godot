var _prop = 1
var prop:
	get = get_prop, set = set_prop

func get_prop():
	return _prop

func set_prop(value):
	_prop = value

func test():
	print(prop)

	prop = 2

	print(prop)
