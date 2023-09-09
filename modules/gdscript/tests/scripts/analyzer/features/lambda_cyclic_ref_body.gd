# GH-70592

var f: Callable = func ():
	x = 2
	return 1

var x: int = f.call()

var g: Array[Callable] = [
	func ():
		y += 10
		return 1,
	func ():
		y += 20
		return 2,
]

var y: int = g[0].call() + g[1].call()

func test():
	print(x)
	f.call()
	print(x)

	print(y)
	g[0].call()
	g[1].call()
	print(y)

	# This prevents memory leak in CI. TODO: Investigate it.
	# Also you cannot run the `EditorScript` twice without the cleaning. Error:
	# Condition "!p_keep_state && has_instances" is true. Returning: ERR_ALREADY_IN_USE
	f = Callable()
	g.clear()
