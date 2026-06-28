var weakling = 'not float'
func weak(x: float = weakling):
	print(x)
	print('typeof x is', typeof(x))

func test():
	print(typeof(weak()))
	print('not ok')
