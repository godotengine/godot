func literal(x: float = 1):
	print('x is ', x)
	print('typeof x is ', typeof(x))

var inferring := 2
func inferred(x: float = inferring):
	print('x is ', x)
	print('typeof x is ', typeof(x))

var weakling = 3
func weak(x: float = weakling):
	print('x is ', x)
	print('typeof x is ', typeof(x))

func test():
	literal()
	inferred()
	weak()
	print('ok')
