const const_color: Color = 'red'

func func_color(arg_color: Color = 'blue') -> bool:
	return arg_color == Color.BLUE

func test():
	Utils.check(const_color == Color.RED)

	Utils.check(func_color() == true)
	Utils.check(func_color('blue') == true)

	var var_color: Color = 'green'
	Utils.check(var_color == Color.GREEN)

	print('ok')
