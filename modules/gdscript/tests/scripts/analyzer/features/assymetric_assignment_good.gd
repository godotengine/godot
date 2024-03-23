const const_color: Color = 'red'

func func_color(arg_color: Color = 'blue') -> bool:
	return arg_color == Color.BLUE

@warning_ignore("assert_always_true")
func test():
	assert(const_color == Color.RED)

	assert(func_color() == true)
	assert(func_color('blue') == true)

	var var_color: Color = 'green'
	assert(var_color == Color.GREEN)

	print('ok')
