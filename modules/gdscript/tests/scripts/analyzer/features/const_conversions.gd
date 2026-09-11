const const_float_int: float = 19
const const_float_plus: float = 12 + 22
const const_float_cast: float = 76 as float

func test():
	Utils.check(typeof(const_float_int) == TYPE_FLOAT)
	Utils.check(str(const_float_int) == '19.0')
	Utils.check(typeof(const_float_plus) == TYPE_FLOAT)
	Utils.check(str(const_float_plus) == '34.0')
	Utils.check(typeof(const_float_cast) == TYPE_FLOAT)
	Utils.check(str(const_float_cast) == '76.0')

	print('ok')
