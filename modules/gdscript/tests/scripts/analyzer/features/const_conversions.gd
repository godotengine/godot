const const_float_int: float = 19
const const_float_plus: float = 12 + 22
const const_float_cast: float = 76 as float

const const_packed_empty: PackedFloat64Array = []
const const_packed_ints: PackedFloat64Array = [52]

func test():
	Utils.check(typeof(const_float_int) == TYPE_FLOAT)
	Utils.check(str(const_float_int) == '19.0')
	Utils.check(typeof(const_float_plus) == TYPE_FLOAT)
	Utils.check(str(const_float_plus) == '34.0')
	Utils.check(typeof(const_float_cast) == TYPE_FLOAT)
	Utils.check(str(const_float_cast) == '76.0')

	Utils.check(typeof(const_packed_empty) == TYPE_PACKED_FLOAT64_ARRAY)
	Utils.check(str(const_packed_empty) == '[]')
	Utils.check(typeof(const_packed_ints) == TYPE_PACKED_FLOAT64_ARRAY)
	Utils.check(str(const_packed_ints) == '[52.0]')
	Utils.check(typeof(const_packed_ints[0]) == TYPE_FLOAT)
	Utils.check(str(const_packed_ints[0]) == '52.0')

	print('ok')
