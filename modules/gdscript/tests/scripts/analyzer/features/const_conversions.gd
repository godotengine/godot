const const_float_int: float = 19
const const_float_plus: float = 12 + 22
const const_float_cast: float = 76 as float

const const_packed_empty: PackedFloat64Array = []
const const_packed_ints: PackedFloat64Array = [52]

@warning_ignore("assert_always_true")
func test():
	assert(typeof(const_float_int) == TYPE_FLOAT)
	assert(str(const_float_int) == '19')
	assert(typeof(const_float_plus) == TYPE_FLOAT)
	assert(str(const_float_plus) == '34')
	assert(typeof(const_float_cast) == TYPE_FLOAT)
	assert(str(const_float_cast) == '76')

	assert(typeof(const_packed_empty) == TYPE_PACKED_FLOAT64_ARRAY)
	assert(str(const_packed_empty) == '[]')
	assert(typeof(const_packed_ints) == TYPE_PACKED_FLOAT64_ARRAY)
	assert(str(const_packed_ints) == '[52]')
	assert(typeof(const_packed_ints[0]) == TYPE_FLOAT)
	assert(str(const_packed_ints[0]) == '52')

	print('ok')
