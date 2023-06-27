const constant_float = 1.0
const constant_int = 1
enum { enum_value = 1 }

class Iterator:
	func _iter_init(_count):
		return true
	func _iter_next(_count):
		return false
	func _iter_get(_count) -> StringName:
		return &'custom'

func test():
	var hard_float := 1.0
	var hard_int := 1
	var hard_string := '0'
	var hard_iterator := Iterator.new()

	var variant_float: Variant = hard_float
	var variant_int: Variant = hard_int
	var variant_string: Variant = hard_string
	var variant_iterator: Variant = hard_iterator

	for i in 1.0:
		print(typeof(i) == TYPE_FLOAT)
	for i in 1:
		print(typeof(i) == TYPE_INT)
	for i in 'a':
		print(typeof(i) == TYPE_STRING)
	for i in Iterator.new():
		print(typeof(i) == TYPE_STRING_NAME)

	for i in hard_float:
		print(typeof(i) == TYPE_FLOAT)
	for i in hard_int:
		print(typeof(i) == TYPE_INT)
	for i in hard_string:
		print(typeof(i) == TYPE_STRING)
	for i in hard_iterator:
		print(typeof(i) == TYPE_STRING_NAME)

	for i in variant_float:
		print(typeof(i) == TYPE_FLOAT)
	for i in variant_int:
		print(typeof(i) == TYPE_INT)
	for i in variant_string:
		print(typeof(i) == TYPE_STRING)
	for i in variant_iterator:
		print(typeof(i) == TYPE_STRING_NAME)

	print('ok')
