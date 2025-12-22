const UNTYPED_ARRAY: Variant = [1000]
const TYPED_ARRAY: Variant = [2000] as Array[int]
const UNTYPED_DICT: Variant = {1000: 1000}
const TYPED_DICT: Variant = {2000: 2000} as Dictionary[int, int]

const TEST_STRING_INDEX = "data"[0]
const TEST_VECTOR2_ATTR = Vector2(1, 2).x
const TEST_ARRAY_INDEX = [100, 200][0]
const TEST_DICT_ATTR = {a = 1, b = 2}.a

const TEST_CALL_ARRAY = Array()
const TEST_CALL_DICT = Dictionary()

const TEST_BINARY_OP = [1] + [2]
const TEST_TERNARY_OP = [123] if [true] else [456]

const TEST_CAST_UNTYPED_ARRAY = [11] as Array
const TEST_CAST_TYPED_ARRAY = [22] as Array[int]
const TEST_CAST_UNTYPED_DICT = {11: 111} as Dictionary
const TEST_CAST_TYPED_DICT = {22: 222} as Dictionary[int, int]

const TEST_TYPE_TEST_LITERAL_UNTYPED_ARRAY = [33] is Array
const TEST_TYPE_TEST_LITERAL_TYPED_ARRAY = [44] is Array[int]
const TEST_TYPE_TEST_LITERAL_UNTYPED_DICT = {33: 333} is Dictionary
const TEST_TYPE_TEST_LITERAL_TYPED_DICT = {44: 444} is Dictionary[int, int]

const TEST_TYPE_TEST_UNTYPED_UNTYPED_ARRAY = UNTYPED_ARRAY is Array
const TEST_TYPE_TEST_UNTYPED_TYPED_ARRAY = UNTYPED_ARRAY is Array[int]
const TEST_TYPE_TEST_UNTYPED_UNTYPED_DICT = UNTYPED_DICT is Dictionary
const TEST_TYPE_TEST_UNTYPED_TYPED_DICT = UNTYPED_DICT is Dictionary[int, int]

const TEST_TYPE_TEST_TYPED_UNTYPED_ARRAY = TYPED_ARRAY is Array
const TEST_TYPE_TEST_TYPED_TYPED_ARRAY = TYPED_ARRAY is Array[int]
const TEST_TYPE_TEST_TYPED_UNTYPED_DICT = TYPED_DICT is Dictionary
const TEST_TYPE_TEST_TYPED_TYPED_DICT = TYPED_DICT is Dictionary[int, int]

@export_custom(
	PROPERTY_HINT_TYPE_STRING,
	"%d/%d:Zero,One,Two" % [TYPE_INT, PROPERTY_HINT_ENUM],
)
var test_export_custom: Array

func test():
	var script: Script = get_script()
	var constants: Dictionary = script.get_script_constant_map()
	for constant: StringName in constants:
		if constant.begins_with("TEST_"):
			print("const %s = %s" % [constant, var_to_str(constants[constant]).replace("\n", "")])
			if constants[constant] is Array:
				var array: Array = constants[constant]
				Utils.check(array.is_read_only())
				array = get(constant)
				Utils.check(array.is_read_only())
			elif constants[constant] is Dictionary:
				var dict: Dictionary = constants[constant]
				Utils.check(dict.is_read_only())
				dict = get(constant)
				Utils.check(dict.is_read_only())

	for property in get_property_list():
		if str(property.name).begins_with("test_"):
			Utils.print_property_extended_info(property, self)
