func expect_typed(typed: Dictionary[int, int]):
	print(typed.size())

func test():
	var float_dict: Dictionary[float, float] = { 1.0: 0.0 }
	var integer := 1

	var dict_1: Dictionary[int, int] = { "Hello": "World" }
	var dict_2: Dictionary[int, int] = float_dict
	var dict_3: Dictionary[Object, Object] = { integer: integer }
	expect_typed(float_dict)

	# Subscript access for Variant-typed key should not error (GH-121780).
	var variant_key_dict: Dictionary[Variant, int] = { "key": 1 }
	var key_untyped = "key"
	var key_string_typed: String = "key"
	var key_variant_typed: Variant = "key"
	variant_key_dict["key"]
	variant_key_dict[key_untyped]
	variant_key_dict[key_string_typed]
	variant_key_dict[key_variant_typed]
