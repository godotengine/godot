func test_match(a, b):
	match a:
		b:
			prints("Error:", var_to_str(a), "matched", var_to_str(b))
	match b:
		a:
			prints("Error:", var_to_str(b), "matched", var_to_str(a))

func test():
	test_match(1, 1.0)
	test_match(Vector2(1, 2), Vector2i(1, 2))
	test_match(Vector3(1, 2, 3), Vector3i(1, 2, 3))
	test_match(Vector4(1, 2, 3, 4), Vector4i(1, 2, 3, 4))
	test_match(Rect2(1, 2, 3, 4), Rect2i(1, 2, 3, 4))

	test_match("abc", ^"abc")
	test_match("abc", &"abc")
	test_match(^"abc", &"abc")

	# Note: Typed arrays have some issues about this.
	var int_array: Array[int] = [1]
	var float_array: Array[float] = [1]
	test_match(int_array, float_array)

	test_match([1], PackedInt64Array([1]))
	test_match([1.0], PackedFloat64Array([1]))

	test_match(PackedInt32Array(), PackedFloat32Array())
	test_match(PackedInt64Array(), PackedFloat64Array())
	test_match(PackedInt32Array(), PackedInt64Array())
	test_match(PackedFloat32Array(), PackedFloat64Array())
