func sort_descending(a, b) -> bool:
	return a > b

# Color can not be compared
func color_compare(a: Color, b: Color) -> bool:
	return a.to_rgba32() > b.to_rgba32()

func test():
	var bytes := PackedByteArray([3, 2, 1])
	print(bytes.bsearch_custom(3, sort_descending))

	var colors := PackedColorArray([Color(.75, .75, .75, .75), Color(.5, .5, .5, .5), Color(.25, .25, .25, .25)])
	print(colors.bsearch_custom(Color(.75, .75, .75, .75), color_compare, true))

	var float32s := PackedFloat32Array([3.75, 2.5, 1.25])
	print(float32s.bsearch_custom(3.75, sort_descending, false))

	var float64s := PackedFloat64Array([3.75, 2.5, 1.25])
	print(float64s.bsearch_custom(3.75, sort_descending))

	var int32s := PackedInt32Array([3, 2, 1])
	print(int32s.bsearch_custom(3, sort_descending, true))

	var int64s := PackedInt64Array([3, 2, 1])
	print(int64s.bsearch_custom(3, sort_descending, false))

	var strings := PackedStringArray(["3", "2", "1"])
	print(strings.bsearch_custom("3", sort_descending))

	var vector2s := PackedVector2Array([Vector2(3, 3), Vector2(2, 2), Vector2(1, 1)])
	print(vector2s.bsearch_custom(Vector2(3, 3), sort_descending, true))

	var vector3s := PackedVector3Array([Vector3(3, 3, 3), Vector3(2, 2, 2), Vector3(1, 1, 1)])
	print(vector3s.bsearch_custom(Vector3(3, 3, 3), sort_descending, false))
