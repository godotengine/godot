func sort_descending(a, b) -> bool:
	return a > b

# Color can not be compared
func color_compare(a: Color, b: Color) -> bool:
	return a.to_rgba32() > b.to_rgba32()

func test():
	var bytes := PackedByteArray([1, 2, 3])
	bytes.sort_custom(sort_descending)
	print(Array(bytes)) # Use Array() to print bytes correctly

	var colors := PackedColorArray([Color(.25, .25, .25, .25), Color(.5, .5, .5, .5), Color(.75, .75, .75, .75)])
	colors.sort_custom(color_compare)
	print(Array(colors)) # Use Array() to print colors correctly

	var float32s := PackedFloat32Array([1.25, 2.5, 3.75])
	float32s.sort_custom(sort_descending)
	print(float32s)

	var float64s := PackedFloat64Array([1.25, 2.5, 3.75])
	float64s.sort_custom(sort_descending)
	print(float64s)

	var int32s := PackedInt32Array([1, 2, 3])
	int32s.sort_custom(sort_descending)
	print(int32s)

	var int64s := PackedInt64Array([1, 2, 3])
	int64s.sort_custom(sort_descending)
	print(int64s)

	var strings := PackedStringArray(["1", "2", "3"])
	strings.sort_custom(sort_descending)
	print(strings)

	var vector2s := PackedVector2Array([Vector2(1, 1), Vector2(2, 2), Vector2(3, 3)])
	vector2s.sort_custom(sort_descending)
	print(vector2s)

	var vector3s := PackedVector3Array([Vector3(1, 1, 1), Vector3(2, 2, 2), Vector3(3, 3, 3)])
	vector3s.sort_custom(sort_descending)
	print(vector3s)
