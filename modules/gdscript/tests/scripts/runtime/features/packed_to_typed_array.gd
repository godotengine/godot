func test():
	test_typed_array()
	test_array()

func test_typed_array():
	var _bytes : Array[int] = Array(PackedByteArray())
	var _i32s : Array[int] = Array(PackedInt32Array())
	var _i64s : Array[int] = Array(PackedInt64Array())
	var _f32s : Array[float] = Array(PackedFloat32Array())
	var _f64s : Array[float] = Array(PackedFloat64Array())
	var _strings : Array[String] = Array(PackedStringArray())
	var _vec2s : Array[Vector2] = Array(PackedVector2Array())
	var _vec3s : Array[Vector3] = Array(PackedVector3Array())
	var _colors : Array[Color] = Array(PackedColorArray())

func test_array():
	var _bytes : Array = Array(PackedByteArray())
	var _i32s : Array = Array(PackedInt32Array())
	var _i64s : Array = Array(PackedInt64Array())
	var _f32s : Array = Array(PackedFloat32Array())
	var _f64s : Array = Array(PackedFloat64Array())
	var _strings : Array = Array(PackedStringArray())
	var _vec2s : Array = Array(PackedVector2Array())
	var _vec3s : Array = Array(PackedVector3Array())
	var _colors : Array = Array(PackedColorArray())
