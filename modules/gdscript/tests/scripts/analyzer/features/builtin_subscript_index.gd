# https://github.com/godotengine/godot/issues/120318
func test():
	# TYPE_VECTOR2
	var v2 := Vector2(1, 2)
	print(v2[0])
	print(v2["x"])

	# TYPE_VECTOR2I
	var v2i := Vector2i(1, 2)
	print(v2i[0])
	print(v2i["x"])

	# TYPE_VECTOR3
	var v3 := Vector3(1, 2, 3)
	print(v3[0])
	print(v3["x"])

	# TYPE_VECTOR3I
	var v3i := Vector3i(1, 2, 3)
	print(v3i[0])
	print(v3i["x"])

	# TYPE_TRANSFORM2D
	var t2d := Transform2D.IDENTITY
	print(t2d[0])
	print(t2d["x"])

	# TYPE_VECTOR4
	var v4 := Vector4(1, 2, 3, 4)
	print(v4[0])
	print(v4["x"])

	# TYPE_VECTOR4I
	var v4i := Vector4i(1, 2, 3, 4)
	print(v4i[0])
	print(v4i["x"])

	# TYPE_QUATERNION
	var q := Quaternion(1, 2, 3, 4)
	print(q[0])
	print(q["x"])

	# TYPE_BASIS
	var b := Basis.IDENTITY
	print(b[0])
	print(b["x"])

	# TYPE_PROJECTION
	var proj := Projection.IDENTITY
	print(proj[0])
	print(proj["x"])
