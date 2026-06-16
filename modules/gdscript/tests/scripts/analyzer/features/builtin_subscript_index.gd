# https://github.com/godotengine/godot/issues/120318
func test():
	# TYPE_VECTOR2
	var v2 := Vector2(1, 2)
	print(v2[0])
	print(v2[1])
	print(v2["x"])
	print(v2["y"])

	# TYPE_VECTOR2I
	var v2i := Vector2i(1, 2)
	print(v2i[0])
	print(v2i[1])
	print(v2i["x"])
	print(v2i["y"])

	# TYPE_VECTOR3
	var v3 := Vector3(1, 2, 3)
	print(v3[0])
	print(v3[1])
	print(v3[2])
	print(v3["x"])
	print(v3["y"])
	print(v3["z"])

	# TYPE_VECTOR3I
	var v3i := Vector3i(1, 2, 3)
	print(v3i[0])
	print(v3i[1])
	print(v3i[2])
	print(v3i["x"])
	print(v3i["y"])
	print(v3i["z"])

	# TYPE_TRANSFORM2D
	var t2d := Transform2D.IDENTITY
	print(t2d[0])
	print(t2d[1])
	print(t2d[2])
	print(t2d["x"])
	print(t2d["y"])
	print(t2d["origin"])

	# TYPE_VECTOR4
	var v4 := Vector4(1, 2, 3, 4)
	print(v4[0])
	print(v4[1])
	print(v4[2])
	print(v4[3])
	print(v4["x"])
	print(v4["y"])
	print(v4["z"])
	print(v4["w"])

	# TYPE_VECTOR4I
	var v4i := Vector4i(1, 2, 3, 4)
	print(v4i[0])
	print(v4i[1])
	print(v4i[2])
	print(v4i[3])
	print(v4i["x"])
	print(v4i["y"])
	print(v4i["z"])
	print(v4i["w"])

	# TYPE_QUATERNION
	var q := Quaternion(1, 2, 3, 4)
	print(q[0])
	print(q[1])
	print(q[2])
	print(q[3])
	print(q["x"])
	print(q["y"])
	print(q["z"])
	print(q["w"])

	# TYPE_BASIS
	var b := Basis.IDENTITY
	print(b[0])
	print(b[1])
	print(b[2])
	print(b["x"])
	print(b["y"])
	print(b["z"])

	# TYPE_PROJECTION
	var proj := Projection.IDENTITY
	print(proj[0])
	print(proj[1])
	print(proj[2])
	print(proj[3])
	print(proj["x"])
	print(proj["y"])
	print(proj["z"])
	print(proj["w"])
