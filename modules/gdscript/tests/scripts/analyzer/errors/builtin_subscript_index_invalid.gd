func test():
	# TYPE_RECT2
	var r: Rect2 = Rect2()
	print(r[0])

	# TYPE_RECT2I
	var ri: Rect2i = Rect2i()
	print(ri[0])

	# TYPE_PLANE
	var p: Plane = Plane()
	print(p[0])

	# TYPE_AABB
	var a: AABB = AABB()
	print(a[0])

	# TYPE_TRANSFORM3D
	var t: Transform3D = Transform3D()
	print(t[0])

	# TYPE_OBJECT
	var o: Object = null
	print(o[0])
