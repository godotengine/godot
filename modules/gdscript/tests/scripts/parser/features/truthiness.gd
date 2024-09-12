func test():
	# The checks below should all evaluate to `true` for this test to pass.
	Utils.check(true)
	Utils.check(not false)
	Utils.check(500)
	Utils.check(not 0)
	Utils.check(500.5)
	Utils.check(not 0.0)
	Utils.check("non-empty string")
	Utils.check(["non-empty array"])
	Utils.check({"non-empty": "dictionary"})
	Utils.check(Vector2(1, 0))
	Utils.check(Vector2i(-1, -1))
	Utils.check(Vector3(0, 0, 0.0001))
	Utils.check(Vector3i(0, 0, 10000))

	# Zero position is `true` only if the Rect2's size is non-zero.
	Utils.check(Rect2(0, 0, 0, 1))

	# Zero size is `true` only if the position is non-zero.
	Utils.check(Rect2(1, 1, 0, 0))

	# Zero position is `true` only if the Rect2's size is non-zero.
	Utils.check(Rect2i(0, 0, 0, 1))

	# Zero size is `true` only if the position is non-zero.
	Utils.check(Rect2i(1, 1, 0, 0))

	# A fully black color is only truthy if its alpha component is not equal to `1`.
	Utils.check(Color(0, 0, 0, 0.5))

	print("ok")
