func test():
	Vector2(
		1,
		2
	)

	Vector3(
		3,
		3.5,
		4,  # Trailing comma should work.
	)

	Vector2i(1, 2,)  # Trailing comma should work.

	Vector3i(6,
	9,
		12)
