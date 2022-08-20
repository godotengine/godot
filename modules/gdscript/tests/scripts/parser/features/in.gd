func test():
	print("dot" in "Godot")
	print(not "i" in "team")

	print(true in [true, false])
	print(not null in [true, false])
	print(null in [null])

	print(26 in [8, 26, 64, 100])
	print(not Vector2i(10, 20) in [Vector2i(20, 10)])

	print("apple" in { "apple": "fruit" })
	print("apple" in { "apple": null })
	print(not "apple" in { "fruit": "apple" })
