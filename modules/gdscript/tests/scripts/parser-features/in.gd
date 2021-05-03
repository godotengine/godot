func test():
	assert("dot" in "Godot")
	assert(not "i" in "team")

	assert(true in [true, false])
	assert(not null in [true, false])
	assert(null in [null])

	assert(26 in [8, 26, 64, 100])
	assert(not Vector2i(10, 20) in [Vector2i(20, 10)])

	assert("apple" in { "apple": "fruit" })
	assert("apple" in { "apple": null })
	assert(not "apple" in { "fruit": "apple" })
