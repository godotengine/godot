func test():
	# Non-string keys are valid.
	print({ 12: "world" }[12])

	var contents = {
		0: "zero",
		0.0: "zero point zero",
		null: "null",
		false: "false",
		[]: "empty array",
		Vector2i(): "zero Vector2i",
		15: {
			22: {
				4: ["nesting", "arrays"],
			},
		},
	}

	print(contents[0.0])
	# Making sure declaration order doesn't affect things...
	print({ 0.0: "zero point zero", 0: "zero",  null: "null", false: "false", []: "empty array" }[0])
	print({ 0.0: "zero point zero", 0: "zero", null: "null", false: "false", []: "empty array" }[0.0])

	print(contents[null])
	print(contents[false])
	print(contents[[]])
	print(contents[Vector2i()])
	print(contents[15])
	print(contents[15][22])
	print(contents[15][22][4])
	print(contents[15][22][4][0])
	print(contents[15][22][4][1])

	# Currently fails with "invalid get index 'hello' on base Dictionary".
	# Both syntaxes are valid however.
	#print({ "hello": "world" }["hello"])
	#print({ "hello": "world" }.hello)
