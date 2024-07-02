func test():
	const a := 1
	const b := 2
	const c := a + b
	const d := "test"
	const test_str := "%d-%d-%d-%s" % [a, b, c, d]

	print(test_str)
