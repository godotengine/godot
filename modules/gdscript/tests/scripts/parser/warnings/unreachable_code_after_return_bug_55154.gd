func test():
	var foo := "bar"
	match foo:
		"baz":
			return
		_:
			pass
	match foo:
		"baz":
			return
	match foo:
		"bar":
			pass
		_:
			return
	print("reached")
