var foo = 123


func test():
	# Notice the `var` keyword. Without this keyword, no warning would be emitted
	# because no new variable would be created. Instead, the class variable's value
	# would be overwritten.
	var foo = 456
