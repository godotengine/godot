func test():
	var lua_dict = {
		a = 1,
		b = 2,
		a = 3, # Duplicate isn't allowed.
	}
