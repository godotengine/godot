func test():
	var lua_dict_with_string = {
		a = 1,
		b = 2,
		"a" = 3, # Duplicate isn't allowed.
	}
