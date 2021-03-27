func test():
	var lua_dict = {
		a = 1,
		"b" = 2, # Using strings are allowed too.
		"with spaces" = 3, # Especially useful when key has spaces...
		"2" = 4, # ... or invalid identifiers.
	}

	print(lua_dict)
