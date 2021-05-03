func test():
	# Dictionaries with consecutive commas are not allowed.
	var dictionary = { "hello": "world",, }

	dictionary = { hello = "world",, }
