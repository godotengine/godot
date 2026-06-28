func test():
	var never: Variant = false
	if never:
		assert(false)
		assert(false, 'message')
	print('ok')
