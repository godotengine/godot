func test():
	var x := 5

	assert(x > 0)
	assert(x > 0,)
	assert(x > 0, 'message')
	assert(x > 0, 'message',)

	assert(
		x > 0
	)
	assert(
		x > 0,
	)
	assert(
		x > 0,
		'message'
	)
	assert(
		x > 0,
		'message',
	)

	print('OK')
