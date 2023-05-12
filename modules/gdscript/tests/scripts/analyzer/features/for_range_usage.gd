func test():
	var array := [3, 6, 9]
	var result := ''
	for i in range(array.size(), 0, -1):
		result += str(array[i - 1])
	assert(result == '963')
	print('ok')
