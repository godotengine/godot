const array: Array = [0]
const dictionary := {1: 2}

@warning_ignore("assert_always_true")
func test():
	assert(array.is_read_only() == true)
	assert(str(array) == '[0]')
	assert(dictionary.is_read_only() == true)
	assert(str(dictionary) == '{ 1: 2 }')
	print('ok')
