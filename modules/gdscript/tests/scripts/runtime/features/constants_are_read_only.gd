const array: Array = [0]
const dictionary := {1: 2}

func test():
	Utils.check(array.is_read_only() == true)
	Utils.check(str(array) == '[0]')
	Utils.check(dictionary.is_read_only() == true)
	Utils.check(str(dictionary) == '{ 1: 2 }')
	print('ok')
