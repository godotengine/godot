class MyObject:
	var my_prop = 0

func test():
	var obj = MyObject.new()
	print(obj.get('my_prop'))
	print(obj.get('my_prop', -1))
	print(obj.get('missing_prop'))
	print(obj.get('missing_prop', -1))
