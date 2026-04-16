class Base:
	func foo():
		pass


class Other:
	pass


func test():
	var obj = Base.new():
		extends Other
		func bar():
			pass
	print(obj)
