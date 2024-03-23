class A extends InstancePlaceholder:
	func _init():
		print('no')

class B extends A:
	pass

func test():
	B.new()
