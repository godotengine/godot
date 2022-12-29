class A extends CanvasItem:
	func _init():
		print('no')

class B extends A:
	pass

func test():
	B.new()
