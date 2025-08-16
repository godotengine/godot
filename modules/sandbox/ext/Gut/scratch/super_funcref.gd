extends SceneTree

class SuperClass:

	func print_hello():
		print('hello')

class  SubClass:
	extends SuperClass

	func print_hello():
		print('world')

func _init():

	var sub = SubClass.new()
	var fref = sub.print_hello
	print(fref.is_valid())
	fref.call_func()

	quit()
