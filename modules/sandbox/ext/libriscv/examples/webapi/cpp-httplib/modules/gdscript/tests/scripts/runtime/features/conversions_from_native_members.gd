class Foo extends Node:
	func _init():
		name = 'f'
		var string: String = name
		Utils.check(typeof(string) == TYPE_STRING)
		Utils.check(string == 'f')
		print('ok')

func test():
	var foo := Foo.new()
	foo.free()
