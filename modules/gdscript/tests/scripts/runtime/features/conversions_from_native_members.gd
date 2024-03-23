class Foo extends Node:
	func _init():
		name = 'f'
		var string: String = name
		assert(typeof(string) == TYPE_STRING)
		assert(string == 'f')
		print('ok')

func test():
	var foo := Foo.new()
	foo.free()
