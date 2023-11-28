func test():
	var _sub1 = SubClass1.new("test")
	var _sub2 = SubClass2.new()
	var _sub3 = SubClass3.new()

# Both with params.
class SuperClass1:
	func _init(param):
		prints("SuperClass1 init", param)

class SubClass1 extends SuperClass1:
	func _init(param):
		prints("SubClass1 init", param)
		super(param)

# Only super with params.
class SuperClass2:
	func _init(param):
		prints("SuperClass2 init", param)

class SubClass2 extends SuperClass2:
	func _init():
		prints("SubClass2 init")
		super("super")

# Super without params.
class SuperClass3:
	func _init():
		prints("SuperClass3 init")

class SubClass3 extends SuperClass3:
	func _init():
		prints("SubClass3 init")
		super()
