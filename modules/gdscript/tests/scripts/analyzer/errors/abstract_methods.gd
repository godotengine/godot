abstract class AbstractClass:
	abstract func some_func()

class ImplementedClass extends AbstractClass:
	func some_func():
		pass

abstract class AbstractClassAgain extends ImplementedClass:
	abstract func some_func()

class Test1:
	abstract func some_func()

class Test2 extends AbstractClass:
	pass

class Test3 extends AbstractClassAgain:
	pass

class Test4 extends AbstractClass:
	func some_func():
		super()

	func other_func():
		super.some_func()

func test():
	pass
