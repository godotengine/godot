# `class` extends Reference by default.
class Say:
	func say():
		super()
		print("say something")


func test():
	# Reference doesn't have a `say()` method, so the `super()` call in the method
	# definition will cause a run-time error.
	Say.new().say()
