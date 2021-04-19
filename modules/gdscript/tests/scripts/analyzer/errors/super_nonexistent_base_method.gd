# `class` extends RefCounted by default.
class Say:
	func say():
		super()
		print("say something")


func test():
	# RefCounted doesn't have a `say()` method, so the `super()` call in the method
	# definition will cause a run-time error.
	Say.new().say()
