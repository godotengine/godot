extends RefCounted

@abstract class ABCLocal:
	@abstract func simulate() -> void

	func coroutine() -> void:
		@warning_ignore("redundant_await")
		await 0

	func other_method() -> void:
		await simulate()

class ImplLocal extends ABCLocal:
	func simulate() -> void:
		await coroutine()

class Impl extends ABC:
	func simulate() -> void:
		await coroutine()

class ImplChild extends ABCChild:
	func simulate() -> void:
		await coroutine()

func test() -> void:
	# Local file.
	var l: ABCLocal = ImplLocal.new()
	await l.simulate()
	await l.other_method()

	# Direct child.
	var a: ABC = Impl.new()
	await a.simulate()
	await a.other_method()

	# Indirect child.
	var b: ABCChild = ImplChild.new()
	await b.simulate()
	await b.other_method()
