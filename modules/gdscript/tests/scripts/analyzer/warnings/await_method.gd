extends RefCounted

@abstract class ClassA extends Node:
	@abstract func simulate() -> ClassA

	func coroutine() -> void:
		@warning_ignore("redundant_await")
		await 0

class ClassAImpl extends ClassA:
	func simulate() -> ClassA:
		await coroutine()
		return self

func test() -> void:
	var a: ClassA = ClassAImpl.new()
	await a.simulate()
