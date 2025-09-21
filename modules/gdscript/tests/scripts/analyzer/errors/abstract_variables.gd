@abstract class BaseAbstractVar extends Node:
	@abstract var a: int
	@abstract var b: Node2D
	@abstract var c: Vector2

class ConcreteFulfilled extends BaseAbstractVar:
	@override var a = 1
	@override var b = Node2D.new()
	@override var c = Vector2i.ZERO

class ConcreteUnfulfilled extends BaseAbstractVar:
	pass

class ConcreteDeclaresAbstractVar:
	@abstract var d: String

@abstract class DoubleAbstractVar:
	@abstract @abstract var a: int

@abstract class InitializedAbstractVar:
	@abstract var a: int = 1

func test():
	pass
