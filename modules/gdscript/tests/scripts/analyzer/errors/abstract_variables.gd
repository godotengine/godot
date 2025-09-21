@abstract class BaseAbstractVar extends Node:
	@abstract var a: int
	@abstract var b: Node2D
	@abstract var c: Vector2

class ConcreteFulfilled extends BaseAbstractVar:
	var a = 1
	var b = Node2D.new()
	var c = Vector2i.ZERO

class ConcreteOverride extends ConcreteFulfilled:
	var a = 2

class ConcreteUnfulfilled extends BaseAbstractVar:
	pass

class ConcreteDeclaresAbstractVar:
	@abstract var d: String

@abstract class MismatchVar extends BaseAbstractVar:
	var a: String = "test"

@abstract class DoubleAbstractVar extends BaseAbstractVar:
	@abstract var a: int
	@abstract @abstract var d: int

@abstract class ExportAbstractVar extends BaseAbstractVar:
	@export var a: int
	@export @abstract var d: int
	@abstract @export var e: int

@abstract class OnReadyAbstractVar extends BaseAbstractVar:
	@onready var b: Node2D
	@onready @abstract var d: Node2D
	@abstract @onready var e: Node2D

func test():
	pass