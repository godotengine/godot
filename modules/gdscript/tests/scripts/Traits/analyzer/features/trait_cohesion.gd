extends Node
uses TraitA, TraitB
uses TraitC, TraitC.TraitD, ClassA.TraitE
uses GlobalTrait

trait TraitA extends Object:
	func func_one():
		print("function one")

trait TraitB:
	func func_two():
		print("function two")

trait TraitC:
	static func func_three():
		print("function three")

	func func_four()

	trait TraitD:
		uses TraitC
		signal signal_one(args)
		enum EnumOne { VALUE_ONE, VALUE_TWO }

class ClassA:
	trait TraitE:
		uses TraitC.TraitD
		var var_one: String = "trait variable"
		const const_one: String = "trait constant"

static func func_three(): # override using 'static' keyword is optional but result in warning.
	print("overridden function three")

func func_four():
	print("function four implemented in class")


class Actor extends Intermediate:
	uses Interactable

class Intermediate extends BaseCharacter: pass

class BaseCharacter extends Node: pass

trait Interactable extends BaseCharacter:
	func interact():
		print("Interactable")

func test():
	print("signal connect:", signal_one.connect(print) == OK)
	signal_one.emit("test signal")
	print(var_one)
	print(const_one)
	print("Trait Enum value:", EnumOne.VALUE_ONE)
	print("Trait Enum value:", EnumOne.VALUE_TWO)
	func_one()
	func_two()
	func_three()
	func_four()
	var act = Actor.new()
	act.interact()
	act.free()
	print("ok")
