trait TraitA:
	func some_func()

	func some_func_2(_param: Node):
		print("some func")

	func some_func_3(_param1, _param2):
		print("some func")

	func some_func_4():
		print("some func")

trait BadTrait extends CollisionObject2D:
	func move_and_collide()

trait BadTrait2:
	func some_func():
		print("some func")

trait BadTrait3:
	uses TraitA

	func some_func(): # Trait do not override trait function. hence, error.
		print("some func")

class SomeClass extends CharacterBody2D:
	uses TraitA
	# Implementing function some_func() is skipped. Since bodyless in trait leads to an error.

	uses TraitA # Using the same trait again in same class, leads to error.
	uses BadTrait # Trait has 'move_and_collide()' but function shadows native from CharacterBody2D, hence an error.
	uses BadTrait2 # Trait has 'some_func()' but function shadows member of trait TraitA already used in class, hence an error. (trait cohesion mismatch)

	var some_func_4 # Can not shadow trait members.

	# Can not override trait function with different _parameter type.
	func some_func_2(_param: Node2D): # in case _param type as Node or its Node class Parent classes are accepted(i.e Object).
		print("overridden some func")

	# Can not override trait function with different _parameter size.
	func some_func_3(_param1):
		print("overridden some func")

	func _ready():
		var instance = TraitA.new() # Can not instance trait.
		TraitA.some_func() # Can not access trait members directly.
