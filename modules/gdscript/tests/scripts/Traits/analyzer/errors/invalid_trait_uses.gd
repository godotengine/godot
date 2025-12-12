trait TraitA extends Node:
	pass

class ClassA:
	pass

class ClassB extends Object:
	uses TraitA # Using a trait that extends Node but the class extends Object.
    # This should raise an error because inheritance mismatch.
    # in this case, Node or children classes of Node would be accepted as using class inheritance.

class ClassC:
	uses ClassA # Class cannot be used as a trait. Should raise an error.

class Actor extends Intermediate:
	uses Interactable # compatible since both are children of BaseCharacter
	# However, "Intermediate" & "Interactable" conflict over func interact() raising error

class Intermediate extends BaseCharacter:
	func interact():
		print("Interactable")

class BaseCharacter extends Node:
	pass

trait Interactable extends BaseCharacter:
	func interact():
		print("Interactable")
