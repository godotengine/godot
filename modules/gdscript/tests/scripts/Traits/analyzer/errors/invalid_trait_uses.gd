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
