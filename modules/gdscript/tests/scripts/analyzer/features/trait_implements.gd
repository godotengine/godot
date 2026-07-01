# A trait declares a pure contract (bodyless functions). A class promises to
# satisfy it with `implements`, and can then be used statically as that type.
# (The trait is declared as an inner trait here so the whole example fits in one
# file; a top-level `trait Name` instead makes the file itself a named trait.)
class Something extends RefCounted implements SimSystem:
	func get_value() -> int:
		return 42

class OtherSystem extends RefCounted implements SimSystem:
	func get_value() -> int:
		return 7

trait SimSystem:
	func get_value() -> int

func use_system(system: SimSystem) -> int:
	return system.get_value()

func test():
	var something := Something.new()

	# An implementer is compatible with the trait type (Layer A static typing).
	var system: SimSystem = something
	print(system.get_value())

	# Implementers can be passed where the trait type is expected.
	print(use_system(something))
	print(use_system(OtherSystem.new()))
