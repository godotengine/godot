# We could use @export_custom to really test every property usage, but we know for good
# that duplicating scripted properties flows through the same code already thoroughly tested
# in the [Resource] test cases. The same goes for all the potential deep duplicate modes.
# Therefore, it's enough to ensure the exported scriped properties are copied when invoking
# duplication by each entry point.
class TestResource:
	extends Resource
	@export var text: String = "holaaa"
	@export var arr: Array = [1, 2, 3]
	@export var dict: Dictionary = { "a": 1, "b": 2 }

func test():
	# Via Resource type.
	var res := TestResource.new()
	var dupe: TestResource

	dupe = res.duplicate()
	print(dupe.text)
	print(dupe.arr)
	print(dupe.dict)

	dupe = res.duplicate_deep()
	print(dupe.text)
	print(dupe.arr)
	print(dupe.dict)

	# Via Variant type.

	var res_var = TestResource.new()
	var dupe_var

	dupe_var = res_var.duplicate()
	print(dupe_var.text)
	print(dupe_var.arr)
	print(dupe_var.dict)

	dupe_var = res_var.duplicate_deep()
	print(dupe_var.text)
	print(dupe_var.arr)
	print(dupe_var.dict)
