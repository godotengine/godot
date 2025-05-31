func test():
	print("warn")

@warning_ignore("shadowed_variable_base_class")
func get(_property: StringName) -> Variant:
	return null
