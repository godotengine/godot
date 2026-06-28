func print_value(value: Variant) -> void:
	if value is Object:
		@warning_ignore("unsafe_method_access")
		print("<%s>" % value.get_class())
	else:
		print(var_to_str(value))

func test():
	var int_value := 1
	print_value(int_value as Variant)
	print_value(int_value as int)
	print_value(int_value as float)

	var node_value := Node.new()
	print_value(node_value as Variant)
	print_value(node_value as Object)
	print_value(node_value as Node)
	print_value(node_value as Node2D)
	node_value.free()

	var null_value = null
	print_value(null_value as Variant)
	@warning_ignore("unsafe_cast")
	print_value(null_value as Node)
