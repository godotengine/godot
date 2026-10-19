# We don't want to execute it because of errors, just analyze.
func no_exec_test():
	var weak_int = 1
	print(weak_int as Variant) # No warning.
	print(weak_int as int)
	print(weak_int as Node)

	var weak_node = Node.new()
	print(weak_node as Variant) # No warning.
	print(weak_node as int)
	print(weak_node as Node)

	var weak_variant = null
	print(weak_variant as Variant) # No warning.
	print(weak_variant as int)
	print(weak_variant as Node)

	var hard_variant: Variant = null
	print(hard_variant as Variant) # No warning.
	print(hard_variant as int)
	print(hard_variant as Node)

	var array: Array = []
	print(array as Variant) # No warning.
	print(array as Array) # No warning.
	print(array as Array[RID])

	var rid_array: Array[RID] = []
	print(rid_array as Variant) # No warning.
	print(rid_array as Array) # No warning.
	print(rid_array as Array[RID]) # No warning.

	var dict: Dictionary = {}
	print(dict as Variant) # No warning.
	print(dict as Dictionary) # No warning.
	print(dict as Dictionary[RID, RID])

	var rid_dict: Dictionary[RID, RID] = {}
	print(rid_dict as Variant) # No warning.
	print(rid_dict as Dictionary) # No warning.
	print(rid_dict as Dictionary[RID, RID]) # No warning.

func test():
	pass
