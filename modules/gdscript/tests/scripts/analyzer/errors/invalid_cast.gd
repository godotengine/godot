func test():
	var integer := 1
	print(integer as Array)
	print(integer as Node)
	print(integer as Object)

	var object := RefCounted.new()
	print(object as int)

	var string_array: Array[String] = []
	print(string_array as Array[RID])
	print(string_array as Dictionary[String, Variant])
	print(string_array as Dictionary[String, String])
	print(string_array as Dictionary[Variant, String])

	var packed_string_array: PackedStringArray = []
	print(packed_string_array as PackedVector2Array)
	print(packed_string_array as Array[String])

	var string_string_dict: Dictionary[String, String] = {}
	print(string_string_dict as Dictionary[RID, RID])
	print(string_string_dict as Dictionary[RID, Variant])
	print(string_string_dict as Dictionary[Variant, RID])
	print(string_string_dict as Dictionary[String, Variant])
	print(string_string_dict as Dictionary[Variant, String])

	var string_variant_dict: Dictionary[String, Variant] = {}
	print(string_variant_dict as Dictionary[RID, RID])
	print(string_variant_dict as Dictionary[RID, Variant])
	print(string_variant_dict as Dictionary[Variant, RID])
	print(string_variant_dict as Dictionary[String, String])
	print(string_variant_dict as Dictionary[Variant, String])

	var variant_string_dict: Dictionary[Variant, String] = {}
	print(variant_string_dict as Dictionary[RID, RID])
	print(variant_string_dict as Dictionary[RID, Variant])
	print(variant_string_dict as Dictionary[Variant, RID])
	print(string_variant_dict as Dictionary[String, String])
	print(string_variant_dict as Dictionary[String, Variant])
