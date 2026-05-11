extends SceneTree

func _init() -> void:
	# This Array will be printed to stdout, as the result
	var output: Array[Dictionary] = []

	var singletons: PackedStringArray = Engine.get_singleton_list()

	# cls_name is chosen instead of class_name, because class_name is a keyword
	var classes: PackedStringArray = ClassDB.get_class_list()

	for cls_name: String in classes:
		var is_singleton: bool = singletons.has(cls_name)
		var methods: Array[Dictionary] = ClassDB.class_get_method_list(cls_name, false)

		for method: Dictionary in methods:
			# Get the arguments
			var method_args: Array[Dictionary] = []
			for arg: Dictionary in method.get("args", []):
				method_args.push_back({
					"name": String(arg.get("name", "")), # String() is kept for readability
					"type": int(arg.get("type", 0)) # int() is kept for readability
				})

			# Get the return data
			var method_return: Dictionary = method.get("return", {})
			var method_return_type: int = int(method_return.get("type", 0))

			# Create the entry
			var entry: Dictionary = {
				"class": cls_name,
				"name": String(method.get("name", "")), # String() is kept for readability
				"return_type": int(method_return_type), # int() is kept for readability
				"args": method_args,
				"static": false,
				"singleton": is_singleton
			}
			output.append(entry)

		# Synthetic static accessor for singleton classes.
		if is_singleton:
			output.append({
				"class": cls_name,
				"name": "get_singleton",
				"return_type": 24, # Object/ObjectID path in your generator
				"args": [],
				"static": true,
				"singleton": true
			})

	print(JSON.stringify(output, "\t"))
	quit()
