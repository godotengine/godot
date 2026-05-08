extends SceneTree

func _init():
    # get classes
	var classes = ClassDB.get_class_list()
	# sort classes
	classes.sort()
    # stores output
	var output = []
	# focuses on Engine get singletons
	var singletons = Engine.get_singleton_list()

    # for each class
	for cls in classes:
	    # store class name as string
		var cls_name := String(cls)
		# store its methods
		var methods = ClassDB.class_get_method_list(cls_name, false)
		# store the singletons
		var is_singleton = singletons.has(cls_name)

        # In the loop, for methods
		for method in methods:
		    # stores the data structures
			var ret = method.get("return", {})
			# create the entry
			var entry = {
				"class": cls_name,
				"name": String(method.get("name", "")),
				"return_type": int(ret.get("type", 0)),
				"args": [],
				"static": false,
				"singleton": is_singleton
			}

			for arg in method.get("args", []):
				entry["args"].append({
					"name": String(arg.get("name", "")),
					"type": int(arg.get("type", 0))
				})

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
