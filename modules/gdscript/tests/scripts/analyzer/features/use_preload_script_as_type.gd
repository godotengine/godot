const preloaded : GDScript = preload("gdscript_to_preload.gd")

func test():
	var preloaded_instance: preloaded = preloaded.new()
	print(preloaded_instance.something())
