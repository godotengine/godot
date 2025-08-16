const preloaded: GDScript = preload("gdscript_to_preload.notest.gd")

func test():
	var preloaded_instance: preloaded = preloaded.new()
	print(preloaded_instance.something())
