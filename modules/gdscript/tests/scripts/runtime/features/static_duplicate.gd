const PreloadClass = preload("static_duplicate_preload.notest.gd")
const PreloadClassAlias = PreloadClass

func test():
	var dup_preload_one = PreloadClass.duplicate()
	print(dup_preload_one == Vector2.ONE)

	var dup_preload_two = (PreloadClass as GDScript).duplicate()
	print(dup_preload_two is GDScript)

	var dup_preload_alias_one = PreloadClassAlias.duplicate()
	print(dup_preload_alias_one == Vector2.ONE)

	var dup_preload_alias_two = (PreloadClassAlias as GDScript).duplicate()
	print(dup_preload_alias_two is GDScript)

	var PreloadClassAsGDScript = PreloadClass as GDScript
	var dup_preload_class_as_gdscript_one = PreloadClassAsGDScript.duplicate()
	print(dup_preload_class_as_gdscript_one is GDScript)
