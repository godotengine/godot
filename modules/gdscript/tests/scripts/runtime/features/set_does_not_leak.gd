# https://github.com/godotengine/godot/issues/90086

class MyObj:
	var obj : WeakRef

func test():
	var obj_1 = MyObj.new()
	var obj_2 = MyObj.new()
	assert(obj_2.get_reference_count() == 1)
	obj_1.set(&"obj", obj_2)
	assert(obj_2.get_reference_count() == 1)
