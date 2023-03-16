func test():
	return_call()
	return_nothing()
	return_side_effect()
	var r = return_side_effect.call() # Untyped call to check return value.
	prints(r, typeof(r) == TYPE_NIL)
	print("end")

func side_effect(v):
	print("effect")
	return v

func return_call() -> void:
	return print("hello")

func return_nothing() -> void:
	return

func return_side_effect() -> void:
	return side_effect("x")
