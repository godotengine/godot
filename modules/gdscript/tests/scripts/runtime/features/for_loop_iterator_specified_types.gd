func test():
	print("Test range.")
	for e: float in range(2, 5):
		var elem := e
		prints(var_to_str(e), var_to_str(elem))

	print("Test int.")
	for e: float in 3:
		var elem := e
		prints(var_to_str(e), var_to_str(elem))

	print("Test untyped int array.")
	var a1 := [10, 20, 30]
	for e: float in a1:
		var elem := e
		prints(var_to_str(e), var_to_str(elem))

	print("Test typed int array.")
	var a2: Array[int] = [10, 20, 30]
	for e: float in a2:
		var elem := e
		prints(var_to_str(e), var_to_str(elem))

	# GH-82021
	print("Test implicitly typed array literal.")
	for e: float in [100, 200, 300]:
		var elem := e
		prints(var_to_str(e), var_to_str(elem))

	print("Test String-keys dictionary.")
	var d1 := {a = 1, b = 2, c = 3}
	for k: StringName in d1:
		var key := k
		prints(var_to_str(k), var_to_str(key))

	print("Test RefCounted-keys dictionary.")
	var d2 := {RefCounted.new(): 1, Resource.new(): 2, ConfigFile.new(): 3}
	for k: RefCounted in d2:
		var key := k
		prints(k.get_class(), key.get_class())
