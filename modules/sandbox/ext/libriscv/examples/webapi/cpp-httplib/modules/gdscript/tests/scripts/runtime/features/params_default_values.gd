# https://github.com/godotengine/godot/issues/56702

func test():
	const_default()
	func_result_default()
	# calling again will run the initializer again,
	# as the default is not evaluated at time of defining the function (as in python)
	# but every time the function is called (as in C++)
	func_result_default()
	lots_of_defaults("non-optional")
	# somewhat obscure feature: referencing earlier parameters
	ref_default("non-optional", 42)


func const_default(param=42):
	print(param)


var default_val := 0

func get_default():
	default_val += 1
	return default_val


func func_result_default(param=get_default()):
	print(param)


func lots_of_defaults(nondefault, one=1, two=2, three=get_default()):
	prints(nondefault, one, two, three)


func ref_default(nondefault1, nondefault2, defa=nondefault1, defb=nondefault2 - 1):
	prints(nondefault1, nondefault2, defa, defb)
