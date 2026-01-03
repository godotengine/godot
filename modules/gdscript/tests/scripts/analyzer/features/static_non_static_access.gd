@static_unload

static var static_var
var non_static_var

signal my_signal()

static func static_func():
	pass

func non_static_func():
	pass

static var test_static_var_lambda = func ():
	static_func()
	print(static_func)
	static_var = 1
	print(static_var)

var test_non_static_var_lambda = func ():
	static_func()
	print(static_func)
	static_var = 1
	print(static_var)

	non_static_func()
	print(non_static_func)
	non_static_var = 1
	print(non_static_var)
	my_signal.emit()
	print(my_signal)

static var test_static_var_setter:
	set(_value):
		static_func()
		print(static_func)
		static_var = 1
		print(static_var)

var test_non_static_var_setter:
	set(_value):
		static_func()
		print(static_func)
		static_var = 1
		print(static_var)

		non_static_func()
		print(non_static_func)
		non_static_var = 1
		print(non_static_var)
		my_signal.emit()
		print(my_signal)

static func test_static_func():
	static_func()
	print(static_func)
	static_var = 1
	print(static_var)

func test_non_static_func():
	static_func()
	print(static_func)
	static_var = 1
	print(static_var)

	non_static_func()
	print(non_static_func)
	non_static_var = 1
	print(non_static_var)
	my_signal.emit()
	print(my_signal)

func test():
	test_static_var_lambda = null
	test_non_static_var_lambda = null
