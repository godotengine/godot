# FIXME: warning_ignore doesn't work here.
#@warning_ignore("narrowing_conversion")
#const INT_CONST: int = 1.0

const VECTOR2I_CONST: Vector2i = Vector2.ONE

# FIXME: warning_ignore doesn't work here.
#@warning_ignore("narrowing_conversion")
#var int_var: int = 1.0

var vector2i_var: Vector2i = Vector2.ONE

# FIXME: warning_ignore doesn't work here.
#@warning_ignore("narrowing_conversion")
#func int_param_func(x: int = 1.0):
#	return x

func vector2i_param_func(x: Vector2i = Vector2.ONE):
	return x

func test():
	@warning_ignore("narrowing_conversion")
	const LOCAL_INT_CONST: int = 1.0

	@warning_ignore("narrowing_conversion")
	var local_int_var: int = 1.0

	#print(typeof(INT_CONST) == TYPE_INT)
	#print(typeof(int_var) == TYPE_INT)
	print(typeof(LOCAL_INT_CONST) == TYPE_INT)
	print(typeof(local_int_var) == TYPE_INT)
	#print(typeof(int_param_func()) == TYPE_INT)

	const LOCAL_VECTOR2I_CONST: Vector2i = Vector2.ONE

	var local_vector2i_var: Vector2i = Vector2.ONE

	print(typeof(VECTOR2I_CONST) == TYPE_VECTOR2I)
	print(typeof(vector2i_var) == TYPE_VECTOR2I)
	print(typeof(LOCAL_VECTOR2I_CONST) == TYPE_VECTOR2I)
	print(typeof(local_vector2i_var) == TYPE_VECTOR2I)
	print(typeof(vector2i_param_func()) == TYPE_VECTOR2I)
