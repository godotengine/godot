@warning_ignore("narrowing_conversion")
var prop_weak_getter: int: get = weak_getter

@warning_ignore("narrowing_conversion")
var prop_hard_getter: int: get = hard_getter

@warning_ignore("narrowing_conversion")
static var static_prop_weak_getter: int: get = static_weak_getter

@warning_ignore("narrowing_conversion")
static var static_prop_hard_getter: int: get = static_hard_getter

func weak_getter():
	return 1.0

func hard_getter() -> float:
	return 1.0

static func static_weak_getter():
	return 1.0

static func static_hard_getter() -> float:
	return 1.0

static func static_return_float():
	return 1.0

func test():
	var int_var: int
	print(type_string(typeof(prop_weak_getter)))
	print(type_string(typeof(prop_hard_getter)))
	print(type_string(typeof(static_prop_weak_getter)))
	print(type_string(typeof(static_prop_hard_getter)))
	int_var = prop_weak_getter
	print(type_string(typeof(int_var)))
	int_var = prop_hard_getter
	print(type_string(typeof(int_var)))
	int_var = static_prop_weak_getter
	print(type_string(typeof(int_var)))
	int_var = static_prop_hard_getter
	print(type_string(typeof(int_var)))
