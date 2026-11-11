var _prop: int

# Inline setter assigns `String` to `int`.
# Inline getter returns `int` instead of `String`.
var prop_1: String:
	set(value):
		_prop = value
	get:
		return _prop

# Setter function has wrong argument type.
# Getter function has wrong return type.
var prop_2: String:
	set = set_prop_int, get = get_prop_int

# Inline setter parameter uses property type.
var prop_3 := 0:
	set(value):
		var x: String = value
		prop_3 = value

# Untyped/variant is fine at compile time.
var prop_4: String:
	set = set_prop_untyped, get = get_prop_untyped

var prop_5: String:
	set = set_prop_variant, get = get_prop_variant

# Mismatch in getter/setter type for untyped var.
var prop6:
	set = set_prop_int, get = get_prop_string

var prop7: Variant:
	set = set_prop_int, get = get_prop_string

var prop8:
	set = set_prop_untyped, get = get_prop_int

var prop9:
	set = set_prop_int, get = get_prop_variant


func set_prop_int(value: int):
	_prop = value

func get_prop_int() -> int:
	return _prop

func get_prop_string() -> String:
	return ""

func set_prop_untyped(value):
	_prop = value

func get_prop_untyped():
	return _prop

func set_prop_variant(value: Variant):
	_prop = value

func get_prop_variant() -> Variant:
	return _prop

func test():
	pass
