class_name SetterConversionTest

var prop_weak_setter: int:
	set = weak_setter

var prop_hard_setter: int:
	set = hard_setter

static var static_prop_weak_setter: int:
	set = static_weak_setter

var dynamic_props: Dictionary[String, int] = {}

func weak_setter(value):
	prop_weak_setter = value
	prints(var_to_str(value), var_to_str(prop_weak_setter))

func hard_setter(value: float):
	@warning_ignore("narrowing_conversion")
	prop_hard_setter = value
	prints(var_to_str(value), var_to_str(prop_hard_setter))

static func static_weak_setter(value):
	static_prop_weak_setter = value
	prints(var_to_str(value), var_to_str(static_prop_weak_setter))

func _set(property: StringName, value: Variant):
	if property == "d_prop":
		dynamic_props["d_prop"] = value
		prints(var_to_str(value), var_to_str(dynamic_props["d_prop"]))
		return true
	return false

func test() -> void:
	var t: Variant = 1.0
	prop_weak_setter = t
	set("prop_weak_setter", t)
	prop_hard_setter = t
	set("prop_hard_setter", t)
	SetterConversionTest.static_prop_weak_setter = t
	set("static_prop_weak_setter", t)
	set("d_prop", t)
