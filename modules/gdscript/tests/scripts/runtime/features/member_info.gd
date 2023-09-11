class_name TestMemberInfo

class MyClass:
	pass

enum MyEnum {}

static var test_static_var_untyped
static var test_static_var_weak_null = null
static var test_static_var_weak_int = 1
static var test_static_var_hard_int: int

var test_var_untyped
var test_var_weak_null = null
var test_var_weak_int = 1
@export var test_var_weak_int_exported = 1
var test_var_weak_variant_type = TYPE_NIL
@export var test_var_weak_variant_type_exported = TYPE_NIL
var test_var_hard_variant: Variant
var test_var_hard_int: int
var test_var_hard_variant_type: Variant.Type
@export var test_var_hard_variant_type_exported: Variant.Type
var test_var_hard_node_process_mode: Node.ProcessMode
var test_var_hard_my_enum: MyEnum
var test_var_hard_array: Array
var test_var_hard_array_int: Array[int]
var test_var_hard_array_variant_type: Array[Variant.Type]
var test_var_hard_array_node_process_mode: Array[Node.ProcessMode]
var test_var_hard_array_my_enum: Array[MyEnum]
var test_var_hard_array_resource: Array[Resource]
var test_var_hard_array_this: Array[TestMemberInfo]
var test_var_hard_array_my_class: Array[MyClass]
var test_var_hard_resource: Resource
var test_var_hard_this: TestMemberInfo
var test_var_hard_my_class: MyClass

static func test_static_func(): pass

func test_func_implicit_void(): pass
func test_func_explicit_void() -> void: pass
func test_func_weak_null(): return null
func test_func_weak_int(): return 1
func test_func_hard_variant() -> Variant: return null
func test_func_hard_int() -> int: return 1
func test_func_args_1(_a: int, _b: Array[int], _c: int = 1, _d = 2): pass
func test_func_args_2(_a = 1, _b = _a, _c = [2], _d = 3): pass

signal test_signal_1()
signal test_signal_2(a: Variant, b)
signal test_signal_3(a: int, b: Array[int])
signal test_signal_4(a: Variant.Type, b: Array[Variant.Type])
signal test_signal_5(a: MyEnum, b: Array[MyEnum])
signal test_signal_6(a: Resource, b: Array[Resource])
signal test_signal_7(a: TestMemberInfo, b: Array[TestMemberInfo])
signal test_signal_8(a: MyClass, b: Array[MyClass])

func test():
	var script: Script = get_script()
	for property in script.get_property_list():
		if str(property.name).begins_with("test_"):
			if not (property.usage & PROPERTY_USAGE_SCRIPT_VARIABLE):
				print("Error: Missing `PROPERTY_USAGE_SCRIPT_VARIABLE` flag.")
			print("static var ", property.name, ": ", get_type(property))
	for property in get_property_list():
		if str(property.name).begins_with("test_"):
			if not (property.usage & PROPERTY_USAGE_SCRIPT_VARIABLE):
				print("Error: Missing `PROPERTY_USAGE_SCRIPT_VARIABLE` flag.")
			print("var ", property.name, ": ", get_type(property))
	for method in get_method_list():
		if str(method.name).begins_with("test_"):
			print(get_signature(method))
	for method in get_signal_list():
		if str(method.name).begins_with("test_"):
			print(get_signature(method, true))

func get_type(property: Dictionary, is_return: bool = false) -> String:
	match property.type:
		TYPE_NIL:
			if property.usage & PROPERTY_USAGE_NIL_IS_VARIANT:
				return "Variant"
			return "void" if is_return else "null"
		TYPE_BOOL:
			return "bool"
		TYPE_INT:
			if property.usage & PROPERTY_USAGE_CLASS_IS_ENUM:
				return property.class_name
			return "int"
		TYPE_STRING:
			return "String"
		TYPE_DICTIONARY:
			return "Dictionary"
		TYPE_ARRAY:
			if property.hint == PROPERTY_HINT_ARRAY_TYPE:
				return "Array[%s]" % property.hint_string
			return "Array"
		TYPE_OBJECT:
			if not str(property.class_name).is_empty():
				return property.class_name
			return "Object"
	return "<error>"

func get_signature(method: Dictionary, is_signal: bool = false) -> String:
	var result: String = ""
	if method.flags & METHOD_FLAG_STATIC:
		result += "static "
	result += ("signal " if is_signal else "func ") + method.name + "("

	var args: Array[Dictionary] = method.args
	var default_args: Array = method.default_args
	var mandatory_argc: int = args.size() - default_args.size()
	for i in args.size():
		if i > 0:
			result += ", "
		var arg: Dictionary = args[i]
		result += arg.name + ": " + get_type(arg)
		if i >= mandatory_argc:
			result += " = " + var_to_str(default_args[i - mandatory_argc])

	result += ")"
	if is_signal:
		if get_type(method.return, true) != "void":
			print("Error: Signal return type must be `void`.")
	else:
		result += " -> " + get_type(method.return, true)
	return result
