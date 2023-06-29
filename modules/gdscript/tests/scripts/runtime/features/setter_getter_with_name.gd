class_name TestSetterGetterWithName

static var static_normal_untyped:      set = set_static_normal_untyped, get = get_static_normal_untyped
static var static_normal_typed:   int: set = set_static_normal_typed,   get = get_static_normal_typed
static var static_named_untyped:       set = set_static_named_untyped,  get = get_static_named_untyped
static var static_named_typed:    int: set = set_static_named_typed,    get = get_static_named_typed

var normal_untyped:      set = set_normal_untyped, get = get_normal_untyped
var normal_typed:   int: set = set_normal_typed,   get = get_normal_typed
var named_untyped:       set = set_named_untyped,  get = get_named_untyped
var named_typed:    int: set = set_named_typed,    get = get_named_typed

static var _data := {
	static_normal_untyped = 0,
	static_normal_typed   = 0,
	static_named_untyped  = 0,
	static_named_typed    = 0,

	normal_untyped = 0,
	normal_typed   = 0,
	named_untyped  = 0,
	named_typed    = 0,
}

func check(expected: int, only_static: bool = false) -> void:
	for key in _data:
		if only_static and not str(key).begins_with("static_"):
			continue
		if _data[key] != expected:
			prints("Check", expected, "is NOT correct:", key, "==", _data[key])
			return
	prints("Check", expected, "is correct.")

func test():
	var _t

	_t = static_normal_untyped
	_t = static_normal_typed
	_t = static_named_untyped
	_t = static_named_typed
	_t = normal_untyped
	_t = normal_typed
	_t = named_untyped
	_t = named_typed
	check(1)

	_t = get("static_normal_untyped")
	_t = get("static_normal_typed")
	_t = get("static_named_untyped")
	_t = get("static_named_typed")
	_t = get("normal_untyped")
	_t = get("normal_typed")
	_t = get("named_untyped")
	_t = get("named_typed")
	check(2)

	_t = TestSetterGetterWithName.static_normal_untyped
	_t = TestSetterGetterWithName.static_normal_typed
	_t = TestSetterGetterWithName.static_named_untyped
	_t = TestSetterGetterWithName.static_named_typed
	check(3, true)

	_t = (TestSetterGetterWithName as GDScript).get("static_normal_untyped")
	_t = (TestSetterGetterWithName as GDScript).get("static_normal_typed")
	_t = (TestSetterGetterWithName as GDScript).get("static_named_untyped")
	_t = (TestSetterGetterWithName as GDScript).get("static_named_typed")
	check(4, true)

	static_normal_untyped = 10
	static_normal_typed = 10
	static_named_untyped = 10
	static_named_typed = 10
	normal_untyped = 10
	normal_typed = 10
	named_untyped = 10
	named_typed = 10
	check(10)

	set("static_normal_untyped", 20)
	set("static_normal_typed", 20)
	set("static_named_untyped", 20)
	set("static_named_typed", 20)
	set("normal_untyped", 20)
	set("normal_typed", 20)
	set("named_untyped", 20)
	set("named_typed", 20)
	check(20)

	TestSetterGetterWithName.static_normal_untyped = 30
	TestSetterGetterWithName.static_normal_typed = 30
	TestSetterGetterWithName.static_named_untyped = 30
	TestSetterGetterWithName.static_named_typed = 30
	check(30, true)

	(TestSetterGetterWithName as GDScript).set("static_normal_untyped", 40)
	(TestSetterGetterWithName as GDScript).set("static_normal_typed", 40)
	(TestSetterGetterWithName as GDScript).set("static_named_untyped", 40)
	(TestSetterGetterWithName as GDScript).set("static_named_typed", 40)
	check(40, true)

static func set_static_normal_untyped(value, optional = true):
	assert(is_same(optional, true))
	_data.static_normal_untyped = value

static func get_static_normal_untyped(optional = true):
	assert(is_same(optional, true))
	_data.static_normal_untyped += 1
	return 0

static func set_static_normal_typed(value: int, optional: bool = true) -> void:
	assert(is_same(optional, true))
	_data.static_normal_typed = value

static func get_static_normal_typed(optional: bool = true) -> int:
	assert(is_same(optional, true))
	_data.static_normal_typed += 1
	return 0

static func set_static_named_untyped(name, value, optional = true):
	assert(is_same(optional, true))
	_data[name] = value

static func get_static_named_untyped(name, optional = true):
	assert(is_same(optional, true))
	_data[name] += 1
	return 0

static func set_static_named_typed(name: StringName, value: int, optional: bool = true) -> void:
	assert(is_same(optional, true))
	_data[name] = value

static func get_static_named_typed(name: StringName, optional: bool = true) -> int:
	assert(is_same(optional, true))
	_data[name] += 1
	return 0

func set_normal_untyped(value, optional = true):
	assert(is_same(optional, true))
	_data.normal_untyped = value

func get_normal_untyped(optional = true):
	assert(is_same(optional, true))
	_data.normal_untyped += 1
	return 0

func set_normal_typed(value: int, optional: bool = true) -> void:
	assert(is_same(optional, true))
	_data.normal_typed = value

func get_normal_typed(optional: bool = true) -> int:
	assert(is_same(optional, true))
	_data.normal_typed += 1
	return 0

func set_named_untyped(name, value, optional = true):
	assert(is_same(optional, true))
	_data[name] = value

func get_named_untyped(name, optional = true):
	assert(is_same(optional, true))
	_data[name] += 1
	return 0

func set_named_typed(name: StringName, value: int, optional: bool = true) -> void:
	assert(is_same(optional, true))
	_data[name] = value

func get_named_typed(name: StringName, optional: bool = true) -> int:
	assert(is_same(optional, true))
	_data[name] += 1
	return 0
