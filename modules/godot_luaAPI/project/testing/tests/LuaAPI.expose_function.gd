extends UnitTest
var lua: LuaAPI

func testFuncTuple(arg1: String, tuple: LuaTuple):
	if not arg1 == "Hello World!":
		errors.append(LuaError.new_error("arg1 is not 'Hello World!' but is '%s'" % arg1))
		return fail()

	if not tuple.size()==2:
		errors.append(LuaError.new_error("tuple.size() is not 2 but is %d" % tuple.size()))
		return fail()

	var val1 = tuple.pop_front()
	if val1:
		errors.append(LuaError.new_error("val1 is true"))
		return fail()

	var val2 = tuple.pop_front()
	if not val2 == 5:
		errors.append(LuaError.new_error("val2 is not 5 but is %d" % val2))
		return fail()
	return true

func testFuncRef(ref: LuaAPI, arg1: String):
	if not arg1 == "Hello World!":
		errors.append(LuaError.new_error("arg1 is not 'Hello World!' but is '%s'" % arg1))
		return fail()

	if not ref.get_meta("isValid"):
		errors.append(LuaError.new_error("ref meta isValid is false or has no value"))
		return fail()
	return true

func testNormal(a, b): return a+b

func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9940

	lua = LuaAPI.new()
	lua.set_meta("isValid", true)
	var err = lua.push_variant("test1", LuaCallableExtra.with_tuple(testFuncTuple, 2))
	if err is LuaError:
		errors.append(err)
		fail()

	err = lua.push_variant("test2", LuaCallableExtra.with_ref(testFuncRef))
	if err is LuaError:
		errors.append(err)
		fail()

	err = lua.push_variant("test3", testNormal)
	if err is LuaError:
		errors.append(err)
		fail()

	# testName and testDescription are for any needed context about the test.
	testName = "LuaAPI.expose_function"
	testDescription = "
Tests exposeing functions to GDScript. Including
- wants ref
- is tuple
- normal
"

func fail():
	status = false
	done = true

func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)

	var err = lua.do_string("
	result1 = test1('Hello World!', false, 5)
	result2 = test2('Hello World!')
	result3 = test3(5, 5)
	")
	if err is LuaError:
		errors.append(err)
		return fail()

	var result1 = lua.pull_variant("result1")
	if not result1:
		errors.append(LuaError.new_error("result1 is false"))
		fail()

	var result2 = lua.pull_variant("result2")
	if not result2:
		errors.append(LuaError.new_error("result2 is false"))
		fail()

	var result3 = lua.pull_variant("result3")
	if not result3 == 10:
		errors.append(LuaError.new_error("result3 is not 10 but is %d" % result3))
		fail()

	done = true
