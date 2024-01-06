extends UnitTest
var lua: LuaAPI

func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9970

	lua = LuaAPI.new()

	# testName and testDescription are for any needed context about the test.
	testName = "LuaAPI.call_function"
	testDescription = "
Tests both luaAPI.call_Function and pulling the function as a Callable.
"

func fail():
	status = false
	done = true

func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)

	var err = lua.do_string("
	function test(a)
		return a+5
	end
	")
	if err is LuaError:
		errors.append(err)
		return fail()

	if not lua.function_exists("test"):
		errors.append(LuaError.new_error('lua.function_exists("test") returned false'))
		return fail()

	var ret = lua.call_function("test", [5])
	if ret is LuaError:
		errors.append(ret)
		return fail()

	if not ret == 10:
		errors.append(LuaError.new_error("ret is not 10 but is '%d'" % ret))
		return fail()

	var testCallable = lua.pull_variant("test")
	if testCallable is LuaError:
		errors.append(testCallable)
		return fail()

	if not testCallable is Callable:
		errors.append(LuaError.new_error("testCallable is not Callable but is '%d'" % typeof(testCallable), LuaError.ERR_TYPE))
		return fail()

	var cret = testCallable.call(5)
	if cret is LuaError:
		errors.append(cret)
		return fail()

	if not cret is float:
		errors.append(LuaError.new_error("cret is not float but is '%d'" % typeof(cret), LuaError.ERR_TYPE))
		return fail()

	if not cret == 10:
		errors.append(LuaError.new_error("cret is not 10 but is '%d'" % cret))
		return fail()

	done = true
