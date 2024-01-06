extends UnitTest
var lua: LuaAPI

func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9980

	lua = LuaAPI.new()

	# testName and testDescription are for any needed context about the test.
	testName = "LuaAPI.pull_variant()"
	testDescription = "
Runs the fibonacci sequence of 15 and verifies the result
"

func fail():
	status = false
	done = true

func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)

	var err = lua.do_string("
	function Fib(n)
	  local function inner(m)
		if m < 2 then
		  return m
		end
		return inner(m-1) + inner(m-2)
	  end
	  return inner(n)
	end

	result = Fib(15)
	")

	if err is LuaError:
		errors.append(err)
		return fail()

	var result = lua.pull_variant("result")
	if result is LuaError:
		errors.append(err)
		return fail()

	if not result is float:
		errors.append(LuaError.new_error("Result is not float but is '%d'" % typeof(result), LuaError.ERR_TYPE))
		return fail()

	if not result == 610:
		errors.append(LuaError.new_error("Result is not 610 but is '%d'" % result, LuaError.ERR_TYPE))
		return fail()

	# Once done is set to true, the test's _process function will no longer be called.
	done = true
