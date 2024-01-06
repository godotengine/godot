extends UnitTest
var lua: LuaAPI
var testObj: TestObject
class TestObject:
	func __index(ref: LuaAPI, index: String):
		if index=="test1":
			return ref.get_meta("isValid")
		elif index=="test2":
			return 5


func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9850

	lua = LuaAPI.new()
	lua.set_meta("isValid", true)
	testObj = TestObject.new()
	lua.push_variant("testObj", testObj)

	# testName and testDescription are for any needed context about the test.
	testName = "General.object_metamethod"
	testDescription = "
Tests objects overriding lua metamethods.
"

func fail():
	status = false
	done = true

func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)

	var err = lua.do_string("
	result1 = testObj.test1
	result2 = testObj.test2
	")
	if err is LuaError:
		errors.append(err)
		return fail()

	var result1 = lua.pull_variant("result1")
	if result1 is LuaError:
		errors.append(err)
		return fail()

	if not result1:
		errors.append(LuaError.new_error("ref meta isValid is false or has no value"))
		return fail()

	var result2 = lua.pull_variant("result2")
	if result2 is LuaError:
		errors.append(err)
		return fail()

	if not result2 == 5:
		errors.append(LuaError.new_error("result2 is not 5 but is %d" % result2))
		return fail()

	done = true
