extends UnitTest
var lua: LuaAPI

class TestObject:
	var a: String

var testObj: TestObject

func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9800

	testObj = TestObject.new()
	lua = LuaAPI.new()
	var err = lua.push_variant("testObj", testObj)
	if err is LuaError:
		errors.append(err)
		fail()

	# testName and testDescription are for any needed context about the test.
	testName = "General.object_push"
	testDescription = "
This is to test Objects being passed to lua as userdata.
The test object has one variable 'a'
Lua will modify a, and we will confirm the change on GD's side
"

func fail():
	status = false
	done = true

func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)

	var err = lua.do_string("testObj.a = 'Hello from lua!'")
	if err is LuaError:
		errors.append(err)
		return fail()

	if not testObj.a == "Hello from lua!":
		errors.append(LuaError.new_error("testObj.a is not 'Hello from lua!' but is '%s'" % testObj.a))
		return fail()

	done = true
