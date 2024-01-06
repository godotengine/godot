extends UnitTest
var lua: LuaAPI
var co: LuaCoroutine

func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9500

	lua = LuaAPI.new()
	co = lua.new_coroutine()

	co.load_string("
	a = 0
	for i=1,10,1 do
		-- yield is exposed to Lua when the thread is bound.
		yield(1)
		a = a + i
	end
	")

	# testName and testDescription are for any needed context about the test.
	testName = "LuaCoroutine.resume"
	testDescription = "
Runs a lua coroutine yielding for 1 second. It will run 10 times.
a = 0
for each resume
	a = a + i
a should be 55
"

func fail():
	status = false
	done = true

var yieldTime = 0
var timeSince = 0
func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)

	timeSince += delta
	if timeSince <= yieldTime:
		return

	var ret = co.resume([])
	if ret is LuaError:
		errors.append(ret)
		return fail()

	if co.is_done():
		var a = co.pull_variant("a")
		if a is LuaError:
			errors.append(a)
			return fail()

		if not a == 55:
			errors.append(LuaError.new_error("a is not 55 but is '%d'" % a))
			return fail()

		if time < 10 or time > 10.1:
			errors.append(LuaError.new_error("time is not within 10 and 10.1 but is '%s'" % str(time)))
			return fail()

		done = true
		return

	if not ret is Array:
		errors.append(LuaError.new_error("Result is not Array but is '%d'" % typeof(ret), LuaError.ERR_TYPE))
		return fail()
	if not ret.size() == 1:
		errors.append(LuaError.new_error("Result.size() is not 1 but is '%d'" % ret.size()))
		return fail()

	yieldTime = ret[0]

	if not yieldTime == 1:
		errors.append(LuaError.new_error("yieldTime is not 1 but is '%d'" % yieldTime))

	timeSince = 0
