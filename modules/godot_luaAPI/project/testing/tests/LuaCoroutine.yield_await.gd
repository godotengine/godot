extends UnitTest
var lua: LuaAPI
var co: LuaCoroutine

func _test_yield_await()->int:
	await co.yield_await([get_tree().create_timer(3).timeout])
	return 4

func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9490

	lua = LuaAPI.new()
	co = lua.new_coroutine()
	co.push_variant("test_yield_await", _test_yield_await)

	co.load_string("
	a = 0
	for i=1,10,1 do
		-- yield is exposed to Lua when the thread is bound.
		yield(1)
		a = a + i
	end
	b = test_yield_await()
	")

	# testName and testDescription are for any needed context about the test.
	testName = "LuaCoroutine.yield_await"
	testDescription = "
Runs a lua thread yielding for 1 second. It will run 10 times.
a = 0
for each resume
	a = a + i
a should be 55
After which it calls _test_yield_await and sets b to its return.
b should be 4
"

func fail():
	status = false
	done = true

var yieldTime = 0
var timeSince = 0
var proc = true
func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)
	if not proc:
		return

	timeSince += delta
	if timeSince <= yieldTime:
		return

	if co.is_done():
		var a = co.pull_variant("a")
		if a is LuaError:
			errors.append(a)
			return fail()

		if not a == 55:
			errors.append(LuaError.new_error("a is not 55 but is '%d'" % a))
			return fail()

		var b = co.pull_variant("b")
		if b is LuaError:
			errors.append(b)
			return fail()

		if not b == 4:
			errors.append(LuaError.new_error("b is not 4 but is '%d'" % b))
			return fail()

		if time < 13 or time > 13.1:
			errors.append(LuaError.new_error("time is not within 13 and 13.1 but is '%s'" % str(time)))
			return fail()

		done = true
		return

	var ret = co.resume([])
	if ret is LuaError:
		errors.append(ret)
		return fail()

	if not ret is Array:
		errors.append(LuaError.new_error("Result is not Array but is '%d'" % typeof(ret), LuaError.ERR_TYPE))
		return fail()
	if  ret.size() == 0:
		return
	if not ret.size() == 1:
		errors.append(LuaError.new_error("Result.size() is not 1 but is '%d'" % ret.size()))
		return fail()

	if ret[0] is Signal:
		proc = false
		await ret[0]
		proc = true
		return

	yieldTime = ret[0]

	if not yieldTime == 1:
		errors.append(LuaError.new_error("yieldTime is not 1 but is '%d'" % yieldTime))

	timeSince = 0
