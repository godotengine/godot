extends UnitTest
var lua: LuaAPI

func _ready():
	# Since we are using poly here, we need to make sure to call super for _methods
	super._ready()
	# id will determine the load order
	id = 9825

	lua = LuaAPI.new()

	# testName and testDescription are for any needed context about the test.
	testName = "General.base_types"
	testDescription = "
Test base godot types expose to lua.
"

func fail():
	status = false
	done = true

func _process(delta):
	# Since we are using poly here, we need to make sure to call super for _methods
	super._process(delta)

	var err = lua.do_string("
	vec2 = Vector2(1.5, 1.0)
	vec2Floor = vec2.floor()

	vec3 = Vector3(1.5, 1.0, 1.0)
	vec3Floor = vec3.floor()
	")
	if err is LuaError:
		errors.append(err)
		return fail()

	var vec2 = lua.pull_variant("vec2")
	if vec2 is LuaError:
		errors.append(vec2)
		return fail()

	if not vec2 is Vector2:
		errors.append(LuaError.new_error("vec2 is not type Vector2 but is %d" % typeof(vec2)))
		return fail()

	if vec2.x < 1.4 and vec2.x > 1.6:
		errors.append(LuaError.new_error("vec2 is not (1.5, 1.0) but is (%f, %f)" % [vec2.x, vec2.y]))
		return fail()

	var vec2Floor = lua.pull_variant("vec2Floor")
	if vec2Floor is LuaError:
		errors.append(vec2Floor)
		return fail()

	if not vec2Floor is Vector2:
		errors.append(LuaError.new_error("vec2Floor is not type Vector2 but is %d" % typeof(vec2Floor)))
		return fail()

	if vec2Floor.x < 0.9 and vec2Floor.x > 1.1:
		errors.append(LuaError.new_error("vec2Floor is not (1.0, 1.0) but is (%f, %f)" % [vec2Floor.x, vec2Floor.y]))
		return fail()


	var vec3 = lua.pull_variant("vec3")
	if vec3 is LuaError:
		errors.append(vec3)
		return fail()

	if not vec3 is Vector3:
		errors.append(LuaError.new_error("vec3 is not type Vector3 but is %d" % typeof(vec3)))
		return fail()

	if vec3.x < 1.4 and vec3.x > 1.6:
		errors.append(LuaError.new_error("vec3 is not (1.5, 1.0, 1.0) but is (%f, %f, %f)" % [vec3.x, vec3.y, vec3.z]))
		return fail()

	var vec3Floor = lua.pull_variant("vec3Floor")
	if vec3Floor is LuaError:
		errors.append(vec3Floor)
		return fail()

	if not vec3Floor is Vector3:
		errors.append(LuaError.new_error("vec2Floor is not type Vector3 but is %d" % typeof(vec3Floor)))
		return fail()

	if vec3Floor.x < 0.9 and vec3Floor.x > 1.1:
		errors.append(LuaError.new_error("vec3Floor is not (1.0, 1.0, 1.0) but is (%f, %f, %f)" % [vec3Floor.x, vec3Floor.y, vec3Floor.z]))
		return fail()

	done = true
