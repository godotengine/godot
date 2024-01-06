extends Node

var lua: LuaAPI = LuaAPI.new()

func _lua_print(message: String):
	print(message)

func _ready():
	# All builtin libraries are available to bind with. Use Debug, OS and IO at your own risk.
	lua.bind_libraries(["base", "table", "string"])

	lua.push_variant("print", _lua_print)
	lua.push_variant("message", "Hello lua!")

	# Most methods return a LuaError in case of an error
	var err: LuaError = lua.do_string("""
	for i=1,10,1 do
		print(message)
	end
	function get_message()
		return "Hello gdScript!"
	end
	""")
	if err is LuaError:
		print("ERROR %d: %s" % [err.type, err.message])
		return
	
	var ret = lua.call_function("get_message", [])
	if ret is LuaError:
		print("ERROR %d: %s" % [ret.type, ret.message])
		return
	print(ret)
