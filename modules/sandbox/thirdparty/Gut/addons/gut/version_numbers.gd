# ##############################################################################
#
# ##############################################################################
class VerNumTools:

	static func _make_version_array_from_string(v):
		var parts = Array(v.split('.'))
		for i in range(parts.size()):
			var int_val = parts[i].to_int()
			if(str(int_val) == parts[i]):
				parts[i] = parts[i].to_int()
		return parts


	static func make_version_array(v):
		var to_return = []
		if(typeof(v) == TYPE_STRING):
			to_return = _make_version_array_from_string(v)
		elif(typeof(v) == TYPE_DICTIONARY):
			return [v.major, v.minor, v.patch]
		elif(typeof(v) == TYPE_ARRAY):
			to_return = v
		return to_return


	static func make_version_string(version_parts):
		var to_return = 'x.x.x'
		if(typeof(version_parts) == TYPE_ARRAY):
			to_return =  ".".join(version_parts)
		elif(typeof(version_parts) == TYPE_DICTIONARY):
			to_return = str(version_parts.major,  '.',  version_parts.minor,  '.',  version_parts.patch)
		elif(typeof(version_parts) == TYPE_STRING):
			to_return = version_parts
		return to_return


	static func is_version_gte(version, required):
		var is_ok = null
		var v = make_version_array(version)
		var r = make_version_array(required)

		var idx = 0
		while(is_ok == null and idx < v.size() and idx < r.size()):
			if(v[idx] > r[idx]):
				is_ok = true
			elif(v[idx] < r[idx]):
				is_ok = false

			idx += 1

		# still null means each index was the same.
		return GutUtils.nvl(is_ok, true)


	static func is_version_eq(version, expected):
		var version_array = make_version_array(version)
		var expected_array = make_version_array(expected)

		if(expected_array.size() > version_array.size()):
			return false

		var is_version = true
		var i = 0
		while(i < expected_array.size() and i < version_array.size() and is_version):
			if(expected_array[i] == version_array[i]):
				i += 1
			else:
				is_version = false

		return is_version


	static func is_godot_version_eq(expected):
		return VerNumTools.is_version_eq(Engine.get_version_info(), expected)


	static func is_godot_version_gte(expected):
		return VerNumTools.is_version_gte(Engine.get_version_info(), expected)




# ##############################################################################
#
# ##############################################################################
var gut_version = '0.0.0'
var required_godot_version = '0.0.0'

func _init(gut_v = gut_version, required_godot_v = required_godot_version):
	gut_version = gut_v
	required_godot_version = required_godot_v


# ------------------------------------------------------------------------------
# Blurb of text with GUT and Godot versions.
# ------------------------------------------------------------------------------
func get_version_text():
	var v_info = Engine.get_version_info()
	var gut_version_info =  str('GUT version:  ', gut_version)
	var godot_version_info  = str('Godot version:  ', v_info.major,  '.',  v_info.minor,  '.',  v_info.patch)
	return godot_version_info + "\n" + gut_version_info


# ------------------------------------------------------------------------------
# Returns a nice string for erroring out when we have a bad Godot version.
# ------------------------------------------------------------------------------
func get_bad_version_text():
	var info = Engine.get_version_info()
	var gd_version = str(info.major, '.', info.minor, '.', info.patch)
	return 'GUT ' + gut_version + ' requires Godot ' + required_godot_version + \
		' or greater.  Godot version is ' + gd_version


# ------------------------------------------------------------------------------
# Checks the Godot version against required_godot_version.
# ------------------------------------------------------------------------------
func is_godot_version_valid():
	return VerNumTools.is_version_gte(Engine.get_version_info(), required_godot_version)


func make_godot_version_string():
	return VerNumTools.make_version_string(Engine.get_version_info())
