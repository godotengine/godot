@static_unload

static var perm := 0

static var prop := "Hello!":
	get: return prop + " suffix"
	set(value): prop = "prefix " + str(value)

static func get_data():
	return "data"

static var data = get_data()

class Inner:
	static var prop := "inner"
	static func _static_init() -> void:
		prints("Inner._static_init", prop)

	class InnerInner:
		static var prop := "inner inner"
		static func _static_init() -> void:
			prints("InnerInner._static_init", prop)

func test():
	prints("data:", data)

	prints("perm:", perm)
	prints("prop:", prop)

	perm = 1
	prop = "World!"

	prints("perm:", perm)
	prints("prop:", prop)

	prints("other.perm:", StaticVariablesOther.perm)
	prints("other.prop:", StaticVariablesOther.prop)

	StaticVariablesOther.perm = 2
	StaticVariablesOther.prop = "foo"

	prints("other.perm:", StaticVariablesOther.perm)
	prints("other.prop:", StaticVariablesOther.prop)

	@warning_ignore("unsafe_method_access")
	var path = get_script().get_path().get_base_dir()
	@warning_ignore("unsafe_call_argument")
	var other = load(path + "/static_variables_load.gd")

	prints("load.perm:", other.perm)
	prints("load.prop:", other.prop)

	other.perm = 3
	other.prop = "bar"

	prints("load.perm:", other.perm)
	prints("load.prop:", other.prop)
