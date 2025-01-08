@static_unload
class_name StaticVariablesOther

static var perm := 0

static var prop := "Hello!":
	get: return prop + " suffix"
	set(value): prop = "prefix " + str(value)

func test():
	print("ok")
