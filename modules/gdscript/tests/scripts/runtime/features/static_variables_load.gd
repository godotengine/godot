@static_unload

static var perm := 0

static var prop := "Hello!":
	get: return prop + " suffix"
	set(value): prop = "prefix " + str(value)

func test():
	print("ok")
