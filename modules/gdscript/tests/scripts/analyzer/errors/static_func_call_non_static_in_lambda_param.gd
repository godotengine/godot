# GH-83468

func non_static_func():
	pass

static func static_func(
		f := func ():
			var g := func ():
				non_static_func()
			g.call()
):
	f.call()

func test():
	pass
