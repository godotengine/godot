@warning_ignore_start("return_value_discarded")

func test():
	const PATH = "../../utils.notest.gd"

	preload(PATH)
	preload(PATH,)
	preload(
		PATH
	)
	preload(
		PATH,
	)

	print("OK")
