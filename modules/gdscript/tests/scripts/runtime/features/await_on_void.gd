func wait() -> void:
	pass

func test():
	@warning_ignore("redundant_await")
	await wait()
	print("end")
