signal s1()
signal s2()
signal s3()
@warning_ignore("unused_signal")
signal s4()

func no_exec():
	s1.emit()
	print(s2)

func test():
	pass
