signal test_signal

func coroutine():
	await test_signal
	return 1

func test():
	var _result = self.call("coroutine")
