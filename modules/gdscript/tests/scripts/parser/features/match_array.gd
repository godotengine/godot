func foo(x):
	match x:
		["value1"]:
			print('["value1"]')
		["value1", "value2"]:
			print('["value1", "value2"]')

func bar(x):
	match x:
		[
			"value1"
		]:
			print('multiline ["value1"]')
		[
			"value1",
			"value2",
		]:
			print('multiline ["value1", "value2",]')
		[
			"value1",
			[
				"value2",
				..,
			],
		]:
			print('multiline ["value1", ["value2", ..,],]')

func test():
	foo(["value1"])
	foo(["value1", "value2"])
	bar(["value1"])
	bar(["value1", "value2"])
	bar(["value1", ["value2", "value3"]])
