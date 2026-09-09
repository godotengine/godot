func foo(x):
	match x:
		{"key1": "value1", "key2": "value2"}:
			print('{"key1": "value1", "key2": "value2"}')
		{"key1": "value1", "key2"}:
			print('{"key1": "value1", "key2"}')
		{"key1", "key2": "value2"}:
			print('{"key1", "key2": "value2"}')
		{"key1", "key2"}:
			print('{"key1", "key2"}')
		{"key1": "value1"}:
			print('{"key1": "value1"}')
		{"key1"}:
			print('{"key1"}')
		_:
			print("wildcard")

func bar(x):
	match x:
		{0}:
			print("0")
		{1}:
			print("1")
		{2}:
			print("2")
		_:
			print("wildcard")

func baz(x):
	match x:
		{
			"key1": "value1"
		}:
			print('multiline {"key1": "value1"}')
		{
			"key2": "value2",
		}:
			print('multiline {"key2": "value2",}')
		{
			"key3": {
				"key1",
				..,
			},
		}:
			print('multiline {"key3": {"key1", ..,},}')

func test():
	foo({"key1": "value1", "key2": "value2"})
	foo({"key1": "value1", "key2": ""})
	foo({"key1": "", "key2": "value2"})
	foo({"key1": "", "key2": ""})
	foo({"key1": "value1"})
	foo({"key1": ""})
	foo({"key1": "value1", "key2": "value2", "key3": "value3"})
	foo({"key1": "value1", "key3": ""})
	foo({"key2": "value2"})
	foo({"key3": ""})
	bar({0: "0"})
	bar({1: "1"})
	bar({2: "2"})
	bar({3: "3"})
	baz({"key1": "value1"})
	baz({"key2": "value2"})
	baz({"key3": {"key1": "value1", "key2": "value2"}})
