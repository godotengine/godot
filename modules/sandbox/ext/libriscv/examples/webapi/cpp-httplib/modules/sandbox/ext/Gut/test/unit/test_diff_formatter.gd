extends GutTest

class TestFormatter:
	extends GutTest

	var Formatter = load('res://addons/gut/diff_formatter.gd')
	var DiffTool = GutUtils.DiffTool

	func test_demo_eq_format():
		assert_eq([], [])
		assert_eq([1, 2, 3], [1, 2, 3])
		assert_eq({}, {})
		assert_eq({'a':1}, {'a':1})

	func test_equal_arrays():
		pass_test(Formatter.new().make_it(DiffTool.new([1, 2, 3], [1, 2, 3])))

	func test_equal_dictionaries():
		pass_test(Formatter.new().make_it(DiffTool.new({}, {})))


	func test_works_with_strings_and_numbers():
		var a1 = [0, 1, 2, 3, 4]
		var a2 = [0, 'one', 'two', 'three', '4']
		var diff = DiffTool.new(a1, a2)
		pass_test(Formatter.new().make_it(diff))


	func test_complex_real_use_output():
		var d1 = {'a':1, 'dne_in_d2':'asdf', 'b':{'c':88, 'd':22, 'f':{'g':1, 'h':200}}, 'i':[1, 2, 3], 'z':{}}
		var d2 = {'a':1, 'b':{'c':99, 'e':'letter e', 'f':{'g':1, 'h':2}}, 'i':[1, 'two', 3], 'z':{}}
		var diff = DiffTool.new(d1, d2)
		pass_test(Formatter.new().make_it(diff))

	func test_large_dictionary_summary():
		var d1 = {}
		var d2 = {}
		for i in range(3):
			for j in range(65, 91):
				var one_char = PackedByteArray([j]).get_string_from_ascii()
				var key = ''
				for x in range(i + 1):
					key += one_char
				if(key == 'BB'):
					d1[key] = d1.duplicate()
					d2[key] = d2.duplicate()
				else:
					d1[key] = (i + 1) * j
					d2[key] = one_char

		var diff = DiffTool.new(d1, d2)
		var formatter = Formatter.new()
		formatter.set_max_to_display(20)
		pass_test(formatter.make_it(diff))


	func test_mix_of_array_and_dictionaries_deep():
		var a1 = [
			'a', 'b', 'c',
			[1, 2, 3, 4],
			{'a':1, 'b':2, 'c':3},
			[{'a':1}, {'b':2}]
		]
		var a2 = [
			'a', 2, 'c',
			['a', 2, 3, 'd'],
			{'a':11, 'b':12, 'c':13},
			[{'a':'diff'}, {'b':2}]
		]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
		pass_test(Formatter.new().make_it(diff))


	func test_multiple_sub_arrays():
		var a1 = [
			[1, 2, 3],
			[[4, 5, 6], ['same'], [7, 8, 9]]
		]
		var a2 = [
			[11, 12, 13],
			[[14, 15, 16], ['same'], [17, 18, 19]]
		]
		var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
		pass_test(Formatter.new().make_it(diff))

	func test_when_arrays_are_large_then_summarize_truncates():
		var a1 = []
		var a2 = []
		for i in range(100):
			a1.append(i)
			if(i%2 == 0):
				a2.append(str(i))
			else:
				if(i < 90):
					a2.append(i)

		var diff = DiffTool.new(a1, a2)
		var formatter = Formatter.new()
		pass_test(formatter.make_it(diff))


	func test_absolute_max():
		var a1 = []
		var a2 = []
		for i in range(11000):
			a1.append(i)

		var diff = DiffTool.new(a1, a2)
		var formatter = Formatter.new()
		#formatter.set_max_to_display(formatter.UNLIMITED)
		pass_test(formatter.make_it(diff))

	func test_nested_difference():
		var v1 = {'a':{'b':{'c':{'d':1}}}}
		var v2 = {'a':{'b':{'c':{'d':2}}}}
		var diff = DiffTool.new(v1, v2)
		var formatter = Formatter.new()
		pass_test(formatter.make_it(diff))

class TestUsingAssertNe:
	extends GutTest

	var DiffTool = GutUtils.DiffTool

	func test_works_with_strings_and_numbers():
		var a1 = [0, 1, 2, 3, 4]
		var a2 = [0, 'one', 'two', 'three', '4']
		assert_ne(a1, a2)


	func test_mix_of_array_and_dictionaries():
		var a1 = [
			'a', 'b', 'c',
			[1, 2, 3, 4],
			{'a':1, 'b':2, 'c':3},
			[{'a':1}, {'b':2}]
		]
		var a2 = [
			'a', 2, 'c',
			['a', 2, 3, 'd'],
			{'a':11, 'b':12, 'c':13},
			[{'a':'diff'}, {'b':2}]
		]
		assert_ne(a1, a2)


	# func test_multiple_sub_arrays():
	# 	var a1 = [
	# 		[1, 2, 3],
	# 		[[4, 5, 6], ['same'], [7, 8, 9]]
	# 	]
	# 	var a2 = [
	# 		[11, 12, 13],
	# 		[[14, 15, 16], ['same'], [17, 18, 19]]
	# 	]
	# 	var diff = DiffTool.new(a1, a2, GutUtils.DIFF.DEEP)
	# 	pass_test(Formatter.new().make_it(diff))

	# func test_when_arrays_are_large_then_summarize_truncates():
	# 	var a1 = []
	# 	var a2 = []
	# 	for i in range(100):
	# 		a1.append(i)
	# 		if(i%2 == 0):
	# 			a2.append(str(i))
	# 		else:
	# 			if(i < 90):
	# 				a2.append(i)

	# 	var diff = DiffTool.new(a1, a2)
	# 	var formatter = Formatter.new()
	# 	pass_test(formatter.make_it(diff))


	# func test_absolute_max():
	# 	var a1 = []
	# 	var a2 = []
	# 	for i in range(11000):
	# 		a1.append(i)

	# 	var diff = DiffTool.new(a1, a2)
	# 	var formatter = Formatter.new()
	# 	#formatter.set_max_to_display(formatter.UNLIMITED)
	# 	pass_test(formatter.make_it(diff))

	# func test_nested_difference():
	# 	var v1 = {'a':{'b':{'c':{'d':1}}}}
	# 	var v2 = {'a':{'b':{'c':{'d':2}}}}
	# 	var diff = DiffTool.new(v1, v2)
	# 	var formatter = Formatter.new()
	# 	pass_test(formatter.make_it(diff))
