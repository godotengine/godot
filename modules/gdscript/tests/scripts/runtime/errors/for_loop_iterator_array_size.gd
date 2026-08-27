func iterate(v: Variant):
	for i in v:
		print(i)

class BadInit:
	# Whether to push or pop during init
	var push

	func _init(new_push):
		push = new_push

	func _iter_init(iter: Array):
		if push:
			iter.push_back("hi")
		else:
			iter.pop_back()
		return true

	func _iter_next(iter: Array):
		iter.pop_back()
		return !iter.is_empty()

	func _iter_get(iter):
		return iter

func subtest_init_array_large():
	print("SUBTEST_INIT_ARRAY_LARGE")
	for i in BadInit.new(true):
		print(i)

func subtest_init_array_empty():
	print("SUBTEST_INIT_ARRAY_EMPTY")
	for i in BadInit.new(false):
		print(i)

func subtest_variant_init_array_large():
	print("SUBTEST_VARIANT_INIT_ARRAY_LARGE")
	iterate(BadInit.new(true))

func subtest_variant_init_array_empty():
	print("SUBTEST_VARIANT_INIT_ARRAY_EMPTY")
	iterate(BadInit.new(false))

class BadNext:
	var push: bool

	func _init(new_push):
		push = new_push

	func _iter_init(iter: Array):
		iter[0] = 1
		return true

	func _iter_next(iter: Array):
		if push:
			iter.push_back(2)
		else:
			iter.pop_back()
		return true

	func _iter_get(iter):
		return iter

func subtest_next_array_large():
	print("SUBTEST_NEXT_ARRAY_LARGE")
	for i in BadNext.new(true):
		print(i)

func subtest_next_array_empty():
	print("SUBTEST_NEXT_ARRAY_EMPTY")
	for i in BadNext.new(false):
		print(i)

func subtest_variant_next_array_large():
	print("SUBTEST_VARIANT_NEXT_ARRAY_LARGE")
	iterate(BadNext.new(true))

func subtest_variant_next_array_empty():
	print("SUBTEST_VARIANT_NEXT_ARRAY_EMPTY")
	iterate(BadNext.new(false))


func test():
	# Typed
	subtest_init_array_large()
	subtest_init_array_empty()
	subtest_next_array_large()
	subtest_next_array_empty()

	# Untyped
	subtest_variant_init_array_large()
	subtest_variant_init_array_empty()
	subtest_variant_next_array_large()
	subtest_variant_next_array_empty()
