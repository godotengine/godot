# Tests custom iterator correctness both typed and untyped.

class ExponentialIterator:
	var base: int
	var end: int

	func _init(new_base: int, new_end: int):
		self.base = new_base
		self.end = new_end

	func _iter_init(iter: Array):
		iter[0] = 1
		return iter[0] < end

	func _iter_next(iter: Array):
		iter[0] *= base
		return iter[0] < end

	func _iter_get(iter: Variant):
		return iter


@warning_ignore_start("unsafe_method_access")
class StringIterator:
	var s: String

	func _init(value: String):
		self.s = value

	func _iter_init(iter: Array):
		iter[0] = s
		return !iter[0].is_empty()

	func _iter_next(iter: Array):
		iter[0] = iter[0].substr(1)
		return !iter[0].is_empty()

	func _iter_get(iter: Variant):
		return iter.left(1)


func iterate_variant(v: Variant, break_condition: Callable = Callable()):
	for i in v:
		if break_condition.is_valid() and break_condition.call(i):
			break
		print(i)


func iterate_variant_nested(a: Variant, b: Variant, break_condition: Callable = Callable()):
	for i in a:
		for j in b:
			if break_condition.is_valid() and break_condition.call(i, j):
				break
			print(i, j)


func test():
	print("Test typed exp 2,16")
	for i in ExponentialIterator.new(2, 17):
		print(i)

	print("Test variant exp 2,16")
	iterate_variant(ExponentialIterator.new(2, 17))

	print("Test typed exp -3,-27")
	for i in ExponentialIterator.new(-3, 30):
		print(i)

	print("Test typed exp -3,-27")
	iterate_variant(ExponentialIterator.new(-3, 30))

	print("Test typed nested same str iterator")
	var si := StringIterator.new("abc")
	for i in si:
		for j in si:
			print(i, j)

	print("Test nested variant same str iterator")
	iterate_variant_nested(si, si)

	print("Test break from typed str")
	for i in StringIterator.new("world"):
		if i in "hello":
			break
		print(i)

	print("Test break from variant str")
	iterate_variant(StringIterator.new("world"), func(i): return i in "hello")

	print("Test break from nested diff typed")
	for i: String in StringIterator.new("a2z"):
		for j in ExponentialIterator.new(2, 100):
			if j > ord(i):
				break
			print(i, j)

	print("Test break from nested diff variant")
	iterate_variant_nested(
		StringIterator.new("a2z"),
		ExponentialIterator.new(2, 100),
		func(i: String, j): return j > ord(i)
	)
