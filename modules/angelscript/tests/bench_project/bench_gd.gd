extends Node

# GDScript 实现。所有 bench_* 方法返回 30 位以内的整数校验和，三种语言必须完全一致。

func bench_noop() -> int:
	return 1

func bench_arith(n: int) -> int:
	var s := 0
	for i in n:
		var t := i & 0x7FFF
		s = (s + t * t) & 0x3FFFFFFF
	return s

func _add(a: int, b: int) -> int:
	return a + b

func bench_calls(n: int) -> int:
	var s := 0
	for i in n:
		s = _add(s, i) & 0x3FFFFFFF
	return s

class BenchItem:
	var a := 0
	var b := 0

func bench_class(n: int) -> int:
	var items := [BenchItem.new(), BenchItem.new(), BenchItem.new(), BenchItem.new()]
	for i in 4:
		items[i].a = i + 1
		items[i].b = (i + 1) * 2
	var s := 0
	for i in n:
		var it: BenchItem = items[i & 3]
		it.a = (it.a + it.b + 1) & 0x3FFFFFFF
		s = (s + it.a) & 0x3FFFFFFF
	return s

func bench_array(n: int) -> int:
	var arr := []
	for i in n:
		arr.push_back(i)
	var s := 0
	for i in n:
		s = (s + arr[i]) & 0x3FFFFFFF
	return s

func bench_dictionary(n: int) -> int:
	var dict := {"value": 7}
	var s := 0
	for i in n:
		s = (s + int(dict["value"])) & 0x3FFFFFFF
	return s

func bench_packed(n: int) -> int:
	var arr := PackedInt64Array()
	for i in n:
		arr.push_back(i)
	var s := 0
	for i in n:
		s = (s + arr[i]) & 0x3FFFFFFF
	return s

func bench_array_methods(n: int) -> int:
	# A2：Array.size/has 经原生内建方法 thunk；索引读走数组原生索引。
	var arr := []
	for i in 16:
		arr.push_back(i * 7)
	var s := 0
	for i in n:
		s = (s + arr.size() + (1 if arr.has(i % 16 * 7) else 0) + int(arr[i % 16])) & 0x3FFFFFFF
	return s

func bench_string(n: int) -> int:
	var s := ""
	for i in n:
		s += "x"
	return s.length()

func bench_vector(n: int) -> int:
	var v := Vector2(1.5, 2.5)
	var acc := Vector2.ZERO
	for i in n:
		acc = acc + v
	return int(acc.x + acc.y)

func bench_color(n: int) -> int:
	var c := Color(1.0, 2.0, 3.0, 1.0)
	var acc := Color()
	for i in n:
		acc = acc + c
	return int(acc.r) & 0x3FFFFFFF

func bench_engine(n: int) -> int:
	var s := 0
	for i in n:
		s = (s + get_child_count()) & 0x3FFFFFFF
	return s

func bench_hoist(n: int) -> int:
	# 用法层批量化：把跨边界调用提到循环外，循环内只做本地算术。
	var c := get_child_count()
	var s := 0
	for i in n:
		s = (s + c) & 0x3FFFFFFF
	return s

func bench_typed_i64(n: int) -> int:
	var arr := PackedInt64Array()
	for i in n:
		arr.push_back(i)
	var s := 0
	for i in n:
		s = (s + arr[i]) & 0x3FFFFFFF
	return s

func bench_typed_vec2(n: int) -> int:
	var dst := PackedVector2Array()
	var src := PackedVector2Array()
	dst.resize(n)
	src.resize(n)
	for i in n:
		dst[i] = Vector2(1.0, 2.0)
		src[i] = Vector2(0.5, 1.0)
	for i in n:
		dst[i] = dst[i] + src[i] * 2.0
	var last := dst[n - 1] if n > 0 else Vector2.ZERO
	return int(last.x + last.y)

func bench_typed_map(n: int) -> int:
	var dict: Dictionary[StringName, int] = {}
	var key := &"value"
	for i in n:
		dict[key] = i
	var s := 0
	for i in n:
		s = (s + dict.get(key, 0)) & 0x3FFFFFFF
	return s

func bench_batch_sum(n: int) -> int:
	var arr := PackedInt64Array()
	arr.resize(n)
	for i in n:
		arr[i] = i
	var s := 0
	for value in arr:
		s = (s + value) & 0x3FFFFFFF
	return s
