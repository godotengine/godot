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

func bench_packed(n: int) -> int:
	var arr := PackedInt64Array()
	for i in n:
		arr.push_back(i)
	var s := 0
	for i in n:
		s = (s + arr[i]) & 0x3FFFFFFF
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
