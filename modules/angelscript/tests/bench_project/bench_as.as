// godot_base: Node
// AngelScript 实现。方法名与 GDScript / C# 一致，校验和亦须一致。
//
// 句柄语义提醒：AS 里 `h = other`（无 @）对句柄是「对象值拷贝」，句柄重绑定必须写
// `@h = @other`。因此本文件刻意不在循环里重绑定句柄，避免踩这个坑。

class BenchItem {
	int a;
	int b;
}

class bench_as {
	int bench_noop() {
		return 1;
	}

	int bench_arith(int n) {
		int s = 0;
		for (int i = 0; i < n; i++) {
			int t = i & 0x7FFF;
			s = (s + t * t) & 0x3FFFFFFF;
		}
		return s;
	}

	int add(int a, int b) {
		return a + b;
	}

	int bench_calls(int n) {
		int s = 0;
		for (int i = 0; i < n; i++) {
			s = add(s, i) & 0x3FFFFFFF;
		}
		return s;
	}

	int bench_class(int n) {
		BenchItem@ i0 = BenchItem();
		i0.a = 1;
		i0.b = 2;
		BenchItem@ i1 = BenchItem();
		i1.a = 2;
		i1.b = 4;
		BenchItem@ i2 = BenchItem();
		i2.a = 3;
		i2.b = 6;
		BenchItem@ i3 = BenchItem();
		i3.a = 4;
		i3.b = 8;
		int s = 0;
		for (int i = 0; i < n; i++) {
			int k = i & 3;
			if (k == 0) {
				i0.a = (i0.a + i0.b + 1) & 0x3FFFFFFF;
				s = (s + i0.a) & 0x3FFFFFFF;
			} else if (k == 1) {
				i1.a = (i1.a + i1.b + 1) & 0x3FFFFFFF;
				s = (s + i1.a) & 0x3FFFFFFF;
			} else if (k == 2) {
				i2.a = (i2.a + i2.b + 1) & 0x3FFFFFFF;
				s = (s + i2.a) & 0x3FFFFFFF;
			} else {
				i3.a = (i3.a + i3.b + 1) & 0x3FFFFFFF;
				s = (s + i3.a) & 0x3FFFFFFF;
			}
		}
		return s;
	}

	int bench_array(int n) {
		Array arr;
		for (int i = 0; i < n; i++) {
			arr.push_back(i);
		}
		int s = 0;
		for (int i = 0; i < n; i++) {
			s = (s + int(arr[i])) & 0x3FFFFFFF;
		}
		return s;
	}

	int bench_dictionary(int n) {
		Dictionary dict;
		dict[Variant("value")] = Variant(7);
		int s = 0;
		for (int i = 0; i < n; i++) {
			s = (s + int(dict[Variant("value")])) & 0x3FFFFFFF;
		}
		return s;
	}

	int bench_packed(int n) {
		// Packed 标量数组：索引读写走原生 thunk（P2）。
		PackedInt64Array arr;
		for (int i = 0; i < n; i++) {
			arr.push_back(i);
		}
		int s = 0;
		for (int i = 0; i < n; i++) {
			s = (s + int(arr[i])) & 0x3FFFFFFF;
		}
		return s;
	}

	int bench_array_methods(int n) {
		// A2：Array 的 size/has/push_back 走原生内建方法 thunk，索引读走数组原生索引。
		Array arr;
		for (int i = 0; i < 16; i++) {
			arr.push_back(Variant(i * 7));
		}
		int s = 0;
		for (int i = 0; i < n; i++) {
			s = (s + int(arr.size()) + (arr.has(Variant((i % 16) * 7)) ? 1 : 0) + int(arr[i % 16])) & 0x3FFFFFFF;
		}
		return s;
	}

	int bench_string(int n) {
		// AS 原生 string 只注册了赋值与 length()，没有拼接；字符串运算只能用 Godot String（以 Variant 存储）。
		String s = String();
		for (int i = 0; i < n; i++) {
			s = s + "x";
		}
		return int(s.length());
	}

	int bench_vector(int n) {
		Vector2 v(1.5, 2.5);
		Vector2 acc;
		for (int i = 0; i < n; i++) {
			acc = acc + v;
		}
		return int(acc.x + acc.y);
	}

	int bench_color(int n) {
		// 几何值类型原生存储（P3b）：Color 的构造/拷贝/析构走原生布局；运算符仍 generic。
		Color c(1.0, 2.0, 3.0, 1.0);
		Color acc;
		for (int i = 0; i < n; i++) {
			acc = acc + c;
		}
		return int(acc.r) & 0x3FFFFFFF;
	}

	int bench_engine(int n) {
		// as_self() 返回 Object@，本模块不支持向下转换到 Node@，只能自建一个 Node 句柄来调用引擎方法。
		Node @selfn = Node();
		int s = 0;
		for (int i = 0; i < n; i++) {
			s = (s + int(selfn.get_child_count(false))) & 0x3FFFFFFF;
		}
		return s;
	}

	int bench_hoist(int n) {
		// 用法层批量化：把跨边界调用提到循环外，循环内只做本地算术。
		// 与 bench_engine 对比可量化「每次跨边界调用」的代价。
		Node @selfn = Node();
		int c = int(selfn.get_child_count(false));
		int s = 0;
		for (int i = 0; i < n; i++) {
			s = (s + c) & 0x3FFFFFFF;
		}
		return s;
	}

	int bench_typed_i64(int n) {
		Int64Array@ arr = Int64Array();
		arr.reserve(n);
		for (int i = 0; i < n; i++) arr.push_back(i);
		int s = 0;
		for (int i = 0; i < n; i++) s = (s + int(arr.get(i))) & 0x3FFFFFFF;
		return s;
	}

	int bench_typed_vec2(int n) {
		Vector2Array@ dst = Vector2Array();
		Vector2Array@ src = Vector2Array();
		dst.resize(n); src.resize(n);
		dst.fill(Vector2(1.0, 2.0)); src.fill(Vector2(0.5, 1.0));
		dst.add_scaled_in_place(src, 2.0);
		Vector2 last;
		if (n > 0) last = dst.get(n - 1);
		return int(last.x + last.y);
	}

	int bench_typed_map(int n) {
		StringNameInt64Map@ dict = StringNameInt64Map();
		StringName key("value");
		for (int i = 0; i < n; i++) dict.set(key, i);
		int s = 0;
		for (int i = 0; i < n; i++) s = (s + int(dict.get_or(key, 0))) & 0x3FFFFFFF;
		return s;
	}

	int bench_batch_sum(int n) {
		Int64Array@ arr = Int64Array();
		arr.reserve(n);
		for (int i = 0; i < n; i++) arr.push_back(i);
		return int(arr.sum()) & 0x3FFFFFFF;
	}
}
