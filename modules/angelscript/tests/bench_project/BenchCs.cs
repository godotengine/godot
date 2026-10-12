using Godot;

// C# 实现。方法名刻意与 GDScript / AngelScript 保持一致（下划线小写），
// 以便驱动器用同一个名字经 Object.Call 派发；这偏离常规 C# 命名规范，是为基准公平性做的取舍。
public partial class BenchCs : Node
{
	public long bench_noop()
	{
		return 1;
	}

	public long bench_arith(long n)
	{
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			long t = i & 0x7FFF;
			s = (s + t * t) & 0x3FFFFFFF;
		}
		return s;
	}

	private long Add(long a, long b)
	{
		return a + b;
	}

	public long bench_calls(long n)
	{
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			s = Add(s, i) & 0x3FFFFFFF;
		}
		return s;
	}

	private sealed class BenchItem
	{
		public long A;
		public long B;
	}

	public long bench_class(long n)
	{
		var items = new BenchItem[4];
		for (int i = 0; i < 4; i++)
		{
			items[i] = new BenchItem { A = i + 1, B = (i + 1) * 2 };
		}
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			var it = items[(int)(i & 3)];
			it.A = (it.A + it.B + 1) & 0x3FFFFFFF;
			s = (s + it.A) & 0x3FFFFFFF;
		}
		return s;
	}

	public long bench_array(long n)
	{
		var arr = new Godot.Collections.Array();
		for (long i = 0; i < n; i++)
		{
			arr.Add(i);
		}
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			s = (s + (long)arr[(int)i]) & 0x3FFFFFFF;
		}
		return s;
	}

	public long bench_dictionary(long n)
	{
		var dict = new Godot.Collections.Dictionary { ["value"] = 7L };
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			s = (s + (long)dict["value"]) & 0x3FFFFFFF;
		}
		return s;
	}

	public long bench_packed(long n)
	{
		var arr = new long[n];
		for (long i = 0; i < n; i++)
		{
			arr[i] = i;
		}
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			s = (s + arr[i]) & 0x3FFFFFFF;
		}
		return s;
	}

	public long bench_array_methods(long n)
	{
		var arr = new Godot.Collections.Array();
		for (long i = 0; i < 16; i++)
		{
			arr.Add(i * 7);
		}
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			int idx = (int)(i % 16);
			s = (s + arr.Count + (arr.Contains(idx * 7) ? 1 : 0) + (long)arr[idx]) & 0x3FFFFFFF;
		}
		return s;
	}

	public long bench_string(long n)
	{
		string s = "";
		for (long i = 0; i < n; i++)
		{
			s += "x";
		}
		return s.Length;
	}

	public long bench_vector(long n)
	{
		var v = new Vector2(1.5f, 2.5f);
		var acc = Vector2.Zero;
		for (long i = 0; i < n; i++)
		{
			acc = acc + v;
		}
		return (long)(acc.X + acc.Y);
	}

	public long bench_color(long n)
	{
		var c = new Color(1.0f, 2.0f, 3.0f, 1.0f);
		var acc = new Color();
		for (long i = 0; i < n; i++)
		{
			acc = acc + c;
		}
		return (long)acc.R & 0x3FFFFFFF;
	}

	public long bench_engine(long n)
	{
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			s = (s + GetChildCount()) & 0x3FFFFFFF;
		}
		return s;
	}

	public long bench_hoist(long n)
	{
		// 用法层批量化：把跨边界调用提到循环外，循环内只做本地算术。
		long c = GetChildCount();
		long s = 0;
		for (long i = 0; i < n; i++)
		{
			s = (s + c) & 0x3FFFFFFF;
		}
		return s;
	}

	public long bench_typed_i64(long n)
	{
		var arr = new long[n];
		for (long i = 0; i < n; i++) arr[i] = i;
		long s = 0;
		for (long i = 0; i < n; i++) s = (s + arr[i]) & 0x3FFFFFFF;
		return s;
	}

	public long bench_typed_vec2(long n)
	{
		var dst = new Vector2[n];
		var src = new Vector2[n];
		for (int i = 0; i < n; i++) { dst[i] = new Vector2(1, 2); src[i] = new Vector2(0.5f, 1); }
		for (int i = 0; i < n; i++) dst[i] += src[i] * 2.0f;
		return n > 0 ? (long)(dst[n - 1].X + dst[n - 1].Y) : 0;
	}

	public long bench_typed_map(long n)
	{
		var dict = new System.Collections.Generic.Dictionary<StringName, long>();
		var key = new StringName("value");
		for (long i = 0; i < n; i++) dict[key] = i;
		long s = 0;
		for (long i = 0; i < n; i++) { dict.TryGetValue(key, out long value); s = (s + value) & 0x3FFFFFFF; }
		return s;
	}

	public long bench_batch_sum(long n)
	{
		var arr = new long[n];
		for (long i = 0; i < n; i++) arr[i] = i;
		long s = 0;
		foreach (long value in arr) s = (s + value) & 0x3FFFFFFF;
		return s;
	}
}
