extends Node

# 三语言性能基准驱动器。
#
# 设计要点：所有被测量的工作都在目标语言脚本内部完成，驱动器只做「一次调用 + 计时」，
# 因此跨语言调用开销被摊薄到可忽略。三种语言共用同一份算法与迭代次数，用同一个引擎
# 计时器（Time.get_ticks_usec）测量，消除构建与环境差异。
#
# 输出格式（stdout，可被外部脚本解析）：
#   BENCH|<workload>|<lang>|<iters>|<min_us>|<median_us>|<max_us>|<checksum>

const REPS := 7
const DISPATCH_ITERS := 200000

# 工作负载 → 迭代次数。
const WORKLOADS := {
	"bench_arith": 20000000,
	"bench_calls": 5000000,
	"bench_class": 5000000,
	"bench_array": 100000,
	"bench_dictionary": 100000,
	"bench_packed": 100000,
	"bench_array_methods": 100000,
	"bench_string": 8000,
	"bench_vector": 1000000,
	"bench_color": 1000000,
	"bench_engine": 200000,
	"bench_hoist": 200000,
	# 方案 B 基线/验收负载：实现前 AS 使用现有 Godot 容器，实现后切换到强类型容器。
	"bench_typed_i64": 100000,
	"bench_typed_vec2": 100000,
	"bench_typed_map": 100000,
	"bench_batch_sum": 100000,
}

var _targets := {}
# BENCH_SMALL=1 时把负载压到最小，用于从外部墙钟测「引擎启动 + 项目/脚本加载」的固定开销。
var _small := false

func _ready() -> void:
	_small = OS.get_environment("BENCH_SMALL") == "1"
	_targets = {"as": $BenchAS, "cs": $BenchCS, "gd": $BenchGD}
	var version: Dictionary = Engine.get_version_info()
	print("BENCH|engine|%s|0|0|0|0|0" % version["string"])
	for w in WORKLOADS:
		for lang in _targets:
			_run_workload(w, _targets[lang], lang)
	for lang in _targets:
		_run_dispatch(_targets[lang], lang)
	get_tree().quit()

func _stats(times: Array) -> Array:
	times.sort()
	var n := times.size()
	return [times[0], times[n / 2], times[n - 1]]

func _run_workload(name: String, node: Node, lang: String) -> void:
	var iters: int = 1 if _small else WORKLOADS[name]
	var reps: int = 1 if _small else REPS
	# 预热：让 C# 的 JIT 与各运行时的内部缓存进入稳态。
	node.call(name, maxi(1, iters / 100))
	var times: Array = []
	var checksum: Variant = null
	for r in reps:
		var t0 := Time.get_ticks_usec()
		checksum = node.call(name, iters)
		times.append(Time.get_ticks_usec() - t0)
	var st := _stats(times)
	print("BENCH|%s|%s|%d|%d|%d|%d|%s" % [name, lang, iters, st[0], st[1], st[2], str(checksum)])

func _run_dispatch(node: Node, lang: String) -> void:
	# 跨边界方法派发：驱动器经 Object.call 以最小载荷调用目标脚本方法 N 次。
	# 三种目标都经同一调用路径，差异反映的是各语言脚本实例的派发与编组开销。
	var iters: int = 1 if _small else DISPATCH_ITERS
	var reps: int = 1 if _small else REPS
	var times: Array = []
	for r in reps:
		var t0 := Time.get_ticks_usec()
		for i in iters:
			node.call("bench_noop")
		times.append(Time.get_ticks_usec() - t0)
	var st := _stats(times)
	print("BENCH|bench_dispatch|%s|%d|%d|%d|%d|0" % [lang, iters, st[0], st[1], st[2]])
