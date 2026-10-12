# Godot AngelScript 模块

基于 [AngelScript](https://www.angelcode.com/angelscript/) 2.38.0 的脚本语言模块，用于「主包 NativeAOT + 逻辑热更」方案中的热更通道：
游戏逻辑以 `.as` 资源下发，由本模块在运行时编译执行。设计文档见
`docs/superpowers/specs/2026-09-30-godot-angelscript-hot-update-design.md`（当前仅存在于工作区，未随仓库提交）。

## 当前状态（M0–M5）

已可用：

- `ASScriptLanguage`：向 `ScriptServer` 注册 `AngelScript` 语言，扩展名 `as`；引擎懒初始化（`--test`/headless 下不会主动调用 `ScriptServer::init_languages()` 的路径也能工作）。
- `ASEngine`：持有唯一的 `asIScriptEngine`，提供模块编译、函数执行与内建 API 注册；`ensure_initialized()` 时按项目设置构建绑定计划，并按惰性注册策略注册（见下）。
- `ASScript`：`.as`/`.asb` 资源（`Script` 子类），经 `ASResourceFormatLoaderASScript` / `ASResourceFormatSaverASScript` 接入资源系统，`.tscn` 可按路径引用。
- `ASScriptInstance`：脚本实例，支持内建标量属性的读写与方法按名字调用；`_ready`/`_enter_tree`/`_exit_tree`/`_process`/`_physics_process` 经 Node 的 `GDVIRTUAL` 按名字派发，`notification()` 承载 `_notification(int)`。
- 绑定层（M2）：把 ClassDB 内省结果注册给 AS，让热更脚本能直接调用引擎 API。
  - 内建值类型：34 个内建 `Variant` 值类型（`Vector2` / `String` / `Array` / `Dictionary` / `Packed*Array` …）加 `Variant` 自身，统一以 `Variant` 为内存布局，方法/属性/运算符按内省自动注册；AS 原生 `string` 与 Godot `String` 双向转换。
  - 对象类型：`Object` 派生类以引用类型注册，含工厂（可实例化类）、方法、属性（virtual property）、常量、枚举与向上转换（`opImplCast`）；`RefCounted` 走引用计数语义，脚本持有会延长生命周期。
  - `@GlobalScope` 工具函数：签名可表达的工具函数（如 `clampf`）注册为全局函数。
  - 可见性：`angel_script/class_whitelist` / `angel_script/class_blacklist` 项目设置控制绑定范围（空白名单 = 全部放行，`Object` 无条件保留，白名单会回填祖先链）。
  - 惰性注册：`angel_script/binding/startup_mode`（`auto` 默认 / `full` 启动即全量）与 `angel_script/binding/scan_project`（默认 `true`，扫描 `res://` 下 `.as` 源推断所需类型）。`auto` 下先注册 core 与扫描到的基础类型，脚本加载/热更时按需增量注册，遇到无法处理的类型回退全量（`ASBindingLazyRegistry`）。
  - 静态方法：AS 2.38 不认 `static` 关键字，静态方法与内建值类型的静态方法注册为全局函数，名字为 `<声明类/类型>_<成员名>`（如 `Node_print_orphan_nodes`、`Vector2_from_angle`）；dump 的 `global functions` 区与之对应。
  - `--dump-angelscript-api <dir>`：把当前绑定面导出为 `angelscript_api.d.as`（声明，首行为内容哈希）与 `angelscript_unbound.txt`（不可绑定清单）。
  - `--compile-angelscript [<output_dir>]`：扫描 `res://` 下的 `.as` 源文件，逐个离线编译为 `.asb` 字节码容器（默认输出到当前目录）。

尚未提供（后续里程碑）：跨语言通道的对象/字符串回调编组（`_input` 等带参回调仍不派发）、协程（await）、脚本内自定义类注册进 ClassDB。编辑器语言服务（`EditorLanguage`）已实现 `validate()` / `complete_code()` / `find_function()`；`make_function()` 仍为 stub，`lookup_code()` / `format_code()` / 重命名等沿用默认实现。

## 宿主桥（M5 互操作 L2）

AngelScript 脚本可经 `as_host_call(int method_id, const Array &in args) -> Variant` 调用宿主
（业务层）注册的方法，实现 AS ↔ C#（经 NativeAOT）的双向互操作。引擎侧只提供低层机制，
不包含具体业务接口。

- **回调表**：宿主通过 `ASHostBridge::get_singleton()->install(&callbacks)` 安装一张版本化的
  `ASHostCallbacks`。`abi_version` 必须等于 `AS_HOST_BRIDGE_ABI_VERSION`（当前为 `1`）；
  `struct_size` 不得小于本结构大小，否则安装被拒绝。表按**数字 `method_id`** 派发，
  `method_id` 到方法名/签名的映射属业务层。
- **编组**：参数与返回值统一为 `Variant`；只传 Variant 内建类型与 Godot 对象，禁止逐帧
  高频调用，批量数据用 `Array`/`Dictionary`。`RefCounted` 由 `Variant` 续命。
- **反向调用（host → AS）**：复用 Godot 对象模型——宿主持有指向 AS 实例方法的 `Callable`
  （或 `Object.call`）即可回调脚本。
- **`Services` 约定（业务层）**：业务层把 C# 侧能力暴露为一个 autoload 单例 `Services`，
  并将其方法映射到 `as_host_call` 的 `method_id`；引擎不提供 `Services` 对象本身。
- **主线程**：与 AS 引擎一致，所有互操作调用在主线程执行。
- **安装与卸载时机**：`install()` 须在模块 `MODULE_INITIALIZATION_LEVEL_SERVERS` 初始化完成之后
  （例如主程序启动、引擎就绪后）调用；模块反初始化会清空回调表，宿主需在模块重新初始化后重新
  `install()`。`install()` 会**拷贝整张表**，传入的 `ASHostCallbacks` 内存可随即失效；桥不管理
  `user_data`，宿主须保证其指向的对象在 `uninstall()` 之前一直有效。

## 平台支持

模块在所有平台均可构建；AngelScript 的 callfunc 汇编按平台/架构/编译器显式接线：

| 目标 | 汇编实现 |
| --- | --- |
| x86_64 + MSVC | `ml64` 汇编 `as_callfunc_x64_msvc_asm.asm` |
| x86_64 + GCC/Clang/MinGW | 无外部汇编（`as_callfunc_x64_gcc.cpp`）|
| arm64 + Apple | `as_callfunc_arm64_xcode.S` |
| arm64 + 非 Apple | `as_callfunc_arm64_gcc.S` |
| arm32 + Apple | `as_callfunc_arm_xcode.S` |
| arm32 + 非 Apple | `as_callfunc_arm_gcc.S` |
| riscv64 | `as_callfunc_riscv64_gcc.S` |
| 其它（x86_32/wasm32/ppc64/… 及 MSVC arm64）| 定义 `AS_MAX_PORTABILITY`，纯可移植执行 |

平台→汇编的映射与自检见 `asm_sources.py` 与 `tests/asm_sources_test.py`。

绑定层的原生值类型快速路径（`as_binding_native_value_ops.h`，用 C++ thunk 取代 generic 调用桩）只在**已验证平台**默认启用：`x86_64 + MSVC`（`_MSC_VER && _M_X64`）且未定义 `AS_MAX_PORTABILITY`。其它平台与 `AS_MAX_PORTABILITY` 一律走 generic 回退以保证可移植；未验证平台可定义 `AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED` 强行开启（若同时定义 `AS_MAX_PORTABILITY` 会在编译期 `#error` 拒绝）。启用时另有 `sizeof(void *) == 8` 与 `ANGELSCRIPT_VERSION == 23800` 两条 `static_assert` 作 ABI 前置检查。Linux / macOS 等其它平台**尚未验证**，门控保持关闭。
冒烟工程见 `tests/smoke_project/`（`smoke.as` + 入库的 `smoke_bc.asb`，`_ready` 打印 `910001`/`910002`）。

## 脚本约定

- 扩展名：`.as`（源码）与 `.asb`（预编译字节码容器，M4）均可加载。`.asb` 由 `--compile-angelscript` 离线生成：容器携带基类名与所需的 ClassDB 类型符号表，加载前只对这些类型做增量注册（不退回全量注册）；字节码脚本没有 `source_code`，`reload()` 返回 `ERR_INVALID_DATA`，类名同样必须等于文件名，且 `.asb` 必须与生成它的引擎 ABI 一致（版本/签名校验与回退策略由业务层 manifest 负责，本模块不做校验）。
- 加载线程：`.as` 必须在主线程加载（AS 引擎主线程独占，见 `as_engine.h`）；`ResourceLoader.load_threaded_request()` 会失败并打印错误，而不是在后台线程破坏引擎状态。
- 基类指令：源码前 10 行内必须有一行 `// godot_base: <ClassDB 类型名>`，它决定 `ASScript::get_instance_base_type()`（如 `Node`、`Resource`）。缺少或类型不存在都会导致加载失败。扫描感知块注释与字符串字面量：`/* ... */` 内的伪指令、`"..."` 内的文本都不会被当成指令。
- 类名必须等于文件名（不含扩展名），例如 `res://enemy_spawner.as` 里的类必须叫 `enemy_spawner`。本模块不做全局类名注册，脚本一律按路径引用。
- 回调签名必须与 Godot 一致：`void _ready()`、`void _process(double delta)`、`void _physics_process(double delta)`、`void _enter_tree()`、`void _exit_tree()`、`void _notification(int what)`。签名不匹配的回调不会被派发（例如 `void _process()` 视为不存在）。
- 属性：阶段一支持 `bool` / `int` / `int64` / `float` / `double`；方法参数与返回值支持同样的标量类型（`int` 形参按 32 位、`int64` 按 64 位编组）。
- 绑定层类型（M2）：`int`→`int64`、`float`→`double`、`StringName`/`NodePath` 等内建值类型按 `Variant` 存储的类型直接可用；对象类型必须用显式句柄语法（`Node @n = Node();`，本模块关闭了 AS 的隐式句柄），枚举写作 `<声明类>_<枚举名>::<值>`（如 `Node_ProcessMode::PROCESS_MODE_ALWAYS`），常量写作 `<声明类>_<常量名>`（如 `Object_NOTIFICATION_PREDELETE`）。
- 回调形参：需要对象或字符串形参的回调（`_input` / `_shortcut_input` / `_unhandled_input` / `_unhandled_key_input` / `_get_configuration_warnings` 等）暂不会被派发（`_process` 这类标量形参除外），因为 `ASScriptInstance::callp` 的编组目前只支持标量；绑定层已能表达对象与字符串，补全回调通道属后续里程碑。

### 强类型高性能容器

热循环可使用固定引用容器，绕过 `Variant` 元素装箱：

- 数组：`Int64Array`、`Float64Array`、`Vector2Array`、`Vector3Array`、`ColorArray`、`StringNameArray`、`ObjectArray`。
- Map：`Int64Int64Map`、`StringNameInt64Map`、`StringNameFloat64Map`、`StringNameVector2Map`、`StringNameColorMap`、`StringNameObjectMap`。
- 公共数组 API：`size/is_empty/reserve/resize/clear/push_back/get/set/fill/append_array`；数值数组另有 `sum/min_value/max_value/add_scalar/multiply_scalar`，Vector 数组有 `sum/add_in_place/add_scaled_in_place`。
- 数值/Vector/Color 数组可用 `assign_packed` / `to_packed` 在边界一次性转换；`StringNameArray`、`ObjectArray` 可用 `assign_array` / `to_array`。
- `ObjectArray` 与 `StringNameObjectMap` 保存弱 `ObjectID`，不拥有对象；`remove_invalid()` 批量清除 null 或已释放对象。

```angelscript
Int64Array@ values = Int64Array();
values.reserve(1000);
for (int i = 0; i < 1000; i++) values.push_back(i);
int64 total = values.sum(); // 求和循环在 C++ 内执行。

Vector2Array@ positions = Vector2Array();
Vector2Array@ velocities = Vector2Array();
positions.resize(1000); velocities.resize(1000);
positions.add_scaled_in_place(velocities, delta);
```

这些名称表示 AngelScript 模块的原生引用容器，不是 Godot `Packed*Array` 的别名。动态互操作应集中在边界批量转换，不要在内层循环反复转为 `Array` / `Dictionary`。

## 对象句柄语义（M3）

句柄的拥有 / 非拥有语义由**静态类型**决定，在注册时一次判定：

| 静态类型 | 槽内容 | 语义 |
| --- | --- | --- |
| 派生自 `RefCounted`（`Resource@`、`RefCounted@` …） | 裸 `Object*` | **拥有**：AS 持有期间保活（引用计数 +1） |
| 其余（`Node@`、`Object@` …） | `ObjectID` | **非拥有 / 弱引用**：不保活；对象被释放后访问抛明确异常 |

- `Object@` 是弱引用：把 `RefCounted` 实例赋给 `Object@` **不会**保活它；需要保活请声明具体类型（如 `Resource@`）。
- 非拥有句柄每次解引用都要经 `ObjectDB` 查表校验，因此脚本持有句柄不会阻止对象被释放，也不会产生跨边界的引用环。
- 对象被释放后再经非拥有句柄访问，得到 `AngelScript: <成员名>: object is null or has been freed` 异常，而不是野指针。
- **`is null` 不做存活校验**：非拥有槽里存的是 `ObjectID`，`x is null` / `x !is null` 只比较槽原值、不查 `ObjectDB`；对象释放后槽值不变（非 0），所以弱句柄**不会**变 null。不能用 `if (weak is null)` 判断存活，失效只能靠**访问**（调用方法/读写属性）暴露为上面的异常。

### 已知限制

- 热更脚本类不是基类的子类：基类指令只决定节点实例类型（D2），脚本里的 `this` 是 AS 对象而非 Node，因此不能直接写 `add_child(...)`；绑定层 API 需要显式对象句柄（`Node @n = Node(); n.add_child(...)`）。
- 不可绑定的签名一律整体归入 unbound（不半注册），可通过 `--dump-angelscript-api` 的 `angelscript_unbound.txt` 查询：vararg 方法（含内建 `print`）、`METHOD_FLAG_VIRTUAL` 的虚方法（GDVIRTUAL，AS 侧注册也调不到实现）、形参/返回引用不可见类型的方法、非标识符属性名（如 `frame_0/texture`）。
- 静态方法与内建值类型的静态方法不进 unbound，而是注册为全局函数，名字 `<声明类/类型>_<成员名>`（如 `Node_print_orphan_nodes`）；不绑实例、不受 `const` 修饰。`Variant` 形参与返回也可绑定（AS 侧声明为 `Variant`，经已注册的 Variant 拷贝构造编组；真 `void` 返回仍不编组）。
- vararg 的 `print` 家族因此不可用；热更脚本的字符串输出暂用内建 `as_log_string`（见内建 API 表）。
- 不支持向下转换（无 `opCast`）：只能把派生类句柄赋给祖先类句柄，不能反向。
- 非 `RefCounted` 的 `Object`（如 `Node`）不绑定 `free()`（它是 GDVIRTUAL、没有 MethodBind），脚本无法显式释放，退出时会出现 `ObjectDB instances were leaked` 警告；`RefCounted` 由引用计数正常回收。
- 仅支持 64 位平台：非拥有句柄槽里存 64 位 `ObjectID`，必须完整放进指针宽度的槽；32 位平台会截断高位、可能错认对象。绑定层用 `static_assert(sizeof(void *) >= 8)` 在编译期拦截 32 位构建。
- 绑定注册耗时随绑定面线性增长：默认（空白名单）全量放行时，editor 构建下约 1047 个类、12.8 万条成员，全量注册路径需约 2 分钟（`ASBindingPlan::build()` 只占 0.2 s，耗时在 AS `RegisterObjectMethod` 的固有开销，约 1 ms/条；本版 AS 的注册类型之间不支持继承，只能逐类复制全量成员 + `opImplCast`，没有捷径）。用 `angel_script/class_whitelist` 收窄到脚本实际需要的类型可降到秒级（白名单 5 类时注册 878 条）；默认的惰性注册（见「当前状态」）在 `auto` 模式下只先注册 core 与扫描到的基础类型，其余按需增量注册，避免启动即付全量开销。
- 属性访问器名与同类方法重名时（如 `Image.data` 与真实方法 `get_data`），plan 在编译期即用与运行期一致的判定把该属性归入 unbound（reason `accessor-name-taken`），dump 不再声明它（评审 I1）；成员仍可经同名方法访问。
- 脚本属性的读写（`ASScriptInstance::set()` / `get()`）支持标量、全部内建值类型（含 `String` / `Vector2` / `Array` …，AS 侧存储就是一颗 `Variant`）与对象句柄（拥有 / 非拥有按静态类型编码）；但**小写内建 `string` 类型的属性不做读写往返**（`set` / `get` 返回 `false`），需要字符串属性请声明为 `String`（大写，以 Variant 存储）。

示例 `res://main.as`：

```angelscript
// godot_base: Node

class main {
	int ready_count = 0;

	void _ready() {
		ready_count = 42;
		as_log_int(ready_count);
	}
}
```

## 信号（M3）

脚本类用 `signal_<name>` 命名约定声明信号：方法名以 `signal_` 开头即视为信号声明。此类方法不会出现在普通方法表里（`has_method("signal_xxx")` 为假、`get_script_method_list()` 也不含它），只能经 `Object.connect` / `Object.emit_signal` 使用。信号参数只支持可映射到 `Variant` 的类型（标量与内建值类型）；含对象参数或其它不支持类型的声明会被整体忽略。

```angelscript
// godot_base: Node
class signal_demo {
	void signal_ping(int value) {}

	void _ready() {
		// 连接：as_callable(obj, method) 生成宿主方法 Callable，交给 Object.connect。
		// Object.connect 的形参是 StringName、且 flags 形参没有默认值，需显式转换并补 0。
		as_self().connect(StringName("ping"), as_callable(as_self(), "on_ping"), 0);

		// 发射：as_emit_signal(obj, name, args)，args 是信号实参数组。
		Array args;
		args.push_back(Variant(7));
		as_emit_signal(as_self(), "ping", args);
	}

	void on_ping(int value) {
		as_log_int(int(value));
	}
}
```

- `as_self()`：取当前正在执行的脚本实例所承载的节点（`ASScriptInstance::get_owner()`，弱句柄）。AS 脚本类不是 `Node` 子类，脚本内需要用它才能引用宿主节点、进而连接或发射自己声明的信号。
- `as_emit_signal(obj, name, args)`：发射 `obj` 上名为 `name` 的信号，`args` 必须是 `Array`，其元素按顺序作为信号实参。`obj` 为非拥有句柄，为空或已释放时抛 `AngelScript: as_emit_signal: target object is null or has been freed`。
- `as_callable(obj, method)`：把 `obj` 上的方法包装成 `Callable`，配合 `Object.connect` 接收信号。
- 已知限制：信号（以及任何宿主 `Callable`）回调的**参数只支持标量**——`ASEngine::call_function` 只按 AS 形参宽度编组 `bool` / `int`（32/64 位）/ `float`（32/64 位），带对象、字符串或其它内建值类型参数的信号回调无法派发。需要传递复杂数据时，请让回调只接收标量句柄 / ID，或改用属性通道。
- 已知限制：声明渲染不输出默认值，带默认值形参的 Godot 方法与内建值类型方法（如 `connect(...)`、`String.split(...)`）在 AS 侧都没有默认值，必须显式传全部实参，例如 `connect(StringName("ping"), as_callable(as_self(), "on_ping"), 0)`。

## 垃圾回收（M3）

- AS 自身的自动增量回收保持开启。
- 宿主兜底：`ScriptLanguage::frame()` 按项目设置 `angel_script/gc/interval_seconds`（单位秒，默认 `5.0`）驱动一次完整回收。
- 设为 `0` 表示**关闭「按间隔」的宿主触发**（只保留 AS 自身回收），但脚本实例析构产生的显式回收请求不受此限——它会在下一帧触发一次完整回收。需要宿主完全不介入时，请仅依赖 AS 自身回收并接受脚本对象延迟回收。
- 退出前 `finish()` 会做最后一次完整回收。

### 已知泄漏面（约束）

- AS 对象 → Godot `Node`/`Object`：非拥有句柄不保活，**不会**形成环。
- AS 对象 ↔ AS 对象：由 AS 垃圾回收负责，可回收。
- **AS 对象 ↔ `RefCounted`（Godot 侧强引用）**：AS GC 看不见 Godot 的引用计数，可能形成无法回收的环。需要反向引用时请用 `ObjectID` / `WeakRef`，不要在 Godot 侧强引用脚本对象。
- 脚本创建的非 `RefCounted` 对象（如 `Node`）无法从脚本释放，进程退出时可能出现 `ObjectDB instances were leaked` 告警。

## 内建 API

| 声明 | 说明 |
| --- | --- |
| `void as_log_int(int value)` | 把整数打印到 stdout（阶段一引入的宿主调试桩）。 |
| `void as_log_string(const String &in value)` | 把 Godot `String` 打印到 stdout。内建 `print` 是 vararg、无法在 AS 静态类型里表达，热更脚本的字符串输出暂用这个出口。 |
| `Object @as_self()` | 取当前脚本实例承载的节点（弱句柄），见「信号（M3）」。 |
| `void as_emit_signal(Object @obj, const String &in name, const Array &in args)` | 发射 `obj` 上的信号 `name`，`args` 为实参数组。 |
| `Callable as_callable(Object @obj, const String &in method)` | 把 `obj` 上的方法包装成 `Callable`，配合 `Object.connect` 接收信号。 |
| `Variant as_host_call(int method_id, const Array &in args)` | 调用宿主注册的 `method_id` 方法（M5 互操作 L2），`args` 按顺序展平为实参；未安装宿主表或宿主报错时返回空 `Variant`，见「宿主桥（M5 互操作 L2）」。 |

内建函数统一使用泛型调用约定（`asCALL_GENERIC`），与绑定层保持同一形态。`as_log_string` 在绑定层值类型注册之后才注册，因为它用的是绑定层的 Godot `String` 值类型。

## 构建

需要 MSVC x64（含 `ml64.exe`，由 `vcvars64.bat` 提供）与 SCons：

```
scons platform=windows target=editor tests=yes accesskit=no d3d12=no -j8
```

- `accesskit=no d3d12=no` 用于本机缺少对应依赖的场景；有其他平台需求请按需调整。
- MSVC x64 无法内联汇编调用桩，模块用 `ml64` 单独汇编 `thirdparty/angelscript/source/as_callfunc_x64_msvc_asm.asm`；GNUC 平台还需汇编 `as_callfunc_x64_gcc.S` 并关闭严格别名优化（留待平台里程碑）。

## 测试

```
bin\godot.windows.editor.x86_64.console.exe --headless --test --test-case="*AngelScript*"
```

- doctest 过滤需要用名字通配符（`*AngelScript*`），标签形式 `[AngelScript]` 匹配不到。
- 用例注册写在 `tests/*.h`（经自动生成的 `modules/modules_tests.gen.h` 由 `tests/test_main.cpp` 展开），断言与辅助实现放在 `tests/*.cpp`；后者必须先 `#define <文件对应的>_TESTS_IMPL` 再 include 测试头，否则同一批 `TEST_CASE` 会被注册两遍。
- 本仓 doctest 以无异常模式编译，`REQUIRE` 失败不会中断用例，因此测试辅助函数在每个前置条件后都要显式 `return`（否则实现退化时会以空指针崩溃收场）。

## API 导出（绑定面）

```
bin\godot.windows.editor.x86_64.console.exe --headless --dump-angelscript-api <目录>
```

写出两个文件后退出：

- `angelscript_api.d.as`：当前绑定面（受 `angel_script/class_whitelist` / `class_blacklist` 影响）的声明，首行为 `// angelscript-api-version: <sha256>`，可用于判断热更脚本与当前引擎 API 是否匹配。
- `angelscript_unbound.txt`：不可绑定清单，每行 `owner<TAB>member<TAB>reason`，按字典序稳定排序（两次导出内容逐字节一致）。

## 端到端验收（headless）

```
bin\godot.windows.editor.x86_64.console.exe --headless --path modules/angelscript/tests/e2e_project --quit-after 2
```

预期退出码为 0，stdout 依次出现 `Node`、`1`、`3`、`M3:handle-ok`、`7`、`42`：

- `42`：`main.as` 的 `_ready()` 调用 `as_log_int`（阶段一通道）。
- `7`：`signal_demo.as` 演示 M3 信号闭环——`_ready()` 里 `as_self().connect(..., as_callable(...))` 连接自己声明的 `signal_ping`，再 `as_emit_signal(...)` 发射，回调 `on_ping(int)` 打印 `7`（子节点 `_ready()` 先于父节点，故它在 `42` 之前）。
- `M3:handle-ok`：`handle_demo.as` 演示 M3 弱句柄（`Object@`）赋值后仍指向存活对象。
- `Node` / `1` / `3`：`binding_demo.as` 用绑定层 API 演示 `Node.get_class()` 的 String 编组、`add_child` 的对象参数、`get_child_count` 的 int64 返回值与 `process_mode` 属性读写。

没有 `Failed to load AngelScript` / `Failed loading resource` 之类的错误。这一次运行覆盖：`.as` 加载器注册、`.tscn` 按路径挂载脚本、`get_instance_base_type()` 生效、`_ready` 经 `GDVIRTUAL` → `callp()` 派发、内建 API 与绑定层均可调用。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `as_script_language.h/.cpp` | `ScriptLanguage` 实现（单例、语言名 `AngelScript`、扩展名 `as`）。 |
| `as_engine.h/.cpp` | AS 引擎持有者：懒初始化、模块编译、`execute`/`call_function`、内建注册与绑定层接线。 |
| `as_host_bridge.h/.cpp` | M5 宿主回调桥：持有并校验版本化的 `ASHostCallbacks`，按 `method_id` 派发（宿主桥 L2 边界）。 |
| `as_script.h/.cpp` | `Script` 实现（`.as` 资源：基类指令解析、类名校验、编译）。 |
| `as_script_instance.h/.cpp` | `ScriptInstance` 实现（实例创建、属性读写、方法/通知派发）。 |
| `as_resource_format.h/.cpp` | `.as`/`.asb` 的资源加载器与保存器。 |
| `binding/` | M2 绑定层：`as_binding_decl`（内省→声明/编组）、`as_binding_plan`（可见性与计划）、`as_binding_value_types`（内建值类型）、`as_binding_object`（对象类型）、`as_binding_registry`（注册编排）、`as_binding_dumper`（导出产物）。 |
| `register_types.cpp` | 模块注册（`ScriptServer`、资源加载器/保存器、API 导出入口）。 |
| `thirdparty/angelscript/` | AngelScript 2.38.0 官方源码（zlib 许可，见 `LICENSE.txt`）。 |
| `tests/` | doctest 用例；`tests/e2e_project/` 为 headless 验收项目。 |

## 许可

- 第三方 `thirdparty/angelscript/` 使用 zlib 许可，许可全文见 `thirdparty/angelscript/LICENSE.txt`。
- 模块自身源码沿用 Godot 的 MIT 许可头（由 `misc/scripts/copyright_headers.py` 统一维护）。
