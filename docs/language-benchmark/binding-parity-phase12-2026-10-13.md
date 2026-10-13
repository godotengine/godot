# AngelScript 绑定优化前两阶段：实现与验收报告

日期：2026-10-13。实现与正式四阶段计时已经完成。前两阶段显著缩小了主要绑定慢项，但**未达到所有负载追平 GDScript**；本轮数值门槛命中 8/17，其中 batch_sum 只是接近持平，不构成稳定领先证据。

## 1. 目标与判断口径

本轮落实用户确认的绑定优化前两阶段，不修改 AngelScript VM/JIT，不换脚本 API，不用批处理替代原来的慢项。基线为 `20530409fc50563e19954441aa1b81f47ffcee0e`，修改尚未提交。

**追平定义：同一阶段 AS 的进程中位数汇总 / GDScript 的进程中位数汇总 ≤ 1.00。** 加速很多不等于追平；不能把最佳值、不同算法或过去不同配置的结果拼接比较。

## 2. 已实现的路径

### 2.1 generic 元数据从 GetUserData 改为 auxiliary

在值类型方法、行为、全局函数、对象工厂/方法/属性/转换及 utility 注册时，将现有不可变 binding 记录直接作为 auxiliary 传入。generic 跳板使用 `p_gen->GetAuxiliary()`；旧 `SetUserData` 仍保留供兼容和检查，记录的既有生命周期不变。

AngelScript 的函数 `GetUserData` 会取得 engineRWLock；当前 Windows 实现涉及 `WaitForSingleObjectEx` / `ReleaseSemaphore`。一次 Array/Dictionary 表达式会经过多个构造、转换和析构，锁成本不只在 getter 上。auxiliary 路径消除了这些调用点的锁，而非关闭 AngelScript 线程安全。没有修改第三方 VM。

新增遍历注册方法/行为/工厂/全局函数的测试，检查所有带旧 metadata 的函数满足 `GetAuxiliary() == GetUserData()`。测试在生产修改前确实失败，修改后通过。

### 2.2 非调试执行不安装 line callback

异常回调一直保留。行回调仅在 `EngineDebugger::is_active()` 时安装，在执行入口同步；离开调试状态后清除。通过 context 标记和原子 live-count 避免非调试常态在每次 dispatch 中再调用带锁 `GetUserData()`。context 清理回调维护 live-count。

原有断点/暂停/step/next/out 逻辑未删除；同一 context 支持 inactive → active → inactive 的入口切换。内部 RequestContext、宿主 call_function 和脚本工厂执行入口均同步。

边界：这里不是“无开销调试”。启用调试器后仍有逐行回调。无回调的长时间执行若期间才首次出现调试器，不会凭空发现它；下一次执行入口才同步。没有宣称支持任意时刻热附加。

### 2.3 Variant/boxed 生命周期与转换

已验证 MSVC x64 ABI 上，Variant、String、StringName、Array、Dictionary 和五种 Packed 标量数组增加直接默认/拷贝构造、析构及赋值包装。Variant 的 bool/int64/double/String 构造与返回转换走直接 C++ 包装。

保持 Variant 布局、引用计数和非 POD 隐藏返回槽约定，不更改 VM 类型标志以强行启用寄存器返回。小写 `string` 字面量/变量的特殊槽解码仍走既有 generic 路径，未冒险把工厂指针当成地址再解引用。

### 2.4 Packed 与动态容器访问

PackedByte/Int32/Int64/Float32/Float64 的 size、push_back、索引读写直接取 `VariantInternalAccessor<Vector<T>>::get` 返回的内部 Vector 引用，避免临时 Variant 和重复 validated-method 跳转。不能先复制 Vector 再写，否则会增加引用并触发额外 COW。

本实现保持原绑定的复制语义：Variant 拷贝共享 PackedArrayRef，显式 duplicate 分离。负索引、越界默认返回/忽略写、窄化和浮点精度均有回归测试。

Array 的读取直接取 const Array；Dictionary 读取使用 const getptr，缺失键返回 nil 且不插入。写入仍经过核心 `set_indexed`/`set_keyed`，保留 typed/read-only 校验，不以绕过安全检查获取速度。

### 2.5 回退开关

新增 SCons `angelscript_force_generic=yes`，只作用于本模块绑定/测试源码，不重新配置或修改 VM。它触发既有 `AS_NATIVE_VALUE_OPS_FORCE_GENERIC` 门控；默认关闭。

这验证的是本轮及既有 value-ops 的 generic 回退，**不是证明整个模块在所有平台可移植**。此前 typed-containers 的原生包装跨平台缺口未在本轮重构；其 geometry ABI 与 native storage 绑定的问题仍应单独处理。未在其他 CPU/操作系统上运行测试。

## 3. 基准方法与可复现性

执行器：[binding_parity_benchmark.py](<../../modules/angelscript/tests/binding_parity_benchmark.py>)。汇总器：[binding_parity_compare.py](<../../modules/angelscript/tests/binding_parity_compare.py>)。

- before → auxiliary → callbacks → final：逐阶段独立构建并顺序测量，不同时构建和计时。
- 每阶段三个独立进程；原有每项七次重复，普通负载有预热，dispatch 沿用原驱动无单独预热的做法。
- 保持原 17 项 × 三语言 × 三进程 = 153 个汇总样本。每条 BENCH 保存进程内 min/median/max，而非七次逐次原始计时。
- 报告使用三个进程 median 的 median。范围为三个进程 median 的 min～max，不是置信区间；跨阶段校验 checksum。
- 时间单位 ns/逻辑迭代；一个迭代可能含 push+get 等多个操作，不能当成一次 hash lookup 或一次 C++ 调用的耗时。
- AS/C#/GD 始终按原固定次序运行，存在顺序与系统负载噪声；没有随机交错 A/B。
- 硬件沿用 Ryzen 7 7700（8C/16T）、Windows 10 build 19045、约 95.1 GiB 内存。C# 为实际加载的 Debug 程序集，不称为 Release。

**文件身份限制：** 前三个阶段 JSON 记录的是 `.console.exe` 启动器 SHA256，而非真正的大型 `.exe` 引擎。这不能独立证明当时运行内核字节身份，旧二进制也未归档。本轮最终执行器修正为同时记录 launcher/runtime SHA256 和模块 tracked diff SHA256。阶段顺序、构建日志、完整运行输出与源码审查是旧阶段的辅助证据；没有回填猜测的 runtime hash。HEAD 相同并不代表阶段源码相同。

## 4. 最终计时结果

以下均为 **ns/逻辑迭代，越小越好**；before/auxiliary/callbacks/final 每格是三个进程中位数的中位数。

### 4.1 分阶段 AS 耗时

| 原负载 | before | auxiliary | callbacks | final | before/final 加速比 |
|---|---:|---:|---:|---:|---:|
| arith | 47.268 | 36.312 | 13.309 | 14.140 | 3.34× |
| calls | 35.109 | 37.331 | 22.899 | 23.160 | 1.52× |
| class | 51.585 | 54.967 | 32.531 | 35.028 | 1.47× |
| array | 1233.500 | 228.050 | 225.280 | 166.670 | 7.40× |
| dictionary | 1739.920 | 251.890 | 258.110 | 190.670 | 9.13× |
| packed | 102.440 | 95.350 | 79.050 | 66.930 | 1.53× |
| array_methods | 1352.920 | 290.170 | 272.370 | 224.030 | 6.04× |
| string | 1339.500 | 545.375 | 540.375 | 433.625 | 3.09× |
| vector | 74.938 | 83.262 | 68.550 | 66.060 | 1.13× |
| color | 74.239 | 79.080 | 77.627 | 68.378 | 1.09× |
| engine | 50.305 | 50.895 | 39.185 | 35.450 | 1.42× |
| hoist | 26.960 | 18.295 | 9.010 | 9.465 | 2.85× |
| typed_i64 | 97.420 | 83.230 | 71.870 | 70.470 | 1.38× |
| typed_vec2 | 6.280 | 5.190 | 4.140 | 4.600 | 1.37× |
| typed_map | 114.020 | 107.500 | 78.960 | 78.970 | 1.44× |
| batch_sum | 50.080 | 40.260 | 32.090 | 28.080 | 1.78× |
| dispatch | 120.130 | 125.725 | 131.625 | 110.920 | 1.08× |

**可归因的方向、不可过度归因的数字：** auxiliary 阶段 Array 从 1233.50 降到 228.05，Dictionary 从 1739.92 降到 251.89，表明原 generic 元数据读取路径是主要成本之一；最终生命周期/容器优化进一步降到 166.67/190.67。callback 阶段 arith 从 36.31 降到 13.31，符合移除 VM 逐行回调的机制。纯 arith 在 auxiliary 阶段也变化明显，而它不以 generic 元数据为主要工作，因此不能把阶段差分的每个 ns 都当成该改动独有收益。分阶段独立进程运行仍受频率、缓存、调度和代码布局影响，缺少交错重测与硬件计数器归因。

### 4.2 最终三语言对照与剩余差距

AS/GD > 1 表示 AS 较慢；下降百分比相对本轮 before 的 AS。范围列是 final 的三个 AS 进程 median 的 min～max。

| 原负载 | AS final | GD final | C# Debug final | AS/GD | AS 耗时下降 | AS 进程 median 范围 |
|---|---:|---:|---:|---:|---:|---:|
| arith | 14.140 | 33.261 | 1.921 | 0.43× | 70.09% | 14.040～14.378 |
| calls | 23.160 | 93.534 | 2.451 | 0.25× | 34.04% | 22.712～24.428 |
| class | 35.028 | 164.697 | 2.349 | 0.21× | 32.10% | 34.852～35.106 |
| array | 166.670 | 91.030 | 174.160 | **1.83×** | 86.49% | 165.370～170.730 |
| dictionary | 190.670 | 66.520 | 813.990 | **2.87×** | 89.04% | 171.880～197.610 |
| packed | 66.930 | 43.020 | 5.270 | **1.56×** | 34.66% | 63.610～67.500 |
| array_methods | 224.030 | 176.790 | 223.360 | **1.27×** | 83.44% | 223.830～235.970 |
| string | 433.625 | 277.000 | 585.250 | **1.57×** | 67.63% | 423.125～513.625 |
| vector | 66.060 | 13.934 | 7.384 | **4.74×** | 11.85% | 65.044～72.652 |
| color | 68.378 | 13.669 | 16.131 | **5.00×** | 7.89% | 63.456～70.571 |
| engine | 35.450 | 42.360 | 24.730 | 0.84× | 29.53% | 33.150～37.155 |
| hoist | 9.465 | 17.005 | 1.480 | 0.56× | 64.89% | 9.260～9.505 |
| typed_i64 | 70.470 | 42.560 | 3.790 | **1.66×** | 27.66% | 66.430～79.390 |
| typed_vec2 | 4.600 | 58.210 | 25.160 | 0.08× | 26.75% | 3.660～5.800 |
| typed_map | 78.970 | 102.230 | 49.930 | 0.77× | 30.74% | 77.350～81.640 |
| batch_sum | 28.080 | 28.320 | 3.200 | 0.99× | 43.93% | 27.320～29.640 |
| dispatch | 110.920 | 68.750 | 238.785 | **1.61×** | 7.67% | 103.285～122.585 |

**验收结论：** 数值上 8/17 满足 AS/GD≤1，但不把它称为“8 项都稳定领先”：batch_sum 只差约 0.85%，远小于它 8.26% 的 AS 进程中位数范围；typed_vec2 原本已使用批量算法。其余 9 项仍未追平。Vector/Color 的绝对耗时略降，但本轮 GD 也更快，AS/GD 没有改善；dispatch 仅下降 7.67%，其 17.40% 进程范围更大，不能据此认定稳定收益。

### 4.3 数据与构建身份

[四阶段汇总 CSV](<data/binding-parity-comparison.csv>) 包含未舍入数值、下降比例、AS/GD、离散范围和数值门槛判定。

- before：[样本](<data/binding-parity-before.csv>) / [进程汇总](<data/binding-parity-before-summary.csv>) / [元信息](<data/binding-parity-before.json>) / [原始日志](<data/binding-parity-before.log>)。
- auxiliary：[样本](<data/binding-parity-auxiliary.csv>) / [进程汇总](<data/binding-parity-auxiliary-summary.csv>) / [元信息](<data/binding-parity-auxiliary.json>) / [原始日志](<data/binding-parity-auxiliary.log>)。
- callbacks：[样本](<data/binding-parity-callbacks.csv>) / [进程汇总](<data/binding-parity-callbacks-summary.csv>) / [元信息](<data/binding-parity-callbacks.json>) / [原始日志](<data/binding-parity-callbacks.log>)。
- final：[样本](<data/binding-parity-final.csv>) / [进程汇总](<data/binding-parity-final-summary.csv>) / [元信息](<data/binding-parity-final.json>) / [原始日志](<data/binding-parity-final.log>)。

四阶段共 612 个进程级汇总样本，跨语言和跨阶段输出 checksum 校验通过。最终计时开始于 `2026-10-13T17:56:14+0800`。最终 runtime SHA256：`d05610490e0e962a08cdaa69f2d997ca562500dbfb8fd972a761a29b8b8f6200`；模块 tracked diff SHA256：`03da398d05e85c32630a2c02b6ebbefb2abf4109255913d1e347b8547367e0da`。

## 5. 验证

目前已完成：原生配置完整 AngelScript 模块 184 tests / 3711 assertions 通过；原生 value-ops 34 tests / 102 assertions 通过；强制 generic 下同组 34 tests / 70 assertions 通过。计数差异来自平台门控断言，而非删除功能用例。

TCP 协议测试已通过，校验暂停位置 `[9, 5, 10, 11]`、step/out/next/continue、局部变量/全局变量和最终输出 6。不是人工 GUI 验收。

强制 generic 构建完成后已恢复 `angelscript_force_generic=no` 原生构建（成功，用时 43.65s），完成正式 final 计时，随后再次跑完整 AS 模块仍为 184/184 tests、3711/3711 assertions，TCP 协议也再次通过。实际引擎 SHA256 与 final 元信息一致。`git diff --check` 和 Python 语法检查已通过。VM 和基准工程源码 diff 为空；报告本地数据链接已检查可访问。

完整模块测试中 host bridge error 3 与缺省工厂缺失是预期负例 stderr，进程正常返回 0。构建仍提示可选 ANGLE 依赖、既有 [as_script.cpp:525](<../../modules/angelscript/as_script.cpp#L525>) 名称遮蔽警告。

## 6. 解释边界与后续工作

原基准工程 AS engine/hoist 创建的 Node 没有释放，每个进程报告既有 16 个泄漏；本轮为保持 before/after 负载一致没有修改算法。校验值只保证被打印的结果一致，不能排除所有生命周期问题。

C# packed/typed_i64 使用 CLR 数组，typed_map 使用 CLR Dictionary；typed_vec2 的 AS 已有 C++ 批量路径而 GD 为逐元素循环。这些是应用路径比较，不能归因于 VM 速度本身。

保留原 native thunk 测试计数插桩：非线程安全全局计数及其潜在额外成本不是本轮解决的课题，不将这些单线程结果外推为多线程性能保证。没有重跑整个 Godot 引擎测试集、其他平台、GUI 或独立 profiler 性能测试。

第三阶段建议按仍未追平的实际路径拆分，不在本轮自动扩大实施范围：

1. **Dictionary/Array/String 表达式路径。** Dictionary 的 2.87× 仍包括循环内构造 key、返回 Variant、标量转换与生命周期，不能理解成 hash table 本身慢 2.87×。先分离字面量→String→Variant 与返回转换的成本；String 对比仍是 AS `s=s+...`、GD `s+=...`，检查 COW 和临时对象，不以换成 StringBuilder 来冒充原 API 达标。
2. **Vector/Color ABI 与临时对象。** 4.74×/5.00× 是最大的剩余相对差距。它们早已 native storage/native op，继续分析运算、赋值、析构这些跨 ABI 边界和属性读取；任何 POD/返回寄存器改动必须先过 MSVC x64 返回槽测试，不能重复既往失败的 ABI 假设。
3. **Packed/typed_i64 与 dispatch。** 分离逐元素边界和真实容器成本；dispatch 检查 Prepare/Execute/Unprepare、重入时 call-plan 指针稳定性和引用参数资格，独立测量，不删掉正确性保护来凑目标。

不能再简单宣称“把 getter 改成 native 就行”。大范围 VM 优化或改变脚本 API 另行确认。

### 主要实现与验证入口

[generic 值绑定](<../../modules/angelscript/binding/as_binding_value_types.cpp>)、[对象绑定](<../../modules/angelscript/binding/as_binding_object.cpp>)、[utility 注册](<../../modules/angelscript/binding/as_binding_registry.cpp>)、[native value ops](<../../modules/angelscript/binding/as_binding_native_value_ops.cpp>)、[调试器回调](<../../modules/angelscript/as_debugger.cpp>)。

[值类型回归测试](<../../modules/angelscript/tests/test_angelscript_native_value_ops.cpp>)、[调试回归测试](<../../modules/angelscript/tests/test_angelscript_debugger.cpp>)、[最终 TCP trace](<../angelscript-debugger-parity-final-trace.json>)、[实施计划与偏差记录](<../../Documents/计划/2026-10-13-angelscript-binding-parity-phase12.md>)。

## 7. 复现命令

在项目根目录执行（stage 名不可覆盖已有 raw log，复测请使用新名）：

```powershell
cmd /c ".superpowers\sdd\2026-09-30-angelscript-hot-update-phase1\build-tests.cmd module_mono_enabled=yes angelscript_force_generic=yes"
bin\godot.windows.editor.x86_64.mono.console.exe --headless --test --test-case='*native value ops*'
cmd /c ".superpowers\sdd\2026-09-30-angelscript-hot-update-phase1\build-tests.cmd module_mono_enabled=yes angelscript_force_generic=no"
bin\godot.windows.editor.x86_64.mono.console.exe --headless --test --test-case='*[AngelScript]*'
python modules/angelscript/tests/debugger_protocol_test.py --editor bin/godot.windows.editor.x86_64.mono.console.exe --output docs/angelscript-debugger-parity-final-trace.json
python modules/angelscript/tests/binding_parity_benchmark.py --stage final-recheck --runs 3
python modules/angelscript/tests/binding_parity_compare.py
```

最后的比较脚本固定读取 before/auxiliary/callbacks/final 四个正式阶段；新名复测不会自动替换正式 final 数据。
