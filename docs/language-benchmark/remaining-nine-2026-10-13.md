# AngelScript 剩余九项：第二轮优化与实验报告

> 本轮实现、隔离实验、回退/调试验证和复测已完成，尚未完成“剩余九项稳定追平 GDScript”。默认最终二进制合计六个进程、每进程七次重复；保留全部慢样本。性能波动显著，不把表面下降百分比当作已证明的代码收益。

## 范围和基线

本轮在前一轮未提交优化之上继续，保留原有 17 项、三语言实现、迭代次数、API 和校验和，不修改第三方 AngelScript VM/JIT。用户要求内联执行，不自动提交。历史差距见[前一轮报告](binding-parity-phase12-2026-10-13.md)。

原负载使用 [benchmark 驱动](../../modules/angelscript/tests/binding_parity_benchmark.py)，每阶段 3 个独立进程、每进程 7 次测量及原有预热；汇总值为进程中位数的中位数。新加的 [boundary probes](../../modules/angelscript/tests/test_angelscript_boundary_probes.cpp) 将 push/get、字面量/缓存键、几何 add/assign、空 native 边界和 dispatch 分开测量。它们不是三语言同算法排名，也不能把空调用与复杂调用的差值直接归因于某一层。

## 实施内容

### Array 和字面量

Array 安全白名单方法直接委托 Godot Array，减少 ValidatedBuiltInMethod 分发和无用 Variant 参数/返回中间值。写入仍调用核心方法，保留 typed/read_only 约束，而不是修改裸存储。原索引负载的算法保持不变。

新增 `Variant(const string &in)` 和 `String opAdd(const string &in) const`。小写 `string` 字面量由驻留字符串表示，变量/参数需要槽解引用；统一只读解码后 native/generic 使用相同签名。不把循环里的 key 缓存到脚本外，也不把 String 追加改成 StringBuilder，所以原测量仍包含创建和拼接成本。

### dispatch 安全修复和保留上下文实验

复制 ASCallPlan 到调用栈，避免重入扩容实例缓存后继续读取悬空指针。新测试在 native 回调中使外部计划失效，验证返回解码仍使用调用前快照；这不是一个穷举所有 HashMap 重入时序的测试。检查参数和返回值引用限定：Variant host bridge 没有 ref/out 储存契约，不能把引用当普通数值 SetArg。仅匹配函数且 scalar-eligible 的计划使用缓存参数计数。引用参数从不安全的误编组改为明确拒绝，不宣称新增了 ref/out 宿主支持。

隔离实验仅比较全局无参标量函数直接 context API 的 Prepare/Execute 与 Prepare/Execute/Unprepare。它不是 ScriptInstance dispatch 的等价替身。完成方法的 context 持有 `this`；调用方释放对象后，析构仍等到 Unprepare。这会改变可观察生命周期，因此无论计时收益多大，本轮都不将保留 context 用于默认调用路径。保留原 Unprepare 和重入 PushState/PopState。

### 几何值实验

独立构建开关 `angelscript_pod_geometry_probe=yes`，默认关闭。仅 Vector2/Vector3/Color 有编译期平凡复制/析构检查；VM 的 POD 复制语义与 native APP_CLASS 隐藏返回槽 ABI 分开处理，不能把内存返回类型改成寄存器返回。

显式 `Vector2(other)` 等仍需 copy constructor 签名，因此保留默认和复制构造，只试验去掉 opAssign/析构回调。是否接受以及性能见最终实验章节。

### Packed 和 typed_i64

上一轮 scalar Packed 已直接访问 Vector；typed_i64 原本已 reserve。此次先测分离 push/get 和空边界。没有证据表明重写私有存储是主要收益来源时，不改变别名/COW/越界等语义去换取未经证实的加速。

## 隔离实验记录

### 完成上下文保留：拒绝默认启用

[原始样本](data/context-retention-probe.csv) 与[实验元数据](data/context-retention-probe.json)：每模式 3 个进程、每进程 7 次记录、200000 次无参标量调用。每次 Unprepare 的进程中位数为 45.22、54.78、43.26 ns；保留 context 为 32.97、27.60、51.285 ns。汇总为 45.22 对 32.97 ns，约降低 27.1%，但第三个进程反而变慢，不能宣称稳定幅度。

比性能更重要的是语义测试：已完成的 `Item.value()` context 仍持有 `this`，外部释放后 `destroyed==0`，直到 Unprepare 才变成 1。异常后重新 Prepare 可以成功，不代表生命周期等价。因此仅保存测试与 CSV，默认 ScriptInstance 不采用保留策略。无需为了挽救这个不合格方案更改 VM 或推迟热重载清理。

### 几何生命周期：暂保留为默认关闭实验

[拆分数据](data/boundary-probes-geometry-summary.csv) 中 vector_add 为 41.775 ns（进程范围 38.69–43.615），vector_assign 为 14.615 ns（14.37–15.325），color_add 为 43.65 ns（43.45–76.995）。direct 阶段对应 112.19 / 32.125 / 81.185 ns，但其进程波动较大。这是减少 VM→native 赋值/析构边界的正面证据，不是已追平 GDScript 的证据。

POD 模式改变注册行为集合，去掉显式 opAssign/析构，同时保留 native APP_CLASS 隐藏返回槽 flags。即使同版本全套测试通过，也不等于已验证所有历史预编译字节码与外部 API 使用者；本轮默认仍关闭。完整原负载的实验数据与默认最终数据分别保存，不把实验成绩冒充默认启用成绩。

[原负载几何实验汇总](data/binding-parity-nine-geometry-summary.csv)：Vector 为 30.358 ns，同期 GD 为 14.970 ns，仍慢 2.03 倍；Color 为 33.124 ns，GD 为 23.202 ns，仍慢 1.43 倍。AS 进程中位数范围分别为 28.614–56.011 ns 和 30.076–62.603 ns，GD Color 为 12.974–33.246 ns，波动显著。实验足以说明生命周期边界值得继续研究，但没有稳定追平证据，也没有默认启用。其他负载随阶段大幅波动，不能归因于几何开关。

### Packed/typed 存储：不重写

geometry 探针中，empty_native 为 37.935 ns，Packed get/push 为 33.45 / 32.16 ns，typed get/push 为 36.485 / 36.76 ns。这些逻辑迭代并非相同指令序列，不能得出负的存储成本；但未观察到足够大的可分离存储瓶颈，优先减少跨边界次数比冒险改 COW/别名语义更有依据。本轮不重写私有缓冲，也不替换原负载成批量调用。

## 回归与构建

几何实验开启的构建通过完整 AngelScript 模块测试：190 个测试、3797 条断言全部通过。扩展回归包含 Vector2/Vector3/Color 的成员、显式复制、链式赋值、值参数/返回、临时对象、typed-array 返回后独立修改，以及异常断点中的值展开。

[真实 TCP 协议轨迹](../angelscript-debugger-nine-geometry-trace.json) 验证 breakpoint → step → out → next → continue，最终输出 6。这是协议验收，不是手动 GUI 验收。模块测试中的 host bridge error 3 和缺少默认工厂消息来自既有负面测试；测试进程退出成功。

forced-generic 构建通过，value-ops 回退验证 37 个测试 / 83 条断言全部通过，包含本轮字面量签名、Array 约束和通用几何行为。typed geometry container 专属 ABI 检查仅在 native storage 开启时执行；不能从 generic 子集推断 typed containers 已具备通用 ABI。已恢复默认 native（`angelscript_force_generic=no angelscript_pod_geometry_probe=no`），最终完整模块 190 个测试 / 3797 条断言全部通过。[默认配置 TCP 轨迹](../angelscript-debugger-nine-final-trace.json) 同样通过断点与单步协议，输出 6。

未运行整个 Godot 引擎全部测试、跨平台 ABI 或手动编辑器操作；不把模块全测等同于这些验收。

## 最终原负载：先保留异常，再复验

默认最终组 `nine-final` 的 153 条记录通过校验，四阶段合计 612 条记录的迭代次数和校验和一致。[四阶段比较](data/remaining-nine-comparison.csv) 将几何实验与默认最终结果分列；其中 6/17 项在默认最终组达到数值上的 AS/GD ≤ 1，但本轮目标九项无一达到。

该组出现强波动：Vector 的 AS 进程中位数为 88.862–395.410 ns，GD 为 12.851–68.038 ns；Dictionary 的 AS 为 145.460–711.770 ns；dispatch 的 AS 为 171.930–482.895 ns，GD 为 74.610–426.455 ns。没有证据证明全部来自代码，也没有证据允许删除这些慢样本。Array 的汇总从 214.04 到 237.01 ns，Vector 从 66.335 到 325.126 ns，不能隐瞒这些不利数据。

因此用完全相同二进制新增 `nine-confirm` 三进程复验，不覆盖 `nine-final`。复验 153 条记录通过，实际引擎/源码 diff/源基线哈希与第一组一致。[稳定性比较器](../../modules/angelscript/tests/remaining_nine_stability.py) 合并全部六个进程中位数；[合并结果 CSV](data/remaining-nine-stability.csv) 保留两组各自中位数及所有进程范围。不挑选更快阶段，复验后仍未获得稳定环境。

### 目标九项：默认最终六进程合并

单位 ns / 原负载逻辑迭代；越小越好。AS/GD 是汇总中位数之比，**不是有置信保证的速度比**。偶数样本中位数取中间两个值的均值。

| 原负载 | AS | GD | AS/GD | AS 进程中位数范围 | 判定 |
|---|---:|---:|---:|---:|---|
| Array | 246.415 | 118.530 | 2.08 | 182.160–895.090 | 未追平 |
| Dictionary | 178.875 | 79.225 | 2.26 | 145.460–711.770 | 未追平 |
| Array methods | 292.245 | 298.175 | 0.98 | 233.690–1217.990 | 数值接近，不是稳定胜出 |
| String | 501.375 | 394.375 | 1.27 | 440.375–604.625 | 未追平 |
| Packed | 101.915 | 57.455 | 1.77 | 88.960–146.550 | 未追平 |
| typed_i64 | 97.580 | 50.225 | 1.94 | 94.730–140.740 | 未追平 |
| dispatch | 276.985 | 140.370 | 1.97 | 171.930–728.910 | 未追平 |
| Vector | 194.850 | 17.919 | 10.87 | 81.483–395.410 | 未追平，幅度受强波动影响 |
| Color | 87.301 | 17.472 | 5.00 | 62.969–168.560 | 未追平 |

目标九项中仅 Array methods 合并后数值略低于 GD；但 AS 与 GD 各进程范围极宽，GD 为 187.89–946.88 ns，不应宣布稳定追平。Vector 第一组 AS 中位数 325.126 ns，复验为 97.057 ns；两组相同二进制，这也说明不能只用单组数字推断真实回退倍数。

全 17 项合并结果有 8 项数值上 AS/GD ≤ 1，包括原本优势的纯计算、类/调用和批量向量；engine 的 0.98 与 Array methods 的 0.98 都接近噪声区。batch_sum 本次为 1.36，不再呈现前轮的微弱数值优势。完整 C# 列保留在 CSV，因 Debug/CLR 差异不用于证明 AS/GD 绑定层成本。

**本轮没有证据证明新增优化已消除所有慢项，也没有证据证明默认改动没有性能回退。** 字面量 direct 阶段和几何隔离实验给出正面线索，正确性测试通过，但最终验收只能报告当前差距与不确定性；更稳定的交错 A/B 是下一步前置条件。

### 默认最终构建身份

[构建记录](data/binding-parity-nine-final.json)：采样开始于 2026-10-13 19:36:39 +0800；源基线为 `20530409fc50563e19954441aa1b81f47ffcee0e`，工作区未提交。实际运行引擎 SHA256 为 `ac5d6c0602789343d7b9994512e5c107bcfd1f8be90c158a71a85b024da39580`，测量后再次核对一致；tracked module diff SHA256 为 `84c89dd3d6e86e1fad71e7350f00439e6a37657e765df950c08fa668cf3c0847`。这不是 console launcher 的哈希。

## 复现与产物

构建入口为 [build-tests.cmd](../../.superpowers/sdd/2026-09-30-angelscript-hot-update-phase1/build-tests.cmd)。默认配置使用参数 `module_mono_enabled=yes angelscript_force_generic=no angelscript_pod_geometry_probe=no`；实验只将最后一个开关改为 `yes`；generic 回退将前一个开关改为 `yes`、几何实验关闭。构建后执行 `bin\\godot.windows.editor.x86_64.mono.console.exe --headless --test --test-case="*[AngelScript]*"`；generic 本轮验证范围是 `*native value ops*`。

[原负载采集器](../../modules/angelscript/tests/binding_parity_benchmark.py) 的参数为 `--stage <新的阶段名> --runs 3`，不会覆盖已有阶段。保留每个进程的 min/median/max，未保存每次重复的逐次计时。每行代表一组重复的汇总，不是单次运行。

[拆分探针采集器](../../modules/angelscript/tests/boundary_probe_benchmark.py) 同样接受新的 `--stage` 与 `--runs 3`；[上下文保留实验](../../modules/angelscript/tests/context_retention_benchmark.py) 拒绝覆盖已存在结果。[跨阶段比较器](../../modules/angelscript/tests/remaining_nine_compare.py) 校验原负载的语言集合、迭代次数和校验和，并将几何实验成绩与默认最终成绩分列。

### 数据索引

| 阶段 | 逐进程汇总 | 阶段汇总 | 构建标识 | 原始输出 |
|---|---|---|---|---|
| 本轮 before | [CSV](data/binding-parity-nine-before.csv) | [汇总](data/binding-parity-nine-before-summary.csv) | [JSON](data/binding-parity-nine-before.json) | [日志](data/binding-parity-nine-before.log) |
| Array / 字面量 direct | [CSV](data/binding-parity-nine-direct.csv) | [汇总](data/binding-parity-nine-direct-summary.csv) | [JSON](data/binding-parity-nine-direct.json) | [日志](data/binding-parity-nine-direct.log) |
| 几何实验开启 | [CSV](data/binding-parity-nine-geometry.csv) | [汇总](data/binding-parity-nine-geometry-summary.csv) | [JSON](data/binding-parity-nine-geometry.json) | [日志](data/binding-parity-nine-geometry.log) |
| 默认最终 | [CSV](data/binding-parity-nine-final.csv) | [汇总](data/binding-parity-nine-final-summary.csv) | [JSON](data/binding-parity-nine-final.json) | [日志](data/binding-parity-nine-final.log) |
| 同二进制确认 | [CSV](data/binding-parity-nine-confirm.csv) | [汇总](data/binding-parity-nine-confirm-summary.csv) | [JSON](data/binding-parity-nine-confirm.json) | [日志](data/binding-parity-nine-confirm.log) |

拆分探针的各阶段汇总：[before](data/boundary-probes-before-summary.csv)、[direct](data/boundary-probes-direct-summary.csv)、[geometry](data/boundary-probes-geometry-summary.csv)。每个同名前缀还保存 CSV / JSON / log。原始输出日志受仓库通用忽略规则影响，当前在本地存在；本轮没有提交操作。

## 本轮决策与下一步

**保留默认实现：** Array 直接核心方法、安全的 String/Variant 小写 string 重载、ASCallPlan 快照和引用限定检查。它们已通过 native/generic 正确性回归；原 API 及原算法保留，没有用批处理或缓存 key 来替换排名负载。但本轮时序数据不足以给每个改动可靠分配收益百分比。

**不启用：** 已完成 context 保留（推迟析构）；几何 POD（仅实验开关，旧字节码/API 兼容未完整验收）；私有 Packed/typed 缓冲重写（拆分证据不足）。不通过取消类型/只读检查、对象生命周期或调试能力来换速度。

进一步工作应先在空闲、稳定且可记录 CPU 调度/频率的环境重做交错 A/B，再判断默认变化的性能是否值得保留或回退。当前数据不足以证明 Array 和 dispatch 无性能回退，也不足以证明其表面变慢来自代码。之后优先研究几何赋值/析构边界与兼容性，保留原始负载验收；StringBuilder、缓存键和 batch 仅可另列为用法优化，不应替换本轮测试来制造追平。

## 解释边界

- 不连续交错 A/B；同机其他负载和频率变化会影响结果，不能把所有阶段差值归因于代码。本轮 before 的 Array methods 进程中位数为 213.47–1354.59 ns，dispatch 为 101.775–557.905 ns，基线波动已大到不能直接把汇总下降百分比当代码收益。拆分微基准也有明显波动；没有采集调度/CPU 频率轨迹，因此不猜测具体噪声来源。
- C# 仍是当前编辑器实际加载的 Debug 组件；部分 C# 容器是 CLR 容器，不能称为相同 Godot 容器底层实现。
- 原始 AS engine/hoist 探针创建 Node 而未销毁，仍可出现每进程 16 个泄漏警告；本轮没有修改基准来消除这个既有缺陷。
- 原负载固定 AS/C#/GD 顺序，typed_vec2 已使用批量 API；其胜出不等于 VM 标量指令性能。
- forced-generic 只验证 value-ops 回退，不证明历史 typed containers 的跨平台 ABI 完整。
- native thunk 计数与历史版本一致保留；测试构建不是完全移除观测开销的发行版。
- 源码位于 dirty 工作区；基准记录真实运行引擎哈希和 tracked module diff 哈希。未跟踪测试/文档不在 tracked diff 哈希内。
