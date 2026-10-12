# C# / AngelScript / GDScript 全面本地性能复测报告

- 测试时间：2026-10-12，UTC+08:00
- 仓库分支：`angelscript-boundary-opt`
- 仓库提交：`e19dffbca60534dc7a032bb45caa54bb19f3f360`，另含当前工作区未提交的方案 B 强类型容器与派发优化
- Godot：`4.8.dev.custom_build.e19dffbca`，mono editor x86_64
- 基准工程：`modules/angelscript/tests/bench_project/`
- 原始数据：`docs/language-benchmark/data/three-language-comprehensive-2026-10-12.csv`
- 聚合数据：`docs/language-benchmark/data/three-language-comprehensive-2026-10-12-summary.csv`
- 完整日志：`docs/language-benchmark/data/three-language-comprehensive-2026-10-12-raw.log`
- 启动数据：`docs/language-benchmark/data/three-language-comprehensive-2026-10-12-startup.csv`

---

## 1. 执行摘要

### 最重要的结论

1. **C# 仍是纯计算和普通托管数据结构的绝对强项。** 算术、内部函数、类字段 best 分别为 **1.666、2.147、2.036 ns/op**；AngelScript 为 17.763、26.812、43.789，GDScript 为 33.599、78.036、151.133。
2. **AngelScript 的纯脚本计算持续快于 GDScript。** 三项中 AS 约为 GD 耗时的 **0.29–0.53 倍**，即快约 **1.9–3.5 倍**；但仍比 C# 慢 **10.7–21.5 倍**。
3. **Godot 动态 `Array` / `Dictionary` 仍是 GDScript 的主场。** AS 动态 Array 慢于 GD **13.0 倍**，Dictionary 慢 **28.6 倍**，Array 综合方法慢 **7.3 倍**。
4. **方案 B 的粗粒度批处理获得本轮唯一 AS 总冠军。** `Vector2Array.add_scaled_in_place` 的 AS best 为 **3.06 ns/元素**，C# 脚本循环为 22.43，GD 脚本循环为 48.11；AS 分别快约 **7.3 倍和 15.7 倍**。这反映 C++ 批处理优势，不代表单次 Vector2 运算的 AS VM 更快。
5. **强类型 Map 已超过 GDScript Dictionary。** `StringNameInt64Map` 的 AS best 为 **81.96 ns/op**，GD 类型稳定 Dictionary 为 98.19；AS 快约 **1.20 倍**。C# 原生泛型 Dictionary 仍以 42.89 最快。
6. **逐元素强类型 Int64 容器仍未超过 GD Packed 数组。** AS 64.91 vs GD 38.66 ns/op，说明每次 AS→C++ 方法调用仍是主要固定成本。
7. **引擎对象热点已充分收敛。** `get_child_count()`：C# 22.515、AS 36.055、GD 38.600 ns/op；AS 略快于 GD 约 7%，三者的边界增量几乎一致。
8. **动态派发：GD 最快，AS 第二，C# 最慢。** `Object.call` 为 GD 62.485、AS 96.150、C# 214.475 ns/op。AS 的派发计划缓存没有消除 Prepare/Execute/context 成本。

### 一句话选型

- **高密度算法、AI、复杂业务、原生集合：C#。**
- **场景树胶水、Godot 动态容器、动态调用：GDScript。**
- **热更新玩法脚本：AngelScript；纯脚本计算优于 GD，并应通过强类型 Map 和 C++ 批处理避开容器/边界短板。**

---

## 2. 测试环境

| 项目 | 值 |
|---|---|
| CPU | AMD Ryzen 7 7700，8C/16T，MaxClockSpeed 3801 MHz |
| 内存 | Windows 可见 99,765,200 KiB，约 95.1 GiB |
| OS | Windows 10 专业版 10.0.19045，Build 19045 |
| Godot | 4.8-dev custom mono editor x86_64 |
| C# | net8.0，Godot.NET.Sdk 4.7.0，Debug 程序集 |
| AngelScript | vendored 2.38 + 当前工作区方案 B |
| 运行模式 | `--headless`，三语言同一场景、同一进程、同一 Godot 可执行文件 |

C# 使用 Debug 程序集，是因为当前 editor 工程运行时实际加载 `.godot/mono/temp/bin/Debug/LangBench.dll`。CLR 仍会 JIT；本报告不把此前单独构建但未被场景加载的 Release DLL误标为正式被测配置。

---

## 3. 方法学与统计口径

### 3.1 采样

- 共测 **17 个 workload × 3 种语言 × 5 个独立进程 = 255 条正式记录**。
- 每个进程内，每项先执行 `iters / 100` 次预热，再重复 **7 次**。
- 计时使用 `Time.get_ticks_usec()`；普通负载只跨语言调用一次，内部循环完全留在目标语言中。
- `bench_dispatch` 是例外：它故意从 GD 驱动器逐次执行 `Object.call("bench_noop")`。
- 正式运行前 C# 构建为 0 warning / 0 error；AngelScript 测试为 **178 passed / 0 failed / 3570 assertions**。
- 255 条数据的所有 `round + workload` 分组均恰有 AS/CS/GD，且三语言校验和完全一致。

### 3.2 聚合

- `best`：5 个进程各自 `min_us` 的最小值。
- `median`：5 个进程各自进程内 median 的中位数。
- `worst`：5 个进程各自 `max_us` 的最大值。
- `CV`：5 个进程内 median 的变异系数，用来观察跨进程稳定性。
- `ns/op = μs × 1000 / iterations`。

best 接近稳态吞吐下界；median 更能代表本机典型运行。结论同时参考二者，不以单个 best 掩盖抖动。

### 3.3 负载分组

| 分组 | 负载 |
|---|---|
| 纯脚本/语言 | arith、calls、class、hoist |
| Godot 动态容器 | array、dictionary、array_methods |
| 数值与字符串 | packed、string、vector、color |
| 引擎边界 | engine、dispatch |
| 方案 B / 惯用高性能路径 | typed_i64、typed_vec2、typed_map、batch_sum |

---

## 4. 完整结果

单位：**ns/op，越小越好**。单元格为 `best / median`。

| 负载 | AS best / median | C# best / median | GD best / median | 最快 |
|---|---:|---:|---:|---|
| arith | 17.763 / 18.551 | 1.666 / 1.871 | 33.599 / 36.318 | **C#** |
| calls | 26.812 / 28.664 | 2.147 / 2.458 | 78.036 / 85.088 | **C#** |
| class | 43.789 / 46.662 | 2.036 / 2.219 | 151.133 / 159.978 | **C#** |
| array | 1127.190 / 1237.330 | 152.170 / 167.150 | 86.700 / 101.060 | **GD** |
| dictionary | 1625.190 / 1708.020 | 477.240 / 756.070 | 56.860 / 63.830 | **GD** |
| packed | 78.540 / 88.890 | 3.460 / 4.900 | 39.330 / 44.890 | **C#** |
| array_methods | 1186.170 / 1220.360 | 186.120 / 198.470 | 161.990 / 172.700 | **GD** |
| string | 1154.125 / 1273.375 | 184.250 / 301.250 | 260.125 / 301.375 | **C#** |
| vector | 60.731 / 63.931 | 6.074 / 6.615 | 13.922 / 14.741 | **C#** |
| color | 61.217 / 64.776 | 14.203 / 15.177 | 14.423 / 15.336 | **C#** |
| engine | 36.055 / 40.035 | 22.515 / 25.295 | 38.600 / 43.355 | **C#** |
| hoist | 10.930 / 12.470 | 1.370 / 1.420 | 17.120 / 19.010 | **C#** |
| typed_i64 | 64.910 / 74.970 | 3.310 / 3.620 | 38.660 / 43.330 | **C#** |
| typed_vec2 | **3.060 / 4.830** | 22.430 / 24.980 | 48.110 / 54.000 | **AS** |
| typed_map | 81.960 / 91.870 | **42.890 / 47.600** | 98.190 / 109.320 | **C#** |
| batch_sum | 30.670 / 34.230 | 2.870 / 3.330 | 28.700 / 31.710 | **C#** |
| dispatch | 96.150 / 106.280 | 214.475 / 221.425 | **62.485 / 68.105** | **GD** |

### 4.1 AngelScript 相对倍率（best）

| 负载 | AS / GD | AS / C# | 解读 |
|---|---:|---:|---|
| arith | 0.53× | 10.66× | AS 快于 GD，远慢于 C# |
| calls | 0.34× | 12.49× | AS 快于 GD 约 2.9× |
| class | 0.29× | 21.51× | AS 快于 GD 约 3.5× |
| array | 13.00× | 7.41× | AS 动态 Array 短板 |
| dictionary | 28.58× | 3.41× | 本轮最大 AS/GD 差距 |
| packed | 2.00× | 22.70× | AS 逐项 thunk；C# 为 CLR long[] |
| array_methods | 7.32× | 6.37× | 动态容器方法短板 |
| string | 4.44× | 6.26× | AS Godot String 拼接短板 |
| vector | 4.36× | 10.00× | 单项运算仍受 callfunc 影响 |
| color | 4.24× | 4.31× | C# 与 GD 几乎相同 |
| engine | 0.93× | 1.60× | AS 略快于 GD，均与 C# 同量级 |
| hoist | 0.64× | 7.98× | AS 解释器循环优于 GD |
| typed_i64 | 1.68× | 19.61× | 原生存储仍被逐项调用抵消 |
| typed_vec2 | 0.06× | 0.14× | C++ 批处理使 AS 获胜 |
| typed_map | 0.83× | 1.91× | AS 固定 Map 超过 GD |
| batch_sum | 1.07× | 10.69× | AS 与 GD 接近，构建阶段占主导 |
| dispatch | 1.54× | 0.45× | AS 快于 C#，慢于 GD |

---

## 5. 分组分析

### 5.1 纯计算、函数和类

C# 通过 JIT 生成本机代码，三项均约 2 ns/op。AngelScript 是解释执行，但在整数循环、脚本函数和脚本对象字段方面持续优于 GDScript：

- arith：AS 比 GD 快约 1.9×；
- calls：快约 2.9×；
- class：快约 3.5×。

因此，若热更新玩法逻辑主要是本地标量和类状态，不频繁触碰 Godot 动态值，AS 具有实际优势。不过其结果仍比 C# 慢一个数量级以上。

### 5.2 动态 Array 与 Dictionary

GDScript 对 `Variant` 和内建容器拥有 VM 内联/缓存路径。AS 即使已有原生 thunk，仍需在脚本 VM 和 C++ 容器间逐次调用：

- Array：AS 1127 vs GD 86.7，慢 13×；
- Dictionary：AS 1625 vs GD 56.86，慢 28.6×；
- Array 方法组合：慢 7.3×。

结论没有变化：**不要在 AS 热循环内使用动态 Array/Dictionary 做逐元素业务。**

### 5.3 强类型容器和方案 B

#### Int64Array

强类型存储消除了 Variant 元素，但 push/get 每次仍是一次 AS→C++ 调用，因此 64.91 ns/op 仍慢于 GD PackedInt64Array 的 38.66。仅替换存储，不足以克服逐项 callfunc 成本。

#### Vector2 批处理

AS `add_scaled_in_place()` 将整个循环移到 C++，只跨边界一次；测得 3.06 ns/元素。C# 和 GD 对照把循环写在各自语言中，因此分别为 22.43 和 48.11。

这是**不同层次实现的业务等价比较**：它证明 API 粗粒度设计可让 AS 业务负载明显超过 GD/C# 脚本循环，但不证明 AS VM 的 Vector2 指令比 CLR 更快。

#### StringNameInt64Map

固定 `HashMap<StringName,int64_t>` 避免 Variant 键和值，AS 达到 81.96 ns/op，优于 GD Dictionary 的 98.19。C# `Dictionary<StringName,long>` 仍以 42.89 最快。

#### Batch sum

AS `sum()` 已在 C++ 执行，但负载仍包含 N 次 `push_back`，因此端到端 30.67，略慢于 GD 28.70。若输入通过一次 `assign_packed()` 导入或由宿主直接生成，求和阶段的批处理收益会更明显。

### 5.4 String、Vector2、Color

- String：AS 1154，GD 260，C# 184。三方字符串类型并非同一实现；AS 的 Godot String 反复拼接仍是短板。
- Vector2：C# 6.07、GD 13.92、AS 60.73。AS 的值已原生存储，但每次运算符仍通过原生调用边界。
- Color：C# 14.20 与 GD 14.42 几乎持平；AS 为 61.22。

### 5.5 引擎调用与调用提取

`engine - hoist` 可粗估循环内一次 `get_child_count` 的附加成本：

- AS：36.055 − 10.930 ≈ **25.125 ns**；
- C#：22.515 − 1.370 ≈ **21.145 ns**；
- GD：38.600 − 17.120 ≈ **21.480 ns**。

三者增量非常接近，说明 AS 的热点引擎 thunk 已基本把边界压到 GD/C# 同一量级。AS engine 略快于 GD，剩余差异主要来自各自循环成本。

### 5.6 动态 Object.call

GD 62.49、AS 96.15、C# 214.48。AS 已缓存方法解析和标量编组布局，但每次仍需要 context Prepare/Execute/Unprepare、返回 Variant 和调试/Profiler 兼容路径；所以布局缓存只带来小收益。

---

## 6. 稳定性、启动时间与异常值

### 6.1 启动/加载

10 次 `BENCH_SMALL=1` 混合工程端到端 wall-clock：

- min：**779.59 ms**
- median：**799.32 ms**
- P90：**908.14 ms**
- max：**913.88 ms**

这是 Godot headless 启动、项目加载、C# 程序集、AS/GD 脚本、极小负载和退出的组合时间，不能拆成单语言排名。

### 6.2 方差

多数负载的跨进程 median CV 在 2–8%。值得注意：

- C# string：27.41%，受字符串分配和 GC/调度影响明显；
- AS typed_vec2：13.68%，绝对耗时极短，微秒计时量化噪声被放大；
- C# batch_sum：13.09%；
- C# array/calls/engine：约 11%。

C# packed 的 worst 达 222.28 ns/op，但 best/median 为 3.46/4.90，说明个别重复遭遇系统或分配抖动，不能拿 worst 代表稳态吞吐。

### 6.3 退出告警

5 个正式进程每轮均报告 **16 个 ObjectDB instances leaked**。基准中的 AS `bench_engine`/`bench_hoist` 每次调用会新建临时 Node，而模块当前不暴露非 RefCounted `free()`；这些泄漏发生在退出检查，不位于负载计时区间，但属于基准已知缺陷。

---

## 7. 公平性与解释边界

1. **`bench_packed` 并非同一容器：** AS/GD 用 Godot PackedInt64Array，C# 用 CLR `long[]`，C# 结果代表其惯用原生数组上限。
2. **typed/batch 是业务等价而非底层同构：** AS typed_vec2 把循环放在 C++，C#/GD 对照在语言层循环；结果用于评价 API 设计，而不是 VM 指令速度。
3. **字符串实现不同：** System.String、Godot String、GDScript String 的分配和拼接策略不同。
4. **AS engine 接收对象不同：** 因当前 AS 无法把 `as_self()` 下转为 Node，AS 使用临时 Node；三方返回的 child count 都为 0，但对象身份不相同。
5. **固定执行顺序：** 工作负载及语言顺序未随机化；5 个独立进程与预热降低但不能消除顺序效应。
6. **未锁频、未绑核：** 当前数据用于同机相对比较，不应用作跨机器绝对阈值。
7. **工作区版本：** 新强类型容器代码尚未提交，提交哈希只标识基底；报告明确绑定当前工作区状态。
8. **微基准不是整帧性能：** 实际游戏还受渲染、物理、资源、GC、信号和架构影响。

---

## 8. 工程选型建议

| 业务场景 | 首选 | 原因 |
|---|---|---|
| 密集算法、AI、生成、复杂业务 | C# | 纯计算和普通对象快一个数量级 |
| 场景树胶水、编辑器友好逻辑 | GDScript | 集成直接、动态派发最快 |
| Godot Array/Dictionary 密集处理 | GDScript | 内建容器优势巨大 |
| 热更新的本地标量/类逻辑 | AngelScript | 比 GD 快约 1.9–3.5× |
| 热更新的批量 Vector/实体更新 | AngelScript + C++ 批处理 | 本轮可超过 GD/C# 脚本循环 |
| StringName→标量热表 | AS 固定 Map 或 C# Dictionary | 二者均优于 GD Dictionary 对照 |
| 高频逐元素数值访问 | C# 原生数组或 GD Packed | AS 每项 callfunc 仍偏贵 |
| 高频引擎 API | 任一语言均应批量化/提取 | 单次边界增量已接近，减少次数最有效 |

### 推荐混合架构

- GDScript：场景组织、信号、UI、Godot 动态容器胶水。
- C#：计算密集、长期复杂业务、原生数据结构。
- AngelScript：需要热更新的玩法规则；API 设计采用强类型存储、固定 Map 和一次调用处理整批数据。

---

## 9. 验证状态

- C# Debug 构建：成功，0 warning / 0 error。
- 冒烟：17 workload × 3 语言 = 51 行，无 SCRIPT ERROR。
- AngelScript 测试：178 passed / 0 failed / 3570 assertions。
- 正式数据：255 / 255 条，所有三语言校验和匹配。
- 此前同一工作区全量 Godot 测试：1603 passed / 1 failed / 3 skipped；唯一失败是 GDScript 输出快照 `call_native_static_method.gd` 中出现 stray Node 行，不是 AngelScript 断言失败。本轮未重复全量测试，只复跑了 AngelScript 子集。

---

## 10. 复现命令

```powershell
cd D:\GitWorkspace\GodotWorkspace\godot

# C# 场景实际加载的 Debug 程序集
dotnet build .\modules\angelscript\tests\bench_project\LangBench.csproj -c Debug

# 单个正式进程：内部每项预热并重复 7 次
.\bin\godot.windows.editor.x86_64.mono.console.exe `
  --headless `
  --path .\modules\angelscript\tests\bench_project

# AngelScript 正确性测试
.\bin\godot.windows.editor.x86_64.mono.console.exe `
  --headless --test --test-case="*[AngelScript]*"

# 启动/加载冒烟
$env:BENCH_SMALL='1'
.\bin\godot.windows.editor.x86_64.mono.console.exe `
  --headless `
  --path .\modules\angelscript\tests\bench_project
Remove-Item Env:BENCH_SMALL
```

复测时应至少运行 5 个独立正式进程，并验证每个 `round + workload` 的 AS/CS/GD 校验和唯一。
