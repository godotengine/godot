# C# / AngelScript / GDScript 三语言性能对比报告

- 测试时间：2026-10-14 13:13–13:20（UTC+08:00）
- 仓库提交：`2bfae432e8763a396a5999053403fcf04d5e6738`（工作区另含未提交的本地补丁与测试）
- 引擎：Godot `4.8.dev.mono.custom_build`，mono editor x86_64，MSVC 14.51
- 被测二进制 SHA256：`bcc3ce1556f46b190d0805c225564249859bff0c648a6dd6cae38bd0c4459829`
- C# 程序集：Debug（`LangBench.csproj`）
- 基准工程：`modules/angelscript/tests/bench_project/`（算法、迭代数、预热、校验和均未改动）
- 原始数据：[samples.csv](<data/three-language-2026-10-14/samples.csv>)、[summary.csv](<data/three-language-2026-10-14/summary.csv>)、[analysis.json](<data/three-language-2026-10-14/analysis.json>)
- 运行元数据：[metadata.json](<data/three-language-2026-10-14/metadata.json>)、[environment.json](<data/three-language-2026-10-14/environment.json>)、[result.json](<data/three-language-2026-10-14/result.json>)

> **先说三条最重要的结论。**
> 1. **C# 仍是纯计算与托管数据结构的绝对强项**，在 17 项中 11 项最快；GDScript 在 5 项最快（全部是 Godot 动态容器/字符串/派发）；AngelScript 只在 1 项最快（`Vector2Array` 批处理）。
> 2. **AngelScript 的纯脚本计算显著快于 GDScript**（arith/class/calls/hoist 约快 2.3–5.1 倍），但**在 Godot 动态容器、几何值类型和动态派发上明显落后**（vector 3.97×、color 3.60×、dictionary 1.76×、dispatch 1.63×）。
> 3. **本轮发现一个严重的方法学问题：AngelScript 的绝对性能对二进制布局极其敏感。** 同一份 AS 运行时代码，仅因无关测试代码改变布局，AS 耗时即可相差 **1.0–1.8 倍**（中位 1.17×），而 GDScript 与 C# 稳定在 ±5%。因此本报告给出 AS/GD 的**区间**而非单点值；上一份报告中的 AS 绝对值与本次不可直接比较。

---

## 1. 测量条件（必须先读）

### 1.1 空闲门未通过：这是带负载的运行

本轮沿用受限定的 60 秒墙钟空闲门（连续 3 个有效样本总 CPU < 25%）：

| 项目 | 结果 |
|---|---|
| 空闲门通过数 | **0 / 6** |
| 门内 CPU 中位数（353 个样本） | 29.13%（范围 18.08%–100%） |
| **运行期**总 CPU 中位数（121 个样本） | **36.15%**（范围 29.85%–53.75%） |
| 运行期运行队列中位数 / 最大 | 0 / 57 |
| 每进程墙钟 / CPU 时间 | 21.19–21.56 s / 20.98–21.41 s |

严格门禁首次尝试在第一个进程前就以 `idle_unavailable` 停止（[拒绝证据](<data/three-language-refused-2026-10-14/result.json>)，58 个样本最低 30.0% CPU）。因此本报告的数据来自预先声明的降级方案：**门继续采样并记录，但不再中止**，并在下文中把运行标注为"带负载"，绝不称为空闲或稳定。

**为什么这仍然可用：** 三种语言在**同一个进程、同一场景、同一时刻**依次执行，主指标采用**同进程内逐项配对**的语言比值。全局负载会同时推高三种语言，因此对"谁更快"的判断影响远小于对绝对值的影响。另用一次**门全部通过**的低负载历史运行做交叉校验（见 §5.1）。

### 1.2 采样与聚合

- **6 个独立进程 × 17 负载 × 3 语言 = 306 条有效记录**，全部进程 exit 0，亲和性均为 `0xf000`，优先级均为 AboveNormal（32768）。
- 每进程内每项先预热 `iters/100`，再重复 **7 次**；报告使用进程内 median。
- 计时用 `Time.get_ticks_usec()`；普通负载只跨语言调用一次，循环完全在目标语言内部。
- `bench_dispatch` 例外：由 GD 驱动器逐次执行 `Object.call("bench_noop")`。
- 三语言的迭代次数与校验和在每进程内完全一致，且跨 6 个进程一致（解析器强制校验，不一致直接失败）。
- `best` = 6 个进程各自 `min_us` 的最小值；`median` = 6 个进程各自进程内 median 的中位数；`ns/op = μs×1000/iters`。
- 比值区间：对 6 个配对差值做 10000 次整进程 bootstrap（固定 seed `20261014`），取 2.5/97.5 分位。

固定语言顺序为 `as → cs → gd`（驱动器未改动，未做顺序随机化）。

---

## 2. 完整结果

单位 **ns/op，越小越好**；单元格为 `best / median`。

| 负载 | AS best/median | C# best/median | GD best/median | 最快 |
|---|---:|---:|---:|---|
| arith | 9.81 / 10.02 | 1.72 / 1.75 | 32.79 / 33.38 | **C#** |
| calls | 19.38 / 19.79 | 2.19 / 2.30 | 75.41 / 77.31 | **C#** |
| class | 27.81 / 29.65 | 2.09 / 2.13 | 147.60 / 151.08 | **C#** |
| hoist | 7.15 / 8.01 | 1.44 / 1.48 | 17.80 / 19.19 | **C#** |
| array | 136.43 / 143.63 | 151.30 / 156.24 | 87.35 / 92.50 | **GD** |
| dictionary | 109.99 / 115.59 | 483.19 / 781.87 | 57.07 / 65.70 | **GD** |
| array_methods | 179.88 / 183.39 | 185.78 / 197.47 | 160.23 / 171.91 | **GD** |
| string | 332.12 / 361.88 | 199.38 / 352.75 | 273.25 / 293.94 | **GD** |
| packed | 57.39 / 59.33 | 3.51 / 5.08 | 39.34 / 43.02 | **C#** |
| vector | 56.77 / 57.94 | 6.32 / 6.52 | 14.20 / 15.25 | **C#** |
| color | 56.48 / 57.95 | 14.25 / 14.91 | 14.43 / 16.18 | **C#** |
| engine | 29.36 / 31.28 | 23.29 / 23.91 | 40.41 / 44.06 | **C#** |
| dispatch | 98.78 / 103.91 | 209.94 / 213.77 | 61.69 / 63.69 | **GD** |
| typed_i64 | 52.72 / 55.62 | 3.49 / 3.75 | 40.38 / 42.59 | **C#** |
| typed_vec2 | 2.58 / 3.42 | 23.11 / 24.55 | 47.13 / 48.68 | **AS** |
| typed_map | 69.76 / 72.82 | 46.47 / 48.83 | 99.21 / 108.52 | **C#** |
| batch_sum | 25.93 / 27.14 | 2.96 / 3.17 | 29.57 / 31.80 | **C#** |

### 2.1 语言倍率（含布局不确定带）

`AS/GD < 1` 表示 AS 更快。**"布局带"是本报告特有的不确定区间**：AS 侧因二进制布局可再放大至多该倍数（见 §6），GD/C# 无此问题。

| 负载 | AS/GD | 95% 区间 | **AS/GD 布局带** | C#/GD | AS/C# |
|---|---:|---|---:|---:|---:|
| arith | 0.301 | 0.298–0.303 | 0.30–0.45 | 0.053 | 5.74 |
| calls | 0.256 | 0.253–0.260 | 0.26–0.36 | 0.030 | 8.66 |
| class | 0.196 | 0.190–0.205 | 0.20–0.28 | 0.014 | 13.70 |
| hoist | 0.427 | 0.388–0.444 | 0.43–0.62 | 0.077 | 5.46 |
| array | 1.564 | 1.529–1.669 | 1.56–1.76 | 1.714 | 0.92 |
| dictionary | 1.759 | 1.649–1.944 | 1.76–1.88 | 11.978 | 0.15 |
| array_methods | 1.067 | 1.043–1.100 | 1.07–1.18 | 1.131 | 0.93 |
| string | 1.217 | 1.193–1.394 | 1.20–1.22 | 1.200 | 1.05 |
| packed | 1.372 | 1.340–1.441 | 1.37–1.70 | 0.118 | 11.58 |
| vector | 3.974 | 3.786–4.066 | 3.97–4.34 | 0.431 | 8.91 |
| color | 3.603 | 3.501–3.797 | 3.60–3.80 | 0.926 | 3.93 |
| engine | 0.715 | 0.697–0.726 | 0.72–0.83 | 0.543 | 1.31 |
| dispatch | 1.626 | 1.596–1.653 | 1.51–1.63 | 3.349 | 0.48 |
| typed_i64 | 1.311 | 1.280–1.347 | 1.31–1.62 | 0.090 | 14.90 |
| typed_vec2 | 0.070 | 0.062–0.088 | 0.07–0.08 | 0.507 | 0.13 |
| typed_map | 0.676 | 0.666–0.689 | 0.68–0.81 | 0.453 | 1.50 |
| batch_sum | 0.854 | 0.826–1.025 | 0.85–1.09 | 0.102 | 8.64 |

**几何平均（17 项，跨负载的总体印象）：** C#/GD = **0.335**（C# 约快 3.0 倍）；AS/GD = **0.810**（AS 约快 1.23 倍，但受布局带影响）；AS/C# = **2.416**。

**逐项胜负：**

- C# 最快 **11 项**：arith、calls、class、hoist、packed、vector、color、engine、typed_i64、typed_map、batch_sum
- GDScript 最快 **5 项**：array、array_methods、dictionary、string、dispatch
- AngelScript 最快 **1 项**：typed_vec2（C++ 批处理路径）
- AS 数值上不慢于 GD 的共 **8 项**：arith、calls、class、hoist、engine、typed_map、typed_vec2、batch_sum（其中 batch_sum 区间跨 1，判为不确定）

---

## 3. 分组分析

### 3.1 纯脚本计算：AS 稳定领先 GD，但都比 C# 慢一个数量级

| 负载 | AS | GD | AS/GD | C# |
|---|---:|---:|---:|---:|
| arith | 10.02 | 33.38 | **0.30** | 1.75 |
| calls | 19.79 | 77.31 | **0.26** | 2.30 |
| class | 29.65 | 151.08 | **0.20** | 2.13 |
| hoist | 8.01 | 19.19 | **0.43** | 1.48 |

AngelScript 在这四项上比 GDScript 快 **2.3–5.1 倍**，说明若热更新玩法逻辑以本地标量、脚本函数调用和脚本对象字段为主，AS 具备实际优势。C# 由 JIT 生成本机代码，仍快 AS **5.5–13.7 倍**。

### 3.2 Godot 动态容器：GDScript 的主场，AS 仍未追平

| 负载 | AS | GD | AS/GD | C# | 说明 |
|---|---:|---:|---:|---:|---|
| array | 143.63 | 92.50 | 1.56 | 156.24 | AS 已快于 C# |
| dictionary | 115.59 | 65.70 | 1.76 | 781.87 | 注意 C# 用的是 `Godot.Collections.Dictionary`（托管绑定对象），因此很慢 |
| array_methods | 183.39 | 171.91 | 1.07 | 197.47 | AS 已接近 GD |
| string | 361.88 | 293.94 | 1.22 | 352.75 | 三方字符串实现不同 |

与更早的报告相比，AS 在动态容器上的差距已大幅收窄（历史报告 AS/GD 为 13×/28.6×/7.3×，本轮为 1.56×/1.76×/1.07×），这是原生索引访问器与原生方法 thunk 的直接结果。**但结论方向不变：不要指望 AS 在动态 Array/Dictionary 的逐元素热循环里赢过 GDScript。**

注意 `bench_dictionary` 的 C# 实现用的是 `Godot.Collections.Dictionary`（托管互操作对象），每轮中位数 781.87 ns 且 best 483.19，方差大；这反映的是 API 选择，不代表 CLR 字典的速度。C# 若用原生 `Dictionary`（如 `typed_map` 项所示）则快得多。

### 3.3 数值与字符串：AS 的几何值类型仍是短板

| 负载 | AS | GD | C# | AS/GD |
|---|---:|---:|---:|---:|
| packed | 59.33 | 43.02 | 5.08 | 1.37 |
| vector | 57.94 | 15.25 | 6.52 | **3.97** |
| color | 57.95 | 16.18 | 14.91 | **3.60** |
| string | 361.88 | 293.94 | 352.75 | 1.22 |

`Vector2`/`Color` 的**存储**已是原生布局，但每次运算符仍要跨一次 AS↔C++ 调用边界；C# 的 `Vector2` 是纯托管结构体，GDScript 的内建类型走 VM 内联路径，因此两者都比 AS 快 3.5–4 倍。这与历史结论一致，属于已知结构性问题。

`packed`：AS/GD 为 1.37（AS 慢 37%），C# 用 CLR `long[]` 因此极快（5.08）。

### 3.4 引擎调用与调用提取

| 负载 | AS | GD | C# |
|---|---:|---:|---:|
| engine（循环内调用 `get_child_count`） | 31.28 | 44.06 | 23.91 |
| hoist（提到循环外） | 8.01 | 19.19 | 1.48 |
| **差值（单次边界增量估计）** | **22.70** | **25.06** | **22.43** |

三者的**单次引擎边界增量几乎相同（22.4–25.1 ns）**。这是本报告中最稳健的一条结论之一：AS 的原生热点 thunk 已把单次引擎调用成本压到与 GD/C# 同一量级。AS 的 `engine` 绝对值快于 GD，主要来自它更快的脚本循环底噪；真正的优化方向是**减少调用次数**，而不是继续压低单次边界。

### 3.5 动态派发：GD 最快，AS 居中，C# 最慢

`Object.call("bench_noop")` 逐次派发：**GD 63.69 / AS 103.91 / C# 213.77 ns**。

- GD 最快（VM 内直接解析）；C# 最慢（跨托管边界 + 每次装箱/编组）；
- AS 的 `ASCallPlan` 缓存只省掉方法名解析与标量编组布局分析，每次仍需 context `Prepare/Execute/Unprepare`、返回值转 `Variant` 等固定成本，因此仍比 GD 慢 1.63 倍，但比 C# 快约 2.06 倍。

### 3.6 强类型容器与批处理

| 负载 | AS | GD | C# | AS/GD |
|---|---:|---:|---:|---:|
| typed_i64（逐元素强类型 Int64） | 55.62 | 42.59 | 3.75 | 1.31 |
| typed_map（StringName→int64） | 72.82 | 108.52 | 48.83 | **0.68** |
| typed_vec2（Vector2Array 批处理） | 3.42 | 48.68 | 24.55 | **0.07** |
| batch_sum（sum() 批处理） | 27.14 | 31.80 | 3.17 | 0.85（区间跨 1） |

- **AS 固定 Map 已明显超过 GD 类型化 Dictionary**（快约 1.49 倍），但仍慢于 C# 泛型字典。
- **`typed_vec2` 是 AS 唯一的总冠军**：把整个循环放进 C++，只跨一次边界，比 GD 快 14 倍、比 C# 快 7 倍。这是**API 粗粒度设计的胜利，不是 AS VM 的胜利**，不可推广为"AS 的 Vector2 运算更快"（对照见 §3.3）。
- `typed_i64` 逐元素 push/get 每次仍是一次 AS→C++ 调用，因此仍慢于 GD Packed 数组。

---

## 4. 稳定性

6 个进程内 median 的变异系数中位数：**AS 2.5%、C# 2.5%、GD 2.2%**，整体稳定。个别项偏高：

| 语言 | 最高 CV | 原因 |
|---|---|---|
| AS | typed_vec2 17.9%、batch_sum 13.8%、string 9.0% | 绝对耗时极短（3–27 ns），微秒计时量化噪声被放大 |
| C# | string 19.5%、typed_i64 12.3%、batch_sum 11.4% | 字符串分配/GC、以及托管↔原生边界抖动 |
| GD | dictionary 6.9%、vector 4.8% | 相对平稳 |

结论：**不要用 best 值代表稳态**，也不要用 worst 值判断回退。本报告同时给出 best 与 median。

---

## 5. 与前次对比（谨慎）

### 5.1 与低负载历史运行交叉校验

最近一次**门全部通过**的三语言测量是 2026-10-14 09:36 的 A/B 实验 B 臂（同一提交、同一工程、6 进程、12/12 门通过）。用它作为低负载参照：

| 语言 | 本轮 median ÷ 低负载 median（17 项） | 范围 |
|---|---:|---|
| C# | **1.03** | 1.00–1.24 |
| GDScript | **1.05** | 0.91–1.21 |
| AngelScript | **0.80** | 0.53–1.04 |

C# 与 GD 在两次会话间只差 3–5%，说明**本轮负载与上轮低负载对两者的影响都很小**；但 AS 系统性快了 20%（arith 甚至快 47%）。若只看这张表，很容易误判成"AS 优化又生效了"。**下一节说明为什么这主要是二进制布局效应。**

### 5.2 与更早报告（2026-10-12）的差异

不应把本报告的绝对值与 [three-language-comprehensive-2026-10-12.md](three-language-comprehensive-2026-10-12.md) 直接相减，原因有二：**（a）** 该报告基于不同提交且工作区含未提交实现；**（b）** 本轮证明 AS 绝对值本身存在 1.0–1.8× 的构建敏感度。可以说的方向性变化是：AS 在动态 Array/Dictionary 上的差距已从十余倍收窄到 1.1–1.8 倍，`typed_map` 从落后 GD 变为领先 GD。

---

## 6. 关键方法学发现：AngelScript 性能对二进制布局极其敏感

### 6.1 现象

在核对"AS 为何比上轮快 20%"时，我把当前构建与保留的对照二进制（`.superpowers/bench-ab-20261013/candidate`，提交 `8c0143716d`）**在同一次会话内交替运行**，用同一工程、同一 C# 程序集、同一批负载：

| 对照运行（每次均为 4 进程交替） | AS arith 当前÷保留 | GD arith | C# arith |
|---|---:|---:|---:|
| 初始构建（补丁＋新测试＋测试断言） | **1.85×** | 0.98× | 0.99× |
| 还原 `as_builder` 补丁 | 1.72× | 1.03× | 0.99× |
| 再移除两个新测试文件 | 1.46× | 1.02× | 1.01× |
| 反向顺序重跑（排除时间漂移） | 1.51× | 1.02× | 1.00× |
| 强制重编 AS 虚拟机目标文件 | 1.45× | 0.96× | 1.00× |
| 仅重新链接同一批目标文件 | 1.51× | 1.03× | 1.00× |
| 重建最终配置 | 1.81× | 0.94× | 1.01× |
| **源码完全对齐保留版本**（撤销最后一行测试断言） | **0.99×** | 0.99× | 1.00× |

**排除的假设：** 编译标志（逐键比对完全一致）、工具链（三个二进制的 PE 链接器版本均为 14.51，Rich header 产品计数完全相同）、AS 虚拟机目标文件（强制重编后现象不变）、注册期补丁、新增测试文件、运行顺序与时间漂移（反向重跑结论不变）、单纯重新链接（结论不变）。

**这是确定性的布局效应，不是逐次运行的随机噪声：** 同一二进制的重复对照复现到约 2% 以内（同一构建两次测得 arith 14.01 / 13.78），而"有该测试断言"的 7 次构建稳定落在 11.2–14.4 ns、"无该断言"的 3 次测量稳定落在 20.3–21.2 ns。换言之，**只要源码集合固定，结果可复现；但一处与基准无关的测试改动就会确定性地把 AS 推到另一个档位。** 另有一次同配置独立运行（[带负载运行](<data/three-language-busy-2026-10-14/summary.csv>)，更早的构建）测得 AS arith 10.10 ns，与本轮 10.02 ns 一致，说明"快档位"本身也可跨构建复现。

**唯一能翻转现象的是一个只在 `--test` 下运行的测试断言。** 它在被测基准中根本不执行，却改变了 AS 的链接布局：

| 工作负载类型 | 布局放大系数（AS） |
|---|---:|
| 纯脚本 VM 循环（arith、hoist、class、calls、batch_sum） | **1.28–1.51** |
| 逐元素容器（packed、typed_i64、typed_map、array） | 1.07–1.24 |
| 几何/字符串/派发（color、dictionary、string、vector、dispatch） | 0.93–1.09 |

机制上最合理的解释是**代码布局效应**：AS 的字节码解释器是一个巨型 `switch` 循环，其位置相对指令缓存/分支预测器的对齐会显著影响吞吐；把无关代码加进 `tests/test_main.cpp` 会整体平移该函数的布局。**我没有把这条机制做成定论**——但"同一份 AS 代码在不同布局下吞吐相差 1.0–1.8 倍"是 8 次对照实验重复得到的事实。

### 6.2 影响与正确做法

1. **AS 的绝对值不是源码的稳定属性**，而是"源码集合＋具体链接布局"的属性；同一源码集合内可复现，但任何无关改动都可能换档。跨构建、跨报告比较 AS 绝对值是不安全的。
2. **AS/GD 与 AS/C# 的比值同样继承这个不确定度**，上表 §2.1 已给出每个负载的布局带。相对而言，容器/几何/派发类负载的布局带较窄（≤1.2），纯计算类最宽（可达 1.5）。
3. **C# 与 GDScript 未观测到此现象**（各对照中比值均落在 0.9–1.2，多数为 1.00），因此"GD↔C#"的结论是本报告中最可靠的部分。
4. **对后续 AS 优化评测的建议：** 比较两个优化版本时，除了源码差异，"两次构建的布局恰好不同"也会贡献真实差异。要么把两个版本构建进**同一个二进制**（运行期开关），要么对每个臂构建**多个布局样本**并报告分布，而不是各构建一次就下结论。此前基于双二进制的 A/B 结论方向可信，但幅度应视为包含布局不确定度。

---

## 7. 公平性与解释边界

1. **`bench_packed` 与 `bench_typed_i64` 不是同一容器**：AS/GD 用 Godot `PackedInt64Array`，C# 用 CLR `long[]`。C# 数字代表其惯用原生数组上限。
2. **C# 的 `bench_dictionary` 用 `Godot.Collections.Dictionary`**（托管互操作类型），并非 CLR 字典；其 781.87 ns 与巨大方差反映 API 选择。`bench_typed_map` 才展示 CLR 泛型字典的实力。
3. **`typed_vec2` 是业务等价、非底层同构**：AS 把循环放进 C++，C#/GD 在语言层循环。用于评价 API 设计，不用于评价 VM 指令速度。
4. **字符串实现不同**：`System.String`、Godot `String`、GDScript `String` 的分配与拼接策略不同。
5. **AS 的 `bench_engine` 接收对象不同**：AS 无法把 `as_self()` 下转为 `Node`，因此新建临时 `Node`；三方 `child_count` 都是 0，但对象身份不同。
6. **本报告是带负载测量**（门 0/6），运行期 CPU 中位 36.15%。结论依赖同进程配对，绝对值应视为上限偏悲观。
7. **未锁频、未绑核独占**：亲和性 `0xf000` 只限制迁移，不等于独享物理核；数据适合同机相对比较，不可当作跨机器绝对阈值。
8. **固定执行顺序**（as→cs→gd，未随机化）：6 个独立进程与预热降低但不能消除顺序效应。
9. **微基准不等于整帧性能**：真实项目还受渲染、物理、资源、GC、信号与架构影响。
10. **已知基准缺陷**：`bench_engine`/`bench_hoist` 每进程泄漏 16 个临时 `Node`（退出时告警），不在计时区间内，但属于既有问题，本轮未修改以保持可比性。

---

## 8. 工程选型建议

| 业务场景 | 首选 | 依据 |
|---|---|---|
| 密集算法、AI、生成、复杂业务逻辑 | **C#** | 17 项中 11 项最快，纯计算快 AS 5.5–13.7 倍 |
| Godot 动态 Array/Dictionary 逐元素处理 | **GDScript** | array 1.56×、dictionary 1.76×、array_methods 1.07× 领先 AS |
| 字符串拼接、动态 `Object.call` 派发 | **GDScript** | string 1.22×、dispatch 1.63× 领先 AS |
| 热更新的本地标量/类逻辑 | **AngelScript** | 比 GD 快 2.3–5.1 倍 |
| 热更新的批量几何/实体更新 | **AngelScript + C++ 批处理** | `typed_vec2` 比 GD 快 14 倍（API 设计收益） |
| StringName→标量热表 | AS 固定 Map 或 C# 泛型字典 | 两者均快于 GD 类型化 Dictionary |
| 高频逐元素数值访问 | **C# 原生数组 / GD Packed** | AS 每项仍有一次调用边界成本 |
| 高频引擎 API | **任何语言都应批量化/提取** | 三方单次边界增量已同为 22.4–25.1 ns，减少次数最有效 |

**推荐混合架构：** GDScript 负责场景组织、信号、UI 与动态容器胶水；C# 负责计算密集与长期复杂业务；AngelScript 负责需要热更新的玩法规则，并通过强类型容器与"一次调用处理一批数据"的 API 设计规避容器与几何短板。

---

## 9. 验证状态

- 最终二进制完整 AngelScript 回归：**192 / 192 用例、4761 / 4761 断言通过**（[日志](<data/three-language-2026-10-14/angel-script-tests.log>)），仅保留两个既有负面用例日志（host bridge error 3、缺少默认 factory）。
- C# Debug 程序集在测量前重建，`0 warning / 0 error`，以确保被测程序集与记录的源码一致（重建前旧程序集比源码旧）。
- 6 个进程全部 exit 0；三语言校验和与迭代数逐项一致；测量前后引擎、GodotSharp 依赖树、基准工程与 C# 组件哈希复核通过（[result.json](<data/three-language-2026-10-14/result.json>)）。
- 分析模块单元测试 7 项通过（矩阵完整性、校验和不一致拒绝、缺失/重复拒绝、比值与区间计算、边界增量、平局不强行判定胜者）。
- 未声称本轮使用空闲环境；未声称 AS 绝对值稳定。

## 10. 复现

```powershell
cd D:\GitWorkspace\GodotWorkspace\godot

# 1) 重建被测 C# 程序集（Debug）
dotnet build modules/angelscript/tests/bench_project/LangBench.csproj -c Debug

# 2) 分析模块单元测试
python -m unittest discover -s modules/angelscript/tests -p test_three_language_benchmark.py

# 3) 六进程三语言对比（门失败即中止的严格模式）
python modules/angelscript/tests/three_language_benchmark.py --runs 6 `
  --output docs/language-benchmark/data/three-language-repeat

# 高负载环境下按预定降级方案测量（门照常记录但不中止，报告须标注负载）
python modules/angelscript/tests/three_language_benchmark.py --runs 6 --gate-policy observe `
  --output docs/language-benchmark/data/three-language-repeat-busy

# 4) 布局敏感度对照（当前构建 vs 保留构建，交替运行）
python .superpowers/binary-control-20261014.py binary-control-repeat
```

运行器拒绝覆盖已有目录；`--gate-policy abort`（默认）在空闲门失败时返回码 2 且不产生任何性能记录。

## 11. 证据索引

- 结果：[summary.csv](<data/three-language-2026-10-14/summary.csv>)、[analysis.json](<data/three-language-2026-10-14/analysis.json>)、[samples.csv](<data/three-language-2026-10-14/samples.csv>)
- 环境与身份：[metadata.json](<data/three-language-2026-10-14/metadata.json>)、[environment.json](<data/three-language-2026-10-14/environment.json>)、[gates.json](<data/three-language-2026-10-14/gates.json>)、[processes.json](<data/three-language-2026-10-14/processes.json>)、[telemetry.jsonl](<data/three-language-2026-10-14/telemetry.jsonl>)
- 布局敏感度：[layout-band.json](<data/three-language-2026-10-14/layout-band.json>)、[对照脚本](<data/three-language-2026-10-14/binary-control.py>)、[全部对照结果](<data/three-language-2026-10-14/binary-control/>)
- 交叉校验：[cross-check-vs-idle.csv](<data/three-language-2026-10-14/cross-check-vs-idle.csv>)
- 严格门拒绝证据：[three-language-refused-2026-10-14](<data/three-language-refused-2026-10-14/result.json>)
- 6 个原始进程日志：`data/three-language-2026-10-14/01.log` … `06.log`

**残留不确定项（不隐藏）：** 布局敏感度的底层机制（指令缓存/分支预测对齐）未逐条证明；负载下 AS 的绝对值偏悲观；`bench_dictionary` 的 C# 实现使用托管包装容器，与"CLR 字典性能"不可混谈。
