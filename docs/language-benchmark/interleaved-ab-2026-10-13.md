# AngelScript 交错 A/B：组合优化复验

> 已完成：先提交，再运行固定六对交错 A/B，保留612条进程级负载统计。Array、Dictionary、String 的大幅组合收益再次出现；但测试期间系统依然高负载，不能宣布所有负载稳定追平或证明无回退。本阶段没有继续修改生产代码。

## 已完成的提交

本地提交 `8c0143716da82b133937d29df42ab3a4c02608c7`，分支 `develop`，说明 `perf(angelscript): optimize bindings and validate boundary costs`。提交范围为前两轮实现、回归、原始数据和报告；没有推送，也没有纳入无关未跟踪文件。本次复验新增脚本和结果未自动追加提交。

## 比较对象与问题

A 为优化前 `20530409fc50563e19954441aa1b81f47ffcee0e`；B 为上述优化提交。**比较的是两轮优化的总组合，不是 remaining-nine 单独的增量。** 中间阶段没有独立提交或归档二进制，不能把旧元数据的 launcher 哈希当作可复验的实际引擎。

两个版本用相同 MSVC、Windows SDK、editor/mono/tests 构建配置，实验 POD 关闭。A 在独立 detached worktree 编译，B 从主工作树编译后复制到独立目录；实际引擎、GodotSharp 依赖、原基准脚本及 C# Debug 组件记录 SHA256。历史原始算法没有改写，第三方 VM 不变。

## 预定测量方法

[执行计划](../../Documents/计划/2026-10-13-angelscript-interleaved-ab.md) 与 [runner](../../modules/angelscript/tests/interleaved_ab_benchmark.py)。六对按 AB、BA、AB、BA、AB、BA 顺序执行；每个进程完整运行原17项、每项7次重复和原预热。不同引擎运行同一份项目，不改语言顺序。只约束 benchmark 子进程：启动时先挂起，设置亲和性 mask `0xF000`（逻辑CPU12–15）和 AboveNormal 优先级，再恢复；不改系统电源方案、不结束用户程序、不使用实时优先级。

亲和性用于限制跨逻辑CPU迁移，并不等于独占物理核心。预定测试前至多等待60秒低负载窗口（连续三个样本总CPU<25%），不满足则显式记录。**实现偏差：当前 runner 实际上最多循环60次、每次至少等待1秒，再同步采集计数器；高负载下并不保证60秒墙钟上限。** 烟测已观察到样本间隔超过1秒，正式运行保留同一 runner，不中途修改或重新挑样本；报告用实际时间戳列出等待区间。这不改变被测算法，但无法视为严格相同的60秒冷却期。监测 GetSystemTimes 总CPU、可用内存及 PDH 系统运行队列、聚合频率代理。频率代理不是逐核有效频率，也不是完整调度轨迹。监测缺失记为 null，而不是0。

每个进程保留原始stdout/stderr、wall与CPU时间、实际亲和性/优先级；所有样本保留、不择优。校验 A/B/AS/GD/C# 每项迭代数和校验和一致，二进制和输入文件在测量后再次核验。

## 分析标准

主指标为每对 AS B/A 时间比的中位数，辅以两版本各自进程中位数范围、B阶段AS/GD时间比。预定 ±5% 实用区间；固定seed以整对重采样10000次估计95%中位比区间。仅六对样本，区间是探索性证据、并未做17项多重比较校正，不是性能保证。

同时列出 GDScript B/A 对照；绑定实现未改变GD，但两份独立构建及运行环境可能使其波动。归一化 AS B/A ÷ GD B/A 只作诊断，不替代原始指标、不能假定两语言对噪声响应完全一致。区间跨1或±5%不等于证明无回退。

## 正确性和环境记录

新增 runner 与 manifest 单测共11项通过，覆盖顺序、缺行/重复、跨版本checksum/迭代、零计时拒绝、配对估计、GD控制及实际组件哈希。交付前再次运行11项单测、Python语法编译、git diff --check，并核验报告/计划共32处本地文档链接，全部通过。直接实际引擎（不是console launcher）的启动烟测输出原51行，并确认子进程 mask `0xF000`、AboveNormal 生效。此烟测使用 BENCH_SMALL，不能算正式性能样本。

B提交后重建成功（SCons报告2分03秒），独立复制后的引擎通过完整AngelScript模块190项测试、3797条断言；host bridge error 3/缺少默认factory属于既有负面测试。B实际引擎 SHA256：`1d0a1f16bcd48f59ac144718eb1cba6f2d4dbba955109ae60848a75e1d420775`。编译设置比对确认两版本 `/O2`、`speed_trace`、single precision、无LTO、无debug symbols及其他核心编译/链接参数一致；A首次全量构建成功（SCons报告42分08秒）；manifest 已验证两版本编译设置、GodotSharp 依赖文件哈希一致，源码分别对应所述提交且 tracked 构建源码无改动。A模块回归179项、3678条断言全部通过。两份模块回归均已结束后才进行基准烟测与正式计时。

A实际引擎 SHA256：`208379aeeb809195cd80d3e43875d6cc6fd5a9d1046d6b2e7e81dd9df39f6480`。正式采样开始于2026-10-13 20:51:38 +08:00，时间戳覆盖约26.34分钟。全部12个进程退出0，实际亲和性均为0xF000、优先级均为32768（AboveNormal）；612行统计的负载、迭代数、语言和跨版本checksum均验证通过。结束后再次核验实际引擎、GodotSharp依赖、基准/C#组件及runner快照哈希，均未变。

**环境没有达到空闲目标。** 构建和回归均已结束，但正式12次低负载门通过数为0；225个运行期监测点的总CPU中位数100%（范围47.88%–100%），运行队列中位131、最大465。可用内存范围约47.94–49.11 GiB；机器已有高性能电源计划，本次未修改。

PDH频率代理全程3801 MHz/100%，这是聚合代理，不足以证明每核频率稳定。运行期实际采样间隔范围约1.01–2.99秒，各进程间隔中位1.35–1.96秒；等待门首末采样跨度90.88–99.66秒（还不含首个样本之前的等待），确认前述60秒墙钟上限没有被实现。监测和等待并非零成本，本次未隔离监测本身的开销。没有采集用户其他进程的完整调度归因，不能指称某个程序为噪声源，也不能反推历史样本的具体成因。

## 结果

### 全部原始负载

单位为每次迭代ns（不是每条底层指令）。A/B/GD列是各自六个进程中位数的中位数；“配对B/A”是六个对应比值的中位数，**不一定等于B列÷A列**。分类按预定95%区间是否完整低于0.95或高于1.05；“不确定”也可能具有改善倾向，并不等于没有收益。

| 负载 | A AS ns | B AS ns | 配对 B/A | 95%区间 | B GD ns | B AS/GD | 原始分类 |
|---|---:|---:|---:|---|---:|---:|---|
| arith | 42.30 | 21.32 | 0.503 | 0.472–0.520 | 37.00 | 0.58 | 改善信号 |
| array | 1301.35 | 220.20 | 0.170 | 0.145–0.181 | 116.66 | 1.89 | 改善信号 |
| array_methods | 1400.19 | 278.61 | 0.199 | 0.193–0.203 | 215.12 | 1.30 | 改善信号 |
| batch_sum | 53.54 | 46.31 | 0.863 | 0.834–1.001 | 35.92 | 1.29 | 不确定 |
| calls | 44.70 | 34.13 | 0.764 | 0.738–0.772 | 115.11 | 0.30 | 改善信号 |
| class | 63.28 | 51.81 | 0.819 | 0.794–0.861 | 196.66 | 0.26 | 改善信号 |
| color | 92.99 | 85.03 | 0.919 | 0.837–1.003 | 17.07 | 4.98 | 不确定 |
| dictionary | 1831.15 | 174.42 | 0.097 | 0.085–0.100 | 81.30 | 2.15 | 改善信号 |
| dispatch | 152.90 | 147.10 | 0.965 | 0.949–1.184 | 94.22 | 1.56 | 不确定 |
| engine | 61.92 | 52.88 | 0.866 | 0.842–0.984 | 56.19 | 0.94 | 不确定 |
| hoist | 28.50 | 15.34 | 0.534 | 0.487–0.652 | 21.73 | 0.71 | 改善信号 |
| packed | 141.27 | 98.54 | 0.706 | 0.675–0.827 | 52.02 | 1.89 | 改善信号 |
| string | 1420.00 | 457.50 | 0.319 | 0.299–0.340 | 366.75 | 1.25 | 改善信号 |
| typed_i64 | 117.09 | 95.78 | 0.818 | 0.789–1.023 | 51.49 | 1.86 | 不确定 |
| typed_map | 136.49 | 124.28 | 0.911 | 0.906–1.263 | 138.25 | 0.90 | 不确定 |
| typed_vec2 | 6.30 | 4.91 | 1.033 | 0.671–1.359 | 62.97 | 0.08 | 不确定 |
| vector | 93.78 | 85.97 | 0.913 | 0.885–0.923 | 16.69 | 5.15 | 改善信号 |

### 进程范围与未修改的GD对照

范围是六个进程中位数的最小/最大值，并非重复内极值。归一化列是每对 `(AS B/A)/(GD B/A)` 的中位数，只帮助识别共同变化，不作噪声精确扣除。

| 负载 | A AS范围 ns | B AS范围 ns | GD配对B/A | 归一化配对比 |
|---|---:|---:|---:|---:|
| arith | 40.29–45.31 | 21.14–21.43 | 0.905 | 0.553 |
| array | 1163.46–1329.20 | 164.22–224.39 | 0.999 | 0.167 |
| array_methods | 1383.54–1476.64 | 274.64–284.48 | 0.965 | 0.208 |
| batch_sum | 45.29–56.92 | 46.04–46.47 | 0.906 | 0.973 |
| calls | 44.16–45.76 | 32.30–34.92 | 0.995 | 0.746 |
| class | 58.02–63.62 | 49.38–52.01 | 0.987 | 0.830 |
| color | 79.29–94.13 | 72.50–86.14 | 0.956 | 0.994 |
| dictionary | 1772.64–1838.31 | 140.58–183.97 | 1.011 | 0.097 |
| dispatch | 105.97–154.84 | 145.60–148.97 | 1.007 | 0.952 |
| engine | 49.79–63.95 | 52.80–55.37 | 0.971 | 0.888 |
| hoist | 23.16–30.62 | 14.27–15.43 | 0.936 | 0.577 |
| packed | 106.75–146.12 | 97.76–104.55 | 0.938 | 0.764 |
| string | 1413.25–1493.00 | 398.50–520.62 | 1.081 | 0.303 |
| typed_i64 | 88.41–121.12 | 94.46–100.02 | 0.982 | 0.840 |
| typed_map | 95.32–137.96 | 121.56–125.42 | 0.992 | 0.945 |
| typed_vec2 | 2.90–6.80 | 4.13–7.40 | 0.964 | 0.979 |
| vector | 92.46–100.37 | 84.71–86.26 | 0.922 | 0.997 |

### 可以与不可以得出的结论

- **大幅组合改善有较强重复证据。** 配对耗时中位下降：Array 82.96%、Dictionary 90.33%、Array methods 80.08%、String 68.15%、Packed 29.41%。这些不是改算法/换批量API得出的成绩，但也不能将组合收益全部归功于最后一轮的直接Array/literal实现。
- **原始规则得到10项改善信号、0项支持回退、7项不确定。** 这只是本次六对样本的分类，并不证明没有回退；例如dispatch区间延伸至1.184、typed_map至1.263。typed_vec2配对比1.033，尽管独立中位数B<A，配对主指标没有显示改善，不能换指标宣布胜利。
- **Vector不能宣布AS专属收益。** 原始比0.913看似改善8.73%，但未修改的GD也为0.922，配对归一化0.997；这与共同环境/独立构建因素相容。Color、batch_sum归一化也接近1。不应凭这些数字启用实验POD或改动生命周期。
- **没有全面追平GDScript。** B数值上7/17项不慢于GD，仍有10项更慢：Array、Dictionary、Array methods、Packed、String、Vector、Color、typed_i64、batch_sum、dispatch。Vector/Color仍约5倍耗时；历史batch_sum接近持平的结果在本次没有复现。typed_vec2优势是既有批量路径，不是虚拟机逐元素公平对照。
- **当前决策：保留已通过回归的实现，不因本次噪声继续更改生产代码。** 若后续需要判断小于10%的增量或派发回退，应在用户认可的空闲窗口进行同样的交错复验，并先修正等待门/监测时间预算；本次不会停止其他程序或修改电源方案来制造条件。对几何与派发继续优化仍需独立ABI/生命周期证据，当前实验不足以替代它们。

## 证据与复现

[样本数据](<data/interleaved-ab-2026-10-13/samples.csv>)、[运行元数据](<data/interleaved-ab-2026-10-13/metadata.json>)、[构建身份与依赖清单](<data/interleaved-ab-2026-10-13/manifest.json>)、[逐秒监测](<data/interleaved-ab-2026-10-13/telemetry.jsonl>)以及[本次实际使用的 runner 快照](<data/interleaved-ab-2026-10-13/runner-measured.py>)保留在报告目录。运行快照用于审计；复现从测试目录调用[runner 源码](<../../modules/angelscript/tests/interleaved_ab_benchmark.py>)以正确导入共用 parser。

[完整结果CSV](<data/interleaved-ab-2026-10-13/summary.csv>)含17项结果、C#中位数、GD范围及全部配对诊断；[环境分析JSON](<data/interleaved-ab-2026-10-13/analysis.json>)与[进程记录](<data/interleaved-ab-2026-10-13/processes.json>)包含全部运行元信息。

原始日志按实际顺序：[1A](<data/interleaved-ab-2026-10-13/01-A.log>) → [1B](<data/interleaved-ab-2026-10-13/01-B.log>) → [2B](<data/interleaved-ab-2026-10-13/02-B.log>) → [2A](<data/interleaved-ab-2026-10-13/02-A.log>) → [3A](<data/interleaved-ab-2026-10-13/03-A.log>) → [3B](<data/interleaved-ab-2026-10-13/03-B.log>) → [4B](<data/interleaved-ab-2026-10-13/04-B.log>) → [4A](<data/interleaved-ab-2026-10-13/04-A.log>) → [5A](<data/interleaved-ab-2026-10-13/05-A.log>) → [5B](<data/interleaved-ab-2026-10-13/05-B.log>) → [6B](<data/interleaved-ab-2026-10-13/06-B.log>) → [6A](<data/interleaved-ab-2026-10-13/06-A.log>)。没有删点或补跑替换。

[A构建日志](<data/interleaved-ab-2026-10-13/baseline-build.log>)、[A模块回归](<data/interleaved-ab-2026-10-13/baseline-tests.log>)、[B构建日志](<data/interleaved-ab-2026-10-13/candidate-build.log>)、[B模块回归](<data/interleaved-ab-2026-10-13/candidate-tests.log>)保留完整记录。

从仓库根目录运行以下命令；复测应使用新的 `--output` 目录，runner 拒绝覆盖旧样本。保存的两个引擎运行目录不要在复测期间重建或更新。下面两条命令顺序执行，前一条成功后才运行分析：

```powershell
python modules/angelscript/tests/interleaved_ab_benchmark.py --a .superpowers/bench-ab-20261013/baseline/bin/godot.windows.editor.x86_64.mono.exe --b .superpowers/bench-ab-20261013/candidate/bin/godot.windows.editor.x86_64.mono.exe --a-revision 20530409fc50563e19954441aa1b81f47ffcee0e --b-revision 8c0143716da82b133937d29df42ab3a4c02608c7 --pairs 6 --output docs/language-benchmark/data/interleaved-ab-repeat
if ($LASTEXITCODE -eq 0) { python modules/angelscript/tests/interleaved_ab_analysis.py docs/language-benchmark/data/interleaved-ab-repeat }
```

[清单校验工具](<../../modules/angelscript/tests/interleaved_ab_manifest.py>)检查 flags、依赖与源码洁净度；[分析工具](<../../modules/angelscript/tests/interleaved_ab_analysis.py>)只汇总已有样本，不发起新测量。

## 保留的局限

原engine/hoist负载有Node泄漏，完整过程可报告每进程16个；C#实际加载Debug组件、部分用CLR容器；AS typed_vec2本来是批量路径；语言顺序固定。原驱动器每项重复7次，但仅输出最小/中位/最大和最后一次返回值，因此原始日志不是全部7次逐次时间序列；checksum一致也不是每次重复都独立断言，更不是完整语义等价证明。测试构建保留native thunk计数。所有这些沿用相同基准用于A/B，而不是忽略或改写。两轮历史结论分别见[phase12报告](binding-parity-phase12-2026-10-13.md)及[remaining-nine报告](remaining-nine-2026-10-13.md)。
