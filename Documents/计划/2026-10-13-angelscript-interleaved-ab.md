# AngelScript 交错 A/B 复测实施计划

**目标：** 先提交现有优化，然后以同机交错计时判断组合优化的收益和回退，不把噪声解释成因果。

**架构：** A 为优化前提交 20530409fc50563e19954441aa1b81f47ffcee0e，B 为已提交优化 8c0143716da82b133937d29df42ab3a4c02608c7。A 在独立 detached worktree 编译，B 使用当前主工作树编译；同样 MSVC、mono、tests、editor 配置。分别保存实际引擎及依赖哈希，固定原17项基准和 C# Debug 组件。采用预先确定的 AB/BA 交错配对顺序及进程级亲和性，记录 CPU 使用/频率代理指标，所有样本保留。

**技术栈：** Windows、PowerShell、Python 标准库、SCons、Godot mono、原始 GDScript/AngelScript/C# 基准。

---

## 约束与解释边界

- 先完成本地提交；本阶段脚本/结果不自动再提交、不推送、不停止用户进程、不改电源方案。
- A/B 是前两轮优化的组合比较；由于中间阶段未单独提交，本实验不能隔离“remaining-nine”增量。
- 两个原始 engine/hoist 探针的 Node 泄漏、固定语言顺序、C# Debug/CLR 容器差异原样保留并披露。
- 不改变原负载脚本、迭代数、预热和校验和；不修改 VM。POD 实验关闭。
- 不将系统平均频率/CIM CurrentClockSpeed 声称为逐核实测频率，不声称获得完整调度轨迹。采集受支持的性能计数器；缺失项记为 null。
- 统计以进程为单位，7次进程内重复不当作7个独立样本。预定6对，共12个进程，AB/BA交替；非测量的启动/校验可以另列。
- 所有异常样本保留，不按速度删点或择优；校验失败则整次实验无效并报告。若环境依然不稳，结果就是不确定，不强行继续优化或改用户环境。

### 任务 1：提交与构建身份

**文件：** 已提交实现与报告；生成 `.superpowers/bench-ab-20261013/manifest.json`。

- [x] 检查并只提交前两轮相关文件，提交为 8c0143716d；不含无关 untracked。
- [x] 建立 baseline detached worktree `.superpowers/bench-ab-20261013/baseline`，不切换主分支。
- [x] 用一致命令编译 A/B：`scons platform=windows target=editor tests=yes accesskit=no d3d12=no module_mono_enabled=yes -j8`；B 显式 `angelscript_force_generic=no angelscript_pod_geometry_probe=no`。
- [x] B 保存到独立运行目录；A 运行自己的 bin。记录 source revision、runtime exe SHA256、GodotSharp 文件树哈希及 benchmark 文件树哈希。
- [x] 每份跑完整 AngelScript 模块测试：本次 A 179/3678、B 190/3797 全部通过。

### 任务 2：可测试的交错 runner

**创建：** `modules/angelscript/tests/interleaved_ab_benchmark.py`、`modules/angelscript/tests/test_interleaved_ab_benchmark.py`。

- [x] 单元测试预定 AB/BA 顺序、原51行解析、跨版本 checksum/迭代校验、缺行拒绝、配对比与区间计算。runner 与 manifest 共11项测试通过。
- [x] 实现先创建挂起进程设置相同 affinity/AboveNormal 优先级再继续（仅 benchmark 子进程，不能 realtime），不让 launcher 掩盖实际进程；使用实际 `.exe` 并重定向输出验证。烟测51行、exit0、实际mask0xF000、priority32768。
- [x] 已采样总 CPU 使用、可用内存、队列/频率代理，记录225个运行期监测点及实际间隔；监测开销未独立量化，不声称达到零干扰。
- [x] 持久化全部12个进程 raw log、612条统计记录、wall/CPU时间/affinity/身份，输出不覆盖；全部退出0。本次未触发超时，清理超时分支没有实测覆盖。
- [ ] 原定采样前最多等待60秒墙钟未达成：当前实现限制60次采样，首末样本实际跨度90.88–99.66秒。全部12次空闲门失败，已标记并保留全部样本；构建全部结束后才采样。此项作为明确偏差保留，不伪标完成。

### 任务 3：交错执行

**生成：** `docs/language-benchmark/data/interleaved-ab-2026-10-13/` 下元数据、raw、samples、telemetry。

- [x] 小运行验证两份 exe 使用同一个项目和相同 C# 组件，方法和51输出完整。两臂均exit0、亲和性和优先级生效；低负载门均未通过。
- 已识别等待门偏差：实现是60次至少1秒采样，不是严格60秒墙钟。正式运行保持已开始的同一版本，记录实际间隔；不把这一点隐藏为符合原计划。
- [x] 完成固定6对 AB/BA，运行前后实际引擎、依赖、项目及runner快照哈希一致；记录全部环境数据，306行/版本、总612行，无删点补跑。
- [x] 每项验证 A/B/AS/GD/C# 校验和和迭代次数完全一致。原驱动器仅输出重复内min/median/max和最终返回值，不能声称每次重复独立校验。

### 任务 4：分析与交付

**创建：** `docs/language-benchmark/interleaved-ab-2026-10-13.md`。

- [x] 报告17项A/B中位数、配对比、B AS/GD、范围、GD对照及bootstrap区间。原始分类10项改善信号、7项不确定，不能宣称无回退；Vector归一化比0.997，不支持独立AS优化收益。
- [x] 记录空闲门0/12、总CPU中位100%、队列中位131及计数器限制；B仍有10/17项慢于GD。大幅Array/Dictionary/String组合收益再次出现，不把小差异归因于代码。
- [x] runner/manifest最终11项单测、py_compile、git diff --check通过；报告/计划32处本地链接核验通过。主代码生产实现保持已提交版本，本次脚本/数据/报告不自动追加提交。

## 执行结论

本轮复验已结束；等待门墙钟预算偏差已显式保留，不为达标重新挑选样本。详细数据与复现命令见[交错A/B报告](<../../docs/language-benchmark/interleaved-ab-2026-10-13.md>)。如果需要判断派发或几何的小幅增量，应另选获准空闲窗口并先修正等待/监测时间预算，本阶段不擅自停止用户任务。
