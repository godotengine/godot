# AngelScript 空闲门墙钟预算修复实施计划

**目标：** 修复交错 A/B 运行器的60秒墙钟预算与慢监控阻塞，完成确定性回归并进行一次失败即停止的空闲复测尝试。

**架构：** 使用单调时钟绝对截止时间代替60次采样循环；把全部 Windows/PDH 监控初始化及采样放进独立的自有子进程，由父进程通过有期限的 IPC 等待结果。默认空闲门失败即不启动下一基准进程，保存明确状态和全部已采证据。Windows 非实时调度可能导致父进程晚于截止时间获得执行机会，记录实际超出量，绝不把超期结果判为通过。

**技术栈：** Python 标准库（multiprocessing、ctypes、unittest）、Windows API、现有原17项 Godot mono 基准。

---

## 范围与预定规则

- 内联执行，不创建代理，不自动提交，不推送；不修改生产 C++、VM、原基准算法、语言顺序、迭代数和校验和。
- 保留旧报告、全部旧原始数据及[旧运行器快照](<../../docs/language-benchmark/data/interleaved-ab-2026-10-13/runner-measured.py>)。
- A=`20530409fc50563e19954441aa1b81f47ffcee0e`，B=`8c0143716da82b133937d29df42ab3a4c02608c7`。复用保留实际引擎；实验POD及强制generic保持关闭。
- 新输出目录 `docs/language-benchmark/data/interleaved-ab-idle-2026-10-14`，创建式写入，拒绝覆盖。比对旧身份清单中二进制、86文件依赖树、项目和实际Debug组件的哈希。
- 六对 AB/BA 固定顺序，每个进程前等待至多60秒预算；CPU连续3个有效新样本低于25%。单次采样父进程等待上限3秒；CPU平均窗口至多3秒，数据完成时CPU读数年龄至多2秒；缺失、非有限或过时读数不计为空闲。
- 第一个失败空闲门结束整次尝试；不重试挑选窗口，不结束用户进程，不改电源计划。只清理自己创建的监控或基准子进程。
- 如果完整矩阵无法获得，不生成完整配对统计，不对dispatch/vector/color作新性能结论。门通过也不等于运行期持续空闲，需继续检查遥测和GD对照。

### 任务1：添加失败测试

**修改：** [运行器测试](<../../modules/angelscript/tests/test_interleaved_ab_benchmark.py>)。

- [x] 用可注入 `clock`/`sleep` 和假采样器测试：繁忙窗口准确耗尽预算；连续3个低负载样本通过；迟到结果、过时/缺失/NaN读数拒绝；睡眠调度超时后不再请求采样；慢采样获剩余deadline而非重获60秒。
- [x] 使用真正自有监控子进程模拟初始化或采样卡住，确认IPC超时返回且清理仅终止该子进程。
- [x] 运行 `python -m unittest discover -s modules/angelscript/tests -p test_interleaved_ab_benchmark.py`，新增测试在实现前因MonitorTimeout尚未定义而导入失败；实现后回归通过。

关键测试形状（辅助假对象在测试文件定义）：

```python
clock = FakeClock()
monitor = FakeMonitor(clock, cpu=100, latency=1.5)
result = wait_for_idle(monitor, budget=5, clock=clock, sleep=clock.sleep)
self.assertFalse(result["passed"])
self.assertEqual(result["elapsed_s"], 5)
self.assertTrue(all(deadline <= 5 for deadline in monitor.deadlines))
```

### 任务2：实现有期限监控与失败关闭

**修改：** [运行器](<../../modules/angelscript/tests/interleaved_ab_benchmark.py>)、[遥测分析](<../../modules/angelscript/tests/interleaved_ab_analysis.py>)；按需新增分析测试。

- [x] `wait_for_idle` 使用 `deadline = monotonic() + budget`，所有睡眠和IPC等待都取剩余时间；检查采样完成时间并记录elapsed/overshoot/reason。
- [x] 监控worker独占PDH，父进程不调用可能阻塞的PDH初始化/采样；IPC异常/子进程失败不解释为0%CPU。超时后不再复用未完成请求。
- [x] 在每个门开始重置CPU差分基线，避免将上个基准进程的负载平均进下一次空闲判断；记录CPU窗口长度、读取及样本完成时间。
- [x] 基准运行期间也使用有期限监控等待，避免同类阻塞破坏子进程超时。保持亲和性0xF000及AboveNormal仅应用于基准子进程。
- [x] 保存门记录、明确结果状态、空进程列表/仅表头样本文件；门失败状态为 `idle_unavailable`，exit code为2，不输出PASS，不创建summary。失败/部分样本完整保留。
- [x] 分析器接受明确中止状态并汇总空闲遥测；完整成功运行仍需原17项完整矩阵。拒绝把残缺结果解释为完整性能比较。对旧analysis全部原有字段做只读复算比对，完全一致。
- [x] 新增可选旧identity清单校验，避免二进制相同但被测Debug程序集已变。前后复核引擎、依赖和项目。
- [x] 正式尝试前25项回归、Python语法验证和diff检查通过；tracked生产实现无diff。

### 任务3：有界实机复测尝试

**创建：** 新输出目录及冻结运行器快照、身份校验记录。

- [x] 测试结束后才开始正式尝试。使用上述旧A/B、原项目、六对顺序和新默认门规则。
- [x] 一次完整运行结束、exit0、12/12门通过、612行校验通过，无删除或替换。dispatch配对0.973、区间0.915–1.011；Vector/Color约0.949但GD改善更多，归一化1.070/1.029，不支持AS专属收益。
- [x] 开始旧身份清单核验和结束六类身份检查全部通过；门耗时3.032–50.203秒，中位8.593秒，超出量全0。worker正常退出清理0.0179秒、结束身份复核0.4402秒单独记录；实际机器此次未触发60秒耗尽分支。

### 任务4：审查和交付

**创建：** [中文报告](<../../docs/language-benchmark/idle-deadline-2026-10-14.md>)。

- [x] 报告修复机制、25项单元/真实worker测试、完整17项实机结果、残余OS调度及监控干扰边界。运行期CPU中位30.82%、队列中位0但最大162，不称完全无干扰。
- [x] 旧manifest与旧runner快照哈希未变，未写旧证据；tracked生产源码无diff；36处本地链接有效，6份Python脚本无行尾空白。612行与完整summary只读复算一致，12份raw均51条负载记录（另有1条engine元信息行）及既有16实例泄漏警告。后台任务pwsh-6已收集，monitor PID74928已退出。
- [x] 不自动提交；报告明确不启用POD、不改变上下文生命周期，不把GD共同变化认作AS专属收益。

## 自审

核心路径无TBD：截止时间、单次超时、样本有效性、失败关闭、证据保存及一次性尝试规则均预先定义。空闲条件不可人为保证，因此“未达到空闲、未运行基准”是合法且需诚实报告的完成分支。
