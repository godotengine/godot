# AngelScript 派发与几何分项成本归因实施计划

**目标：** 在不改变生产语义的前提下，区分宿主派发包装、上下文阶段及几何临时值/赋值/系统调用边界的成本，形成可复核诊断结论。

**架构：** 独立tests-only C++测试编译小型AS诊断模块，直接调用既有生产桥接与值操作；另用批量预建context测Prepare/Execute/Unprepare阶段，避免每调用计时器淹没几十纳秒信号。几何采用语义等价的标量、原运算符、诊断原地函数对照，并输出静态字节码调用站点和非计时动态native计数；所有差值只作诊断、不假称可加总的精确饼图。

**技术栈：** Godot tests/doctest、AngelScript 2.38公开API、MSVC x64、Python标准库、已验证的有界空闲监控。

---

## 先行提交（已完成）

- 本地提交 `2bfae432e8763a396a5999053403fcf04d5e6738`：`test(angelscript): preserve paired benchmarks and enforce idle deadlines`。
- 60文件，含29份日志；两证据目录50文件全数与Git index逐字节一致。局部.gitattributes保留原始换行与raw日志，不改写历史哈希。
- 25项工具单测通过；没有推送，没有加入无关文件。

## 约束与测量声明

- 不修改VM、生产派发实现、几何注册、POD默认值、旧17项基准及旧证据；新诊断不是新的语言公平对照分数。
- 不保留完成context来修改对象析构时机。每个生产调用仍Unprepare，诊断分阶段的context也及时清理。
- 单次正常回归用小N；显式 `AS_COST_TIMING=1` 时N=200000、预热1轮、原始7轮逐次输出。不将7轮当成7个独立进程。
- 6个独立进程，探针顺序正反交替，以同一可执行文件计时。前门60秒/单次监控3秒/3个有效CPU<25%样本，失败即整次停止，不补跑替换。只限制自有基准子进程mask0xF000/AboveNormal。
- 阶段计时使用256个预建context组成一批，分别计Prepare、Execute+返回读取、Unprepare；记录同规模空计时器控制和完整串行路径，说明批处理缓存/工作集不同，因此不把阶段和强行当作真实逐次调用成本。
- 本轮只提交此前工作；新诊断、构建、结果不自动再提交。

### 任务1：诊断回归与探针

**创建：** `modules/angelscript/tests/test_angelscript_cost_attribution.h`、`modules/angelscript/tests/test_angelscript_cost_attribution.cpp`。

- [ ] 注册独立 `[AngelScript] cost attribution` 测试，默认只运行短正确性检查。
- [ ] 派发：原始Prepare/Execute/读取/Unprepare；额外ExecutionScope；生产call_function有/无call plan；0参和4标量参数；ScriptInstance与Object::callp热缓存。检查每轮checksum、错误码和调用后状态。
- [ ] 阶段计时：256 context批，Prepare/Execute+读取/Unprepare分别聚合，输出总调用数；每阶段返回码与checksum检查在计时段外汇总验证。
- [ ] 几何：Vector2和Color分别比较标量循环、`acc=acc+v`、诊断inout原地加、copy构造+读取、assign+读取、仅读取、默认构造作用域。对比组每轮checksum一致。
- [ ] 原地函数只在测试注册为唯一命名的诊断方法，不改变已注册Godot类型运算符。最初全局 `Vector2 &inout` 方案被AS默认安全引用规则拒绝并导致Invalid configuration；改用receiver方法，避免打开unsafe references。复用生产native thunk计数，另次n=32执行采集，注明其不覆盖原生构造/析构。
- [ ] 通过GetByteCode及asBCInfo公开元信息输出每条指令opcode、CALLSYS目标声明，以及FREE/ALLOC，避免猜测临时值生命周期；静态站点不当作动态计数。
- [ ] 所有探针验证默认Color alpha、拷贝后值独立、正反顺序结果一致；不在循环内调用doctest宏。

测试核心形状：

```cpp
int rc = ctx->Prepare(func);
rc |= ctx->SetArgDWord(0, n);
int executed = ctx->Execute();
int64_t checksum = ctx->GetReturnQWord();
ctx->Unprepare();
CHECK_EQ(rc, asSUCCESS);
CHECK_EQ(executed, asEXECUTION_FINISHED);
CHECK_EQ(checksum, int64_t(n) * 3); // Vector2(1,2) 累加或等价标量
```

### 任务2：运行器与解析测试

**创建：** `modules/angelscript/tests/cost_attribution_benchmark.py`、`modules/angelscript/tests/test_cost_attribution_benchmark.py`。
**修改：** [已有子进程启动器](<../../modules/angelscript/tests/interleaved_ab_benchmark.py>)：仅新增可选命令/环境参数供复用，不改变旧默认命令或门策略。

- [ ] 先写解析测试：7次原始重复完整、checksum稳定、缺失/重复/非正时间拒绝、类型/名称集合一致；实现后运行通过。
- [ ] 新运行器保存runner/probe源码、实际exe及依赖哈希、tracked源码diff、构建信息；拒绝旧二进制缺失测试，要求日志包含测试通过及完整探针矩阵。
- [ ] 输出新创建目录中的逐轮CSV、每进程中位数和六进程范围、字节码/计数原始日志、门与运行期遥测、明确中止/成功状态；前后复核输入身份。

### 任务3：编译、回归、有界执行

- [ ] 原MSVC mono/tests配置增量构建，保持 `angelscript_force_generic=no angelscript_pod_geometry_probe=no`，不覆盖旧A/B归档。
- [ ] 独立cost测试及完整AngelScript模块回归；全Python相关测试及diff检查。编译或正确性失败先修复，不用失败二进制计时。
  - 执行偏差：cost短测160断言通过；单进程全套180秒超时、未到cost测试，verbose停在既有string copy/assignment附近。保留candidate二进制同样180秒超时，既有string测试独立运行通过；因此增加逐用例隔离进程回归（每用例20秒、总600秒），不把隔离通过冒称单进程全套通过。
- [ ] 构建和回归结束后才尝试6进程；新输出 `docs/language-benchmark/data/cost-attribution-2026-10-14`。任何门失败不补跑，明确缺少完整测量。

### 任务4：分析交付

**创建：** `docs/language-benchmark/cost-attribution-2026-10-14.md`。

- [ ] 区分直接观测、跨路径差值及假设；输出各派发路径/阶段、几何对照、字节码证据和动态计数。
- [ ] 明确没有GDScript宿主循环控制时不能直接把C++派发总时间等同于旧GD驱动dispatch；诊断原地函数不是生产优化收益。
- [ ] 报告测试/构建/环境、语义边界、下一优先级，检查旧证据无改动和任务清理；不额外提交。

## 当前验证检查点

- cost短测1项160断言通过；4+25项Python通过；生产C++/VM无diff。
- 单进程全回归180秒超时，旧candidate对照同样超时，不能声称全套通过。
- 191项隔离尝试174直接通过；12个测试名选择问题经source-file/唯一前缀补测后通过，累计186不同用例通过；5个编译/lazy/错误路径仍超时，旧candidate语法错误用例也30秒超时。
- 用户已选先修复。定位原生虚拟属性注册的全引擎扫描二次增长，局部查找补丁后完整192项/4761断言11.666秒通过；不再是未解决验证阻塞。
- 修复后一次正式计时尝试首门60.001秒失败（58个有效样本CPU中位33.99%、最后连续低负载仅1），按规则未启动性能进程，结果idle_unavailable，无补跑。所有后台任务已收集，没有残留Godot进程。28条探针短正确性已验证，opt-in计时/字节码实机完整路径仍待空闲窗口，不冒称成本归因已测完。
- 详见[检查点报告](<../../docs/language-benchmark/cost-attribution-2026-10-14.md>)。

## 自审

可行性：现有SCsub会编入tests/*.cpp并自动收集tests/*.h；公开API足以实现，不需要生产插桩。数值以小整数float累计，N=200000不会触发浮点精度饱和。所有测量保持有限总进程数及超时；没有空闲窗口也应保留明确停止证据。
