# AngelScript 绑定性能优化（前两阶段）实施计划

**目标：** 不改变现有脚本表达式，消除绑定元数据锁及无调试行回调成本，补齐 Variant/容器快速路径，并以原负载多进程中位数检验能否追平 GDScript。

**架构：** 不可变 generic 绑定记录通过 AngelScript auxiliary 参数传递；异常回调始终保留，行回调按执行入口调试器状态同步。Variant 生命周期、标量转换和 Packed 标量数组操作在已验证原生 ABI 上直接包装，未覆盖平台沿用 generic；每一步独立构建和测量，不修改 VM/JIT。

**技术栈：** Godot C++17、AngelScript 2.38、MSVC x64、SCons、doctest、Python、现有三语言基准工程。

---

## 范围与约束

- 用户已经确认内联执行，不再询问执行方式，不创建子代理、不自动提交。
- 同一源码基线 20530409fc；保留既有 17 项负载、迭代次数、7 次进程内重复和校验值，不以新 API/批处理替代原负载。
- C# 实际为 Debug 程序集，仅作为旁证；CLR 与 Godot 容器差异延续此前披露。
- 原生优化遵守 AS_NATIVE_VALUE_OPS_ENABLED 与 AS_NATIVE_VALUE_OPS_FORCE_GENERIC；typed containers 既有跨平台缺口不在此次顺手重构。
- 不追求“保证所有项快于 GD”；结果未达标如实报告。单项 AS/GD<=1.00 才算追平，不拿最佳值冒充中位数。

### 任务 1：固化基准

**文件：** 创建 `modules/angelscript/tests/binding_parity_benchmark.py`；读取 `modules/angelscript/tests/bench_project/bench_driver.gd`；输出 `docs/language-benchmark/data/binding-parity-*.csv` 及 raw logs。

- [x] 创建执行器，顺序运行 3 个独立进程，每进程原有 7 repeats；解析 BENCH 行，验证 51 行、语言间校验一致和非零迭代数。
- [x] 对当前未修改引擎跑 before；记录 executable SHA256、命令、引擎版本和每轮全部输出；不并行跑构建和计时。
- [x] 汇总每项 3 个进程中位数的中位数、最小/最大，避免只报 best。

### 任务 2：移除 generic 元数据锁

**修改：** `modules/angelscript/binding/as_binding_value_types.cpp:92-127,742-745`、`modules/angelscript/binding/as_binding_object.cpp:100-117,226-379,541,593`、`modules/angelscript/binding/as_binding_registry.cpp:68-69,136-144`。
**测试：** `modules/angelscript/tests/test_angelscript_binding_value_types.{h,cpp}`。

- [x] 测试遍历引擎值类型 methods/behaviours 和全局 functions；带旧 UserData 的 generic 记录必须 `CHECK(func->GetAuxiliary() == func->GetUserData())`，并且至少覆盖方法、行为、全局三种注册方式（运行前应失败）。
- [x] 注册传入记录指针；调用只用 `p_gen->GetAuxiliary()`；保留已有 UserData 兼容外部检查且不改变既有生命周期。不得简单禁用 AS 线程锁。
- [x] 运行 binding tests 和全 AS suite；独立构建并测 auxiliary 阶段。

### 任务 3：按调试状态装配行回调

**修改：** `modules/angelscript/as_debugger.{h,cpp}`、`modules/angelscript/as_engine.cpp`、`modules/angelscript/as_script_instance.cpp`。
**测试：** `modules/angelscript/tests/test_angelscript_debugger.{h,cpp}`、`modules/angelscript/tests/debugger_protocol_test.py`。

- [x] 增加可测试的 line callback enabled 状态；测试同一 context 非调试 attach=false，模拟 debugger 后同步=true，移除 debugger 后同步=false，异常装配始终为 true。
- [x] ExecutionScope 接收具体 context，在每次 Execute 前同步；内部 RequestContext 同步。只有状态改变才 Set/ClearLineCallback，避免每次 dispatch 重复注册。
- [x] 保持递归/nested context step 深度平衡；异常/手动暂停/step/next/out 原回归不变。
- [x] 跑完整 AS suite + TCP 协议，独立测 callbacks 阶段。远程初次连接一般发生于启动；不支持无回调长循环中凭空发现新 debugger，文档明确边界。

### 任务 4：Variant 与 Packed 完整路径

**修改：** `modules/angelscript/binding/as_binding_value_types.cpp`、`modules/angelscript/binding/as_binding_native_value_ops.{h,cpp}`。
**测试：** `modules/angelscript/tests/test_angelscript_native_value_ops.{h,cpp}`。

- [x] 测试 Variant bool/int64/double 与 String 负载构造、copy/self assignment、转换及销毁；包括 `int64(Variant(-7))`、`double(Variant(2.5))`、引用计数载荷。
- [x] 已验证 ABI 上注册直接 placement-new Variant 默认/copy/标量构造、析构与 opAssign、标量 opConv。字面量仍使用现有 string_slot 规则，不盲目把工厂指针当槽地址；可单独 generic 专用包装消除大 switch。
- [x] PackedByte/Int32/Int64/Float32/Float64 经 VariantInternal 取真实容器；直接 get/set/size/push_back，不复制 Vector 触发额外 COW；保持原边界返回值与既有错误行为。
- [x] Array/Dictionary 原生路径审查：尽可能去重复临时 Variant，但保留只读/typed 容器验证，不绕过核心约束。
- [x] 扩展负索引/越界、空数组、COW、窄化、float 精度、Dictionary missing key 和引用载荷测试。
- [x] 原生及强制 generic 构建跑相同测试；不可声明未实跑的平台受验证。

### 任务 5：最终验收和报告

**创建：** `docs/language-benchmark/binding-parity-phase12-2026-10-13.md`。

- [x] 完整 AS suite；TCP 调试协议；git diff --check；核对未修改 thirdparty VM/基准算法。
- [x] 最终同配置测 3 进程×7 repeats，与 before/auxiliary/callbacks 各阶段逐项比较；列每项 AS/GD、变化百分比、离散性。
- [x] 强制 generic 测试结束恢复原生构建；最终 binary 对应最终源码。
- [x] 记录局限：基准工程已知 Node 泄漏、C# Debug、测试插桩、未测试 GUI/其他平台、profiler 未另测。若前两阶段仍有剩余差距，说明后续需第三阶段而非编造完成。

## 构建/验证命令

```powershell
cmd /c ".superpowers\sdd\2026-09-30-angelscript-hot-update-phase1\build-tests.cmd module_mono_enabled=yes"
bin\godot.windows.editor.x86_64.mono.console.exe --headless --test --test-case='*[AngelScript]*'
python modules/angelscript/tests/debugger_protocol_test.py
python modules/angelscript/tests/binding_parity_benchmark.py --stage before --runs 3
```

## 检查点

先保证 baseline 数据可复现，再改生产路径；每阶段先测试再测量。性能与构建不同时跑。发生崩溃、ABI 返回槽错误或调试回归时先修复，不能用快但错误的结果验收。

## 执行记录（2026-10-13）

本计划的实现范围已完成；勾选表示执行过对应工作，不代表所有性能项目达标。正式 before/auxiliary/callbacks/final 各 153 样本，四阶段跨语言/阶段 checksum 一致。最终 AS/GD 数值门槛命中 8/17，batch_sum 只是近似持平，余下 9 项未追平，不能宣称完整 parity。

原生完整模块通过 184 tests / 3711 assertions；原生 value-ops 34/102、强制 generic 34/70；TCP 协议通过。generic 验证后恢复原生构建并测量 final，最终实际引擎 SHA256 已记录。没有改 VM/JIT 或原基准工程算法，没有自动提交。

偏差与限制：前三阶段记录的 SHA256 实际是 Windows console launcher，并非 runtime；最终执行器已修正同时记录 runtime 和源码差分指纹，旧阶段无法追溯补造。generic 仅验证 value-ops 门控，不代表此前 typed-containers 已具备通用跨平台回退。局限与全部数值详见[验收报告](<../../docs/language-benchmark/binding-parity-phase12-2026-10-13.md>)。
