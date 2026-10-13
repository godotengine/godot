# AngelScript 剩余九项性能优化实施计划

**目标：** 不替换既有脚本算法或 API，在前两阶段成果上进一步降低 Array、Dictionary、String、Packed、typed_i64、dispatch、Vector/Color 的成本，并如实报告尚未追平项。

**架构：** 优先将 Array 白名单方法直接委托给 Godot Array，复用字面量/变量双形态解码构造 String/Variant。高风险上下文保留与几何 POD 语义使用隔离实验与显式门控，仅在生命周期、ABI、调试与性能证据充分时进入默认路径；不编辑 vendored VM/JIT。

**技术栈：** Godot C++、AngelScript 2.38、MSVC x64、doctest、Python、GDScript/C# 原有基准。

---

## 约束与验收

- 内联执行，不派子代理、不自动提交；保留工作区中上一轮未提交改动及其他文件。
- 原有 bench_project 不改算法、循环规模、校验和或语言实现。原负载以 3 个独立进程、每进程 7 次重复的中位数评价。
- 新增拆分微基准仅用于定位，不能替代原负载追平证据；计时区外准备容器与检查返回值。
- 测试覆盖原生与 angelscript_force_generic=yes；回退验证仅限现有 value ops，不声称 typed containers 已跨平台。
- dispatch 必须保留嵌套 PushState/PopState、异常清理、调试行为；不能引入延迟销毁可观察语义。
- 几何值保留已验证 MSVC 隐藏返回槽 ABI；VM POD 复制与 native 调用 ABI 分开验证。
- 所有未成功实验保留原因与数据，不宣称保证九项全部追平。

### 任务 1：拆分基准与本轮基线

**文件：** 创建 `modules/angelscript/tests/test_angelscript_boundary_probes.{h,cpp}`；沿用 `modules/angelscript/tests/binding_parity_benchmark.py`；输出 `docs/language-benchmark/data/binding-parity-nine-*.{csv,json,log}`。

- [ ] 保存本轮开始状态、实际运行引擎哈希；运行原基准 `--stage nine-before --runs 3`。
- [ ] 添加 opt-in 真 VM 微基准：empty native call、Array size/has/push/get、Variant literal/cached key、Dictionary cached key、String literal concat、Packed push/get、typed_i64 push/get、几何 add/assign、host dispatch。预构造数据不计入 get-only；所有返回值检验。
- [ ] 用 `TEST_CASE("[AngelScript] boundary probes")` 驱动，常规测试仅做正确性，环境变量允许长计时；多次取中位数，输出明确逻辑迭代单位。
- [ ] 基线与后续版本不同时构建/运行计时；计数版本与性能结论分开。

### 任务 2：Array 直接方法及安全字面量路径

**文件：** 修改 `modules/angelscript/binding/as_binding_native_value_ops.{h,cpp}`、`modules/angelscript/binding/as_binding_value_types.{h,cpp}`；测试 `modules/angelscript/tests/test_angelscript_native_value_ops.{h,cpp}`。

- [ ] 先写测试：`Variant("value")`、小写 string 变量/参数/返回值、空串/Unicode/转义、String 与小写 string 加法、复制后独立生命周期、typed/read_only Array push/append/clear/pop 的约束。
- [ ] 检查注册中存在 `Variant(const string &in)` 和 `String opAdd(const string &in) const`，记录红灯。
- [ ] 统一只读字面量引用解码：驻留指针直接使用，变量存储地址解一次；不更改调试器专用槽解码契约。
- [ ] native 与 generic 都添加相同签名；String 中间 Variant 不再用于直接字面量构造。重载决议覆盖 literal、String、string、Variant 及默认参数。
- [ ] 将 Array size/is_empty/hash/clear/push_back/push_front/append/has/count/front/back/pop 委托真实 Array 方法；不可通过裸 Vector 绕过只读/类型校验。
- [ ] 运行 native value ops 和全模块测试，单独测 Array 与 literal 阶段增益。

### 任务 3：dispatch 隔离原型

**文件：** `modules/angelscript/as_engine.{h,cpp}`、`modules/angelscript/as_script_instance.cpp`、`modules/angelscript/tests/test_angelscript_instance.{h,cpp}`、拆分基准。

- [ ] 修正计划缓存对 ref 参数资格判断，执行前复制 ASCallPlan 防止重入 HashMap 扩容悬空；缓存参数数目仅资格确认后使用。
- [ ] 先建立对照：每次 Prepare/Unprepare 与保留同函数已完成上下文；用全局纯标量函数限定原型，测试 GetState、返回值、异常后复用。
- [ ] 单独检验脚本对象引用/析构、函数模块销毁、重载、嵌套回调与异常；若不能维持原生命周期，默认路径不启用上下文保留。
- [ ] 运行 dispatch 多进程计时和调试 TCP 回归，保留有证据的安全部分。

### 任务 4：几何平凡生命周期原型与 Packed/typed 决策

**文件：** `modules/angelscript/binding/as_binding_native_storage.{h,cpp}`、`modules/angelscript/config.py`、`modules/angelscript/SCsub`、原生值测试及拆分基准。

- [ ] 仅为验证的平凡几何类型试验 POD 默认复制，保留正确 ABI 分类，禁用或省掉矛盾的注册行为。独立构建开关默认关闭直至验收。
- [ ] 覆盖 Vector2/Vector3/Color 的函数返回、参数、成员、数组元素、临时对象、链式赋值、跨宿主转换、错误/调试展开。
- [ ] ABI/语义不成立立即撤回默认启用，记录诊断；无须更改 VM 来强行跑通。
- [ ] 用 Packed/typed 拆分 push/get 与 empty-native 成本决定是否做私有缓冲原型；若存储不是主因，不盲目重写容器。

### 任务 5：完整验证与报告

**文件：** 创建 `docs/language-benchmark/remaining-nine-2026-10-13.md`，相关 CSV/JSON/log 和 debugger trace。

- [ ] MSVC native 全 AS suite；forced generic value-ops；恢复 native 重新验证；TCP debugger step/out/next/continue。
- [ ] 原负载 `--stage nine-final --runs 3`，校验每组及跨阶段迭代/校验和；对比当轮 GD 与前后 AS 中位数/范围。
- [ ] `git diff --check`、Python 编译和源码审查；无 VM/原基准算法改动。
- [ ] 报告明确启用/拒绝的实验、剩余差距、样本波动、C# Debug/CLR 差异、原有 Node 泄漏、dirty source 哈希范围及平台限制。

## 执行记录

用户已明确同意建议并沿用内联执行，无需再次询问执行模式。当前工作区保留 phase12 尚未提交的生产与测试变更；新一轮基线在这些成果之上。

### 阶段记录

- 任务 1 已完成：nine-before 原负载 153 行；before/direct/geometry 各 57 条独立拆分样本。各进程返回值验证通过。
- 任务 2 已完成：字面量注册红灯后实现 native/generic 同签名，Array 直接方法及只读/类型约束通过；direct 全模块 186 测试 / 3753 断言；nine-direct 153 行通过。
- 任务 3：ASCallPlan 使用调用前快照，ref/out 拒绝误编组，缓存计数只用于匹配函数。保留 context 的隔离计时已完成；外部释放对象后仍推迟析构，因此明确拒绝进入默认 dispatch。原 Unprepare / PushState / PopState 保留，不再继续投入不等价的模块/热重载保留策略。
- 任务 4 已完成实验评估：Vector2/Vector3/Color POD 保持 APP_CLASS ABI 与显式复制构造，省去 opAssign/析构。开启实验后完整 AS suite 190 测试 / 3797 断言、TCP 协议、原负载 153 行通过；仍不提升为默认：历史字节码/API 兼容未完整验证，而且没有稳定追平。Packed/typed 拆分未呈现明显独立存储成本，不重写缓冲区。
- 任务 5：forced-generic 的 value-ops 37 测试 / 83 断言通过；恢复默认 native、关闭几何实验后，全模块 190 测试 / 3797 断言及 TCP 通过。nine-final 153 行与四阶段总计 612 行校验一致，但性能波动很大；追加同二进制 nine-confirm 三进程，不删除第一组数据，合并六个进程中位数报告。
- 任务 5 已完成：同二进制 nine-confirm 153 行通过，合并最终两组共 306 行 / 六进程；8/17 数值达到 AS/GD≤1，其中目标九项只有 Array methods 为 0.98，且波动过大不能宣称稳定追平。四阶段比较器另验 612 行一致。最终报告已记录全部差距、异常样本、拒绝实验和实际引擎哈希。
- 最终产物：`docs/language-benchmark/remaining-nine-2026-10-13.md`、`docs/language-benchmark/data/remaining-nine-comparison.csv`、`docs/language-benchmark/data/remaining-nine-stability.csv`。未自动提交，未改 VM/原负载；默认 native、POD 实验关闭。
- 细粒度清单是原始计划，未勾选不等于执行失败：以上阶段记录为实际验收；不把已因析构语义拒绝的 context 保留扩展实验描述为完成。稳定追平目标仍未达成，须先获得稳定的交错 A/B 条件。
