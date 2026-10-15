# AngelScript 成本归因与编译回退超时修复

## 已完成提交

本地提交 `2bfae432e8763a396a5999053403fcf04d5e6738`：`test(angelscript): preserve paired benchmarks and enforce idle deadlines`。60个相关文件、13377行新增，含29份原始日志。两证据目录的50个文件与Git index逐字节一致；增加局部换行属性保护历史哈希。未推送、未提交无关文件。

## 下一步已实现的诊断

[实施计划](<../../Documents/计划/2026-10-14-angelscript-cost-attribution.md>)、[C++探针](<../../modules/angelscript/tests/test_angelscript_cost_attribution.cpp>)、[测试注册](<../../modules/angelscript/tests/test_angelscript_cost_attribution.h>)、[成本运行器](<../../modules/angelscript/tests/cost_attribution_benchmark.py>)和[解析测试](<../../modules/angelscript/tests/test_cost_attribution_benchmark.py>)目前未追加提交。

- 9条派发路径：raw0/scoped0、生产bridge有/无plan、4参raw/有plan/无plan、ScriptInstance、Object::callp。
- 3个批量阶段：256个context批量Prepare、Execute+读取、Unprepare；明确不能当作单context逐次路径的可加总成本。
- 16条几何对照：Vector2/Color各自标量、原运算符、诊断原地native/generic、读取、copy、assign、default。
- opt-in计时每项预热1轮、记录7轮原始值；字节码CALLSYS/生命周期站点及独立n=32 native调用计数输出。
- 运行器复用60秒空闲门、亲和性/子进程清理；计划6进程、正反顺序交替，失败即停止，不替换样本。

诊断本身不改生产派发、几何ABI、POD默认值或上下文销毁时机。后来按用户明确选择优先修复验证超时，增加一处vendored AS注册编译器查找优化，详见下文；运行字节码/ABI/生命周期不变。原地诊断仅在测试中添加唯一命名的方法；原有运算符不变。测试构建包含诊断辅助代码，旧A/B归档二进制未覆盖。

## 验证结果

### 已通过

- 增量MSVC mono/tests编译成功；首次构建30.67秒，后续诊断修复也构建成功。
- cost独立短测：1项、160条断言通过。
- Python：新解析4项＋既有A/B与空闲门25项通过，语法编译及diff检查通过。
- 191项AngelScript逐项隔离尝试中174项直接通过。另12项因命令行测试名匹配问题最初未选中（11个中文名、1个包含逗号的名称），不能当作通过；改用source-file/唯一英文前缀后补测通过：debugger文件13项213断言（其中2项与174有重叠），highlighter1项32断言。因此累计186个不同用例具有通过记录。
- 全部派发、几何、native value ops、typed containers、上下文生命周期等相关隔离用例通过。

### 修复前阻塞记录（保留，不冒称当时通过）

完整单进程AngelScript回归180秒超时；verbose诊断显示未执行到cost测试，日志最后停在既有string copy/assignment测试附近。该string测试独立执行通过。**未改动的保留candidate二进制也在全套180秒超时**，因此没有证据把该现象归因于新探针，但也不能声称不存在回归。

隔离执行还有5项每项20秒超时：

1. binding visibility static method as global function
2. binding visibility marshals Variant parameters and returns
3. compile_module reports diagnostics on syntax error
4. lazy compile_module recovers missing type
5. ASScript rejects syntax errors and stays invalid

其中第3项用未改动candidate再作30秒对照，仍超时。其余4项尚未做逐项旧二进制对照，不扩大推断。此分布提示应优先检查编译失败后的lazy绑定全量回退/诊断路径；**这只是定位假设，尚未证明根因**。

探针最初使用全局 `Vector2 &inout` 被默认安全引用规则拒绝，产生Invalid configuration；已改为receiver方法，不开启unsafe references。增加无fallback编译预检以免探针错误进入昂贵回退。初始多次超时和后续成功记录均保留。所有自有Godot进程已退出，未停止用户程序。

## 原始检查证据

- [首次构建](<../../.superpowers/cost-attribution-build-20261014.log>)、[修复后构建](<../../.superpowers/cost-attribution-build-methods-20261014.log>)、[cost短测成功](<../../.superpowers/cost-methods-check-20261014.log>)。
- [预检失败诊断](<../../.superpowers/cost-preflight-20261014.log>)、[全套verbose定位日志](<../../.superpowers/cost-full-as-diagnostic-20261014.log>)。
- [逐项隔离运行脚本](<../../.superpowers/run-cost-isolated-tests-20261014.py>)、[191项隔离结果](<../../.superpowers/cost-isolated-tests-20261014/results.json>)。
- [补充对照状态](<../../.superpowers/cost-extra-controls-20261014.json>)、[debugger补测](<../../.superpowers/cost-control-debugger-20261014.log>)、[highlighter补测](<../../.superpowers/cost-control-highlighter-20261014.log>)。

超时程序使用文件重定向时部分日志为零字节；不是干净通过的证据，超时事实见运行器工具记录与结果JSON。单进程完整回归的超时不能被186项隔离通过替代。

## 用户选择后：根因与修复

用户选择“先定位并修复回归超时”。[修复计划](<../../Documents/计划/2026-10-14-angelscript-compile-timeout.md>)及[本地补丁说明](<../../modules/angelscript/thirdparty/angelscript/LOCAL_PATCHES.md>)记录了授权后的范围变化。

**不是固定类死循环，而是二次增长的全引擎扫描。** 阶段日志显示注册持续推进，不同超时时刻停在不同类。在800个类时，累计method阶段0.433秒、property阶段24.265秒；全局函数名冲突扫描并非该阶段主要成本。根因位于[ValidateVirtualProperty](<../../modules/angelscript/thirdparty/angelscript/source/as_builder.cpp#L1949>)：每注册一个getter/setter，原实现遍历全部engine->scriptFunctions找配对访问器。Godot完整ClassDB接口逐类复制继承成员，函数集合增大后扫描成本急剧累积。语法错误触发lazy全量绑定使其暴露为超时。

**最小修复：** 对 `asFUNC_SYSTEM` 且所属类型非脚本对象的原生属性，仅遍历其所属类型已注册的method id列表；仍保留名称、IsProperty、module、namespace和objectType筛选、原类型兼容校验及返回码。脚本对象和全局属性仍使用原全引擎扫描，因为脚本声明未必已进入类型的方法表。不取消必要回退、不吞错误、不打开unsafe references、不改变执行字节码/ABI/生命周期。

[新属性注册回归](<../../modules/angelscript/tests/test_angelscript_property_registration.cpp>)验证两种注册顺序、不匹配拒绝、不同类型同名隔离、索引属性、全局/脚本属性与256类型规模化注册；[语法错误回归](<../../modules/angelscript/tests/test_angelscript_engine.h#L61>)补充错误后继续编译有效脚本。

| 验证 | 修复后 |
|---|---|
| 新属性注册回归 | exit0，0.307秒 |
| 语法错误＋随后有效编译 | exit0，3.700秒 |
| 单进程完整AngelScript | **192/192、4761断言通过，11.666秒** |

修复前语法错误至少180秒仍未完成，不能给出精确倍率；上述修复后墙钟含启动，不是函数独立计时。全套仅保留原host-bridge error3/缺默认factory负例日志。所有临时生产插桩已移除；旧A/B二进制和数据不变。成本计时使用修复后的新测试二进制，身份单独记录，不与旧成绩混合。

[注册阶段定位证据](<../../.superpowers/compile-timeout-stage-profile-20261014.log>)、[180秒推进日志](<../../.superpowers/compile-timeout-long-trace-20261014.log>)、[修复构建](<../../.superpowers/compile-timeout-fixed-build-20261014.log>)、[修复后三项结果](<../../.superpowers/compile-fixed-results-20261014.json>)、[完整回归日志](<../../.superpowers/compile-fixed-full-20261014.log>)。

## 成本测量

恢复验证后进行一次预定六进程有界尝试，**在第一个进程前按规则停止**，未补跑或选择更快窗口。

- 第一个空闲门预算60秒，实际60.001003秒，超出1.003毫秒（OS调度余量，如实记录）。
- 58个遥测样本均有效，CPU中位33.99%，范围24.23%–54.33%；最后连续低负载计数只有1，未达到3个CPU<25%的条件。
- 队列中位0、最大45；CPU差分窗口约1.001–1.028秒。
- 状态 `idle_unavailable`，启动性能进程数0、性能记录0；没有生成完整summary。工具包装进程显示非零退出，持久化结果明确是空闲门失败，不是引擎运行错误。
- worker正常退出，清理0.0203秒；实际引擎、依赖及探针/运行器源码结束哈希复核均通过。

[运行元数据](<data/cost-attribution-2026-10-14/metadata.json>)、[门记录](<data/cost-attribution-2026-10-14/gates.json>)、[明确停止结果](<data/cost-attribution-2026-10-14/result.json>)、[环境汇总](<data/cost-attribution-2026-10-14/environment.json>)、[原始遥测](<data/cost-attribution-2026-10-14/telemetry.jsonl>)、[修复构建日志](<data/cost-attribution-2026-10-14/build-fixed.log>)、[完整回归日志](<data/cost-attribution-2026-10-14/full-tests-fixed.log>)及[修复前注册分项日志](<data/cost-attribution-2026-10-14/registration-before.log>)保存在新证据目录，旧证据未改。

**本轮没有新的dispatch/Vector/Color成本数字。** 28条探针的默认短正确性路径已通过，Python完整矩阵解析有夹具测试；但重计时及字节码输出的实机路径尚未经过正式执行，不能声称已完成六进程成本归因。也不能把注册期3.7秒修复效果当作运行时几何或派发提速。

## 交付状态与下一步

前轮工具/证据提交已经完成且未推送；本轮新增的注册期查找修复、回归、探针、计划与报告尚未追加提交。生产运行时派发、对象生命周期、ABI、POD默认值未改；第三方AS仅注册编译器的一处局部查找补丁，维护说明明确记录。

下一步应在空闲窗口运行同一批诊断探针，先验证opt-in完整输出，再分析Prepare/Execute/Unprepare及几何临时值边界。不要因本次门失败降低阈值或替换样本，也不应在没有新证据前继续修改派发/几何生产路径。
