# AngelScript 断点与单步调试器实施计划

**目标：** 在现有异常诊断基础上接入 Godot 原生用户行断点、继续、单步进入/跳过/跳出与运行中暂停，复用调用栈和变量面板。

**架构：** AngelScript 行回调同步 ScriptDebugger 的 lines_left/depth 状态，与 GDScript 使用相同远程命令。Execute 前后建立线程局部执行作用域，以 AS 栈深变化模拟函数进入/退出；断下与 line_poll 期间提供仍然活着的 AS 上下文和语言，保证暂停时可查询。模块全局变量复用现有安全变量解码。

**技术栈：** Godot C++、AngelScript 2.38、ScriptDebugger/EngineDebugger、doctest、Windows MSVC/SCons。

---

## 范围与约束

- 实现 Godot 现有 step/next/out/continue 协议，不引入第二套 UI。
- 用户断点与异常断点区分，尊重 skip_breakpoints/ignore_error_breaks。
- 无调试连接时行回调快速返回；不剥除行信息，非可执行行不自动迁移断点。
- 继续不重新 Execute/Prepare；回调内阻塞后原地恢复。
- AS 未捕获异常仍不可继续；任意表达式求值、条件断点、变量修改、内部精确 profiler 不属于断点/单步执行控制，不虚称已经实现。
- 沿用内联执行，不创建子代理，不提交无关文件。

### 任务 1：行回调与单步状态

**文件：** 修改 modules/angelscript/as_debugger.h、modules/angelscript/as_debugger.cpp、modules/angelscript/as_script_language.h、modules/angelscript/as_script_language.cpp。

- [ ] 编写测试：真实 AS 执行，伪 EngineDebugger::debug 接收暂停，记录来源、行号、locals；设置 step 的 depth=-1/lines_left=1，验证下一次行回调暂停。
- [ ] 编写测试：next depth=0 跳过深层函数，out depth=1 停在调用者，递归同一函数仍按栈深判断。
- [ ] 实现 SetLineCallback；每条回调用 GetCallstackSize 同步深度差，仅 stepping 激活时变更 ScriptDebugger 深度。
- [ ] break_context 与 error 在断点和轮询期间保存/恢复；ScriptDebugger 的 break_language 同样保存/恢复，供 core:break 读取现场。
- [ ] 用户断点调用 debug(...,true,false)，异常保持 debug(...,false,true)；保护调试查询重入。

### 任务 2：执行边界

**文件：** 修改 modules/angelscript/as_engine.cpp、modules/angelscript/as_script_instance.cpp。

- [ ] 建立 ASDebugger::ExecutionScope（构造保存外层栈计数，析构扣除当前栈并恢复外层），包住宿主 call_function 和 factory Execute。
- [ ] AS RequestContext/ReturnContext 成对进入/退出内部执行范围，保证全局初始化、构造与析构产生的上下文同样参与步进。
- [ ] 验证顶层返回、上下文复用、嵌套宿主回调；执行结束后 thread_local 不残留失效指针。

### 任务 3：模块全局变量

**文件：** 修改 modules/angelscript/as_debugger.h、modules/angelscript/as_debugger.cpp、modules/angelscript/as_script_language.cpp。

- [ ] 从当前帧函数 GetModule 获取 GetGlobalVarCount/GetGlobalVar/GetAddressOfGlobalVar，调用 decode_var，遵守深度/元素限制。
- [ ] 测试断点期间读取 int、String、命名空间全局变量，未暂停返回空且不修改输出。

### 任务 4：自动化验证

**文件：** 修改 modules/angelscript/tests/test_angelscript_debugger.h、modules/angelscript/tests/test_angelscript_debugger.cpp。

- [ ] 测试可恢复断点、增删与禁用、step/next/out、递归、跨 section、循环重复命中、栈/局部/全局读取、执行边界收尾。
- [ ] 测试 line_poll 触发 core:break 式暂停可以看到 AS 上下文与正确语言。
- [ ] 用派生 EngineDebugger 测试替身只替代传输，不替代 AS 行回调、ScriptDebugger 和实际 Execute。
- [ ] 构建：cmd /c ".superpowers\sdd\2026-09-30-angelscript-hot-update-phase1\build-tests.cmd module_mono_enabled=yes"。
- [ ] 运行调试器测试和 AngelScript 全套测试；出现非零退出调查原因，不视为通过。
- [ ] 尝试真实远程连接，检查 debug_enter/get_stack_dump/get_stack_frame_vars/step/next/out/continue；若 GUI 验收不可用明确记录，不用单元测试冒充 GUI 验收。

### 任务 5：交付

**文件：** 创建 docs/angelscript-debugger-stepping.md。

- [ ] 记录编辑器使用方法、构建与测试结果、运行中暂停、异常限制、不可执行行/无行信息字节码限制。
- [ ] 审查 diff，不破坏现有对象安全、调试变量递归限制和性能优化。

## 执行记录（2026-10-13）

核心实现、行断点/单步 VM 回归、真实 TCP step/out/next/continue 协议验证与文档已落地。VM 回归已覆盖递归、循环重复命中、删除/禁用断点、line_poll 暂停查询、模块全局 String/命名空间变量、跨模块宿主重入和结束后现场清理。

原始清单保留为验收项，不把未逐项验证的内容打勾：人工 GUI 操作、所有构造/析构及全局初始化路径、全引擎测试、多线程并发暂停、混合语言全部组合、性能重测仍未完成。最终构建/测试数字见 [调试器说明](<../../docs/angelscript-debugger-stepping.md>)。

## 自审

核心需求均有对应实现和真实 VM 测试。调试器命令无需修改 Godot 协议；跨语言边界复用全局线程局部 depth。最重要的验证风险是 AS 内部 context 以及跨回调作用域；必须覆盖并记录未验证路径。
