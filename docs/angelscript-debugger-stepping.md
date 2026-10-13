# AngelScript 断点与单步调试

## 交付范围

本次实现把 AngelScript 的执行控制接入 Godot 原有 ScriptDebugger/RemoteDebugger 协议，不新增第二套调试界面。用户行断点、暂停、继续、单步进入（step）、单步跳过（next）、单步跳出（out）已经有真实 VM 自动化验证；真实 TCP 连接验证了行断点、step/out/next/continue、调用栈、局部变量、模块全局变量读取。

这不是“所有调试功能与 GDScript 完全等价”的声明。任意表达式求值、条件断点、修改运行中变量、内部函数精确 profiler 不在本次实现范围内。可执行行断点不会自动迁移到附近语句。

## 编辑器使用

1. 使用包含本次源码修改的 Godot editor 构建打开项目；脚本需要保留行信息。
2. 在 AngelScript 编辑器里对可执行语句所在行切换断点，运行项目并保留远程调试连接。
3. 暂停后使用 Godot 自带调试器的继续、单步进入、单步跳过、单步跳出按钮。调用栈与变量使用原有面板读取。
4. 运行中暂停依赖脚本执行行回调调用 line_poll；如果当前长时间停留在原生宿主函数内，需等待它返回或下一次 AS 行回调，不能抢占任意原生代码。

以上是预期编辑器操作流程；本次验收使用测试替身和真实 TCP 协议客户端，没有人工点击编辑器按钮完成 GUI 验收。TCP 验收不能替代编辑器断点持久化、按钮状态、源代码跳转等 GUI 验收。

## 实现机制

- ASDebugger 安装 SetLineCallback 与 SetExceptionCallback；无活跃调试器时行回调立即返回。
- 行回调读取 GetCallstackSize，以实际栈深变化同步 ScriptDebugger 的 depth；step 使用 depth=-1/lines_left=1，next 使用 depth=0，out 使用 depth=1。
- 同一源码行重复出现不能去重：循环迭代及同一行多个语句可能产生合法的多次停止。
- 回调暂停和 line_poll 期间暴露线程局部 live context，并保存/恢复 break_language。调试查询不使用 Execute 返回后已失效的栈地址。
- 宿主 call_function、脚本实例工厂、AS RequestContext/ReturnContext 包含执行范围，返回时扣除当前栈深；嵌套宿主回调进入另一脚本模块的 next 行为已测试。
- 模块全局变量通过当前栈帧的函数模块获取，使用现有 decode_var 安全解码；遵守递归深度与元素上限。命名空间全局变量显示为 namespace::name。
- 用户断点为可继续、非错误暂停；异常仍走原有错误上报。跳过断点/忽略错误暂停由 Godot 原有 RemoteDebugger 逻辑处理。

## 验证与重现

验证平台：Windows x64、MSVC、AngelScript 2.38、Godot 自定义 mono editor。构建版本输出为 4.8.dev.custom_build.c62001d26；该 hash 是构建标识，当前调试器修改尚未提交，不能把它理解为包含全部修改的提交号。

构建命令：

    cmd /c ".superpowers\sdd\2026-09-30-angelscript-hot-update-phase1\build-tests.cmd module_mono_enabled=yes"

行断点/单步回归命令：

    bin\godot.windows.editor.x86_64.mono.console.exe --headless --test --test-case="*debugger line*"

AngelScript 全模块回归命令：

    bin\godot.windows.editor.x86_64.mono.console.exe --headless --test --test-case="*[AngelScript]*"

真实远程协议命令：

    python modules\angelscript\tests\debugger_protocol_test.py --editor bin\godot.windows.editor.x86_64.mono.console.exe --output docs\angelscript-debugger-protocol-trace.json

协议夹具先在 res://main.as 的调用 child 行停止，step 进入 child，out 返回 _ready，next 到输出行，continue 后输出校验值 6 并正常退出。自动化断言检查暂停都可继续且有栈，停止行依次为 9、5、10、11，局部 x 最终为 4、全局 total 为 2，以及四次 debug_exit。

真实 TCP 原始消息见 [协议记录](<angelscript-debugger-protocol-trace.json>)，测试程序见 [debugger_protocol_test.py](<../modules/angelscript/tests/debugger_protocol_test.py>)，VM 回归见 [test_angelscript_debugger.cpp](<../modules/angelscript/tests/test_angelscript_debugger.cpp>)。

行断点回归覆盖循环重复命中、删除与禁用断点、step/next/out、递归栈深、运行中暂停式 line_poll 查询、上下文复用、跨模块宿主重入、局部变量、String/命名空间模块全局变量和结束后 live context 清理。完整模块回归还包含既有异常、对象安全和变量解码测试。

## 明确限制与剩余验收

- 未捕获 AS 异常不可继续；异常时栈可检查，但不能保证异常后恢复执行。
- 剥除调试信息的字节码不能提供可靠源码行和局部变量；应保留行信息。
- 源路径必须匹配编译 section；调试项目使用 res:// 路径，直接编译测试使用独立模块 section。
- 表达式查看仍只支持已有的局部变量名/this 成员名查表，不支持完整表达式计算。
- 变量展示沿用既有递归/元素限制及 ObjectID 安全解码，不修改对象所有权模型。
- 未完成人工编辑器 GUI 验收、所有跨语言混合栈组合、多线程并发暂停压力测试，也未对 AS 构造/析构和所有全局初始化路径逐项做真实 TCP 验收。
- 无调试器快速返回并不等于零开销；本次没有重新量化安装行回调对性能的影响，不能宣称性能完全无回退。
- 本次运行 AngelScript 模块测试，不等于运行 Godot 全引擎所有测试。

## 最终测试结果

最终源码构建成功，退出码 0；AngelScript 全模块测试 179/179 通过，3678/3678 断言通过，退出码 0。此前单独行断点/单步回归 108/108 断言通过，其最终源码也包含在本次全模块运行中。真实 TCP 协议脚本 PASS，四个可恢复断点、停止行/函数、局部/全局变量、debug_exit 与最终校验值均通过断言，退出码 0。git diff --check 与 Python 语法检查通过。

全模块测试 stderr 中出现宿主桥未安装与无默认工厂诊断，它们属于既有负向用例；doctest 最终统计没有失败，不应把这些预期诊断说成“stderr 完全干净”。构建仍有 ANGLE 可选依赖提示，不影响本次编译结果。

初次协议夹具使用不受支持的 String(int) 表达式触发漫长全绑定回退，已改为内建 as_log_int；协议读取器随后补齐 PackedStringArray 解码，并把输出断言改为去除换行后比较 6。最终成功记录来自修正后重新运行，不采用失败运行结果。
