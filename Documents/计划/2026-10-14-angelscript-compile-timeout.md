# AngelScript 编译回退超时定位实施计划

**目标：** 修复用户已确认优先处理的编译失败/lazy绑定路径超时，并恢复完整模块回归后再进行成本测量。

**架构：** 先以最小语法错误用例及临时阶段日志定位Build/错误映射/全量绑定的耗时或死循环，再针对证实根因添加回归。生产生命周期、native ABI及POD配置不变；修复不通过忽略诊断或取消必要绑定来掩盖问题。

**技术栈：** Godot C++ tests、AngelScript公开接口、Windows有界子进程、现有回归。

---

## 已有失败证据

当前和旧candidate的全套回归均180秒超时；当前5项编译/错误/lazy用例20秒超时，旧candidate最小语法错误30秒超时。cost短测160断言与186不同隔离用例通过；不能替代全套验证。

### 任务1：定位

**检查：** [编译入口](<../../modules/angelscript/as_engine.cpp>)、[lazy注册](<../../modules/angelscript/binding/as_binding_lazy.cpp>)、[注册器](<../../modules/angelscript/binding/as_binding_registry.cpp>)、[对象注册](<../../modules/angelscript/binding/as_binding_object.cpp>)。

- [ ] 加临时阶段/类名日志（构建后仅运行20秒最小语法错误），定位最后推进位置；记录日志，不变更行为。
- [ ] 检查该循环的进度条件、注册重入、错误消息增长或锁边界；实际证据确认后选择最小修复，移除临时日志。

### 任务2：回归与修复

**测试：** [引擎测试](<../../modules/angelscript/tests/test_angelscript_engine.h>)、[lazy测试](<../../modules/angelscript/tests/test_angelscript_lazy_binding.cpp>)，必要时新增独立测试。

- [ ] 保留最小语法错误既有失败回归，再添加“错误后编译有效脚本可恢复”和“绑定失败不重复扫描/不递增无限错误”的精确根因回归；错误报告必须仍含源诊断。
- [ ] 最小修复根因，返回明确错误不污染后续有效编译；不改变调用ABI/对象销毁。
- [ ] 以外部超时验证5个失败用例，再跑全套AngelScript；完整通过才继续性能测量。

测试验收形状：

```cpp
CHECK_FALSE(as_engine_compile_module("bad", "int main( { return 42; }", &error));
CHECK_FALSE(error.is_empty());
CHECK(as_engine_compile_module("good_after_bad", "int main() { return 42; }", &error));
```

### 任务3：记录和恢复主计划

- [ ] 报告真实根因、影响范围、before超时/after结果及残余限制；保留旧证据，不额外自动提交。
- [ ] 恢复成本归因计划的编译/回归/有界六进程步骤，更新二进制哈希，不能与修复前混用。

## 根因证据与精确修复范围（定位后更新）

注册进度持续前进，非固定类死循环。800类累计method=0.433秒、property=24.265秒。vendored `asCBuilder::ValidateVirtualProperty` 为每个getter/setter遍历engine->scriptFunctions查配对访问器，ClassDB全量继承方法/属性注册使扫描呈二次增长。针对已证实根因，修订“不修改VM”的初始约束：允许仅优化原生应用对象属性的注册期查找，不改运行字节码、ABI、生命周期、脚本属性查找或全局查找规则。

**修改：** [AS注册编译器](<../../modules/angelscript/thirdparty/angelscript/source/as_builder.cpp>)。
**新增：** `modules/angelscript/tests/test_angelscript_property_registration.{h,cpp}`。

- 仅funcType=asFUNC_SYSTEM、objectType存在且非asOBJ_SCRIPT_OBJECT，搜索objectType->methods；这些是已经注册的函数id，保留同名/IsProperty/module/namespace/objectType所有过滤条件及原返回码。
- 脚本类型与全局属性保留原engine-wide扫描，因为脚本方法在声明期未必已进入方法列表。
- 测试两种注册顺序、类型不匹配负例、同名不同类隔离、indexed getter/setter、全局/脚本属性回退，以及规模化原生属性注册。负例用单独引擎，避免configFailed污染。
- 删除所有临时COMPILE_TRACE/REGISTER_TRACE/STAGE_TOTAL/SCAN_TRACE，重建后跑失败五项与完整模块回归。

## 修复验证

- 已按所属原生类型方法列表限定扫描，原脚本/全局路径与所有原过滤条件保留；临时生产插桩全部移除。
- 新增属性注册语义测试通过，语法错误+随后正常编译用例3.700秒通过。
- 单进程完整AngelScript回归192/192，4761断言通过，墙钟11.666秒；恢复主计划计时。
- 修复前记录为至少180秒仍未结束，不把它说成精确耗时，也不声称是锁死或固定类死循环。
- 补丁维护说明保存到 [LOCAL_PATCHES](<../../modules/angelscript/thirdparty/angelscript/LOCAL_PATCHES.md>)；不额外自动提交。

## 自审

先定位后修复；不预判全量回退必然是错误映射问题，不改VM来碰运气。定位日志是临时诊断，不纳入最终热路径。
