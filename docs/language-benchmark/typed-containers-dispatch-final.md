# AngelScript 方案 B：强类型容器、批处理与派发计划结果

## 实现摘要

本轮新增固定强类型引用容器，底层直接使用 Godot `Vector<T>` / `HashMap<K,V>`：

- 数组：Int64、Float64、Vector2、Vector3、Color、StringName、ObjectID。
- Map：Int64→Int64，以及 StringName→Int64/Float64/Vector2/Color/ObjectID。
- 批处理：sum、min/max、标量运算、Vector add/add_scaled、失效对象清理。
- 批量互转：五种 Packed 数组，以及 StringName/Object Array。
- 派发：每实例缓存函数和 0–4 个标量参数/返回布局，避免反复 `GetParam()`。

对象容器保存 ObjectID，不拥有 Godot Object；读写句柄沿用模块的非拥有句柄编码。

## 环境与方法

- Godot：4.8-dev custom mono，MSVC x64。
- CPU：AMD Ryzen 7 7700。
- 每负载进程内预热后重复 7 次；before/after 各运行 3 个独立进程。
- 聚合值：3 轮 `min_us` 的最小值，换算 ns/op。
- 原始数据：`data/typed-containers-before.csv`、`data/typed-containers-after.csv`。

## 结果

| 负载 | AS before | AS after | GDScript after | AS/GD | 结论 |
|---|---:|---:|---:|---:|---|
| typed_i64：push + 逐项读取 | 67.31 | 65.21 | 37.95 | 1.72 | 未超过 GD |
| typed_vec2：批量 add_scaled | 2397.14 | 3.09 | 44.49 | 0.07 | AS 快约 14.4× |
| typed_map：StringName→int64 set/get | 3190.67 | 80.97 | 97.08 | 0.83 | AS 快约 1.20× |
| batch_sum：构建 + 批量求和 | 68.81 | 31.22 | 28.29 | 1.10 | 接近，未达到硬门槛 |
| Object.call dispatch | 92.44 | 91.12 | 59.68 | 1.53 | 缓存布局仅小幅改善 |

单位：ns/op；越小越好。

### 门槛判定

- Int64Array push + read ≤ GD：**未通过**。
- Vector2Array 批处理 ≤ GD × 0.7：**通过**，实际约 GD × 0.07。
- batch sum ≤ GD × 0.5：**未通过**。原因是该端到端负载仍包含逐项 push；C++ `sum()` 本身已批量化，但构建成本占主导。
- StringNameInt64Map ≤ GD Dictionary：**通过**。
- Dispatch 目标 ≤60 ns/op：**未通过**；无回退，约改善 1.4%。

## 分析

1. **批处理是最有效路线。** Vector2 循环移入 C++ 后，每元素约 3 ns，显著超过 GDScript。
2. **固定 Map 有效。** 去除 Variant 键值装箱后已超过 GDScript 类型稳定 Dictionary。
3. **逐元素调用仍受 AngelScript callfunc 成本约束。** Int64Array 每次 `get()` 仍是一条脚本→原生方法调用，所以即使存储原生化，也没有超过 GDScript 的内建 Packed 数组字节码路径。
4. **派发布局缓存不是主要瓶颈。** 方法名此前已经缓存；新增参数布局缓存只省掉 `GetParam()`，Prepare/Execute/context 状态与 Variant 返回仍占主要成本。要达到 60 ns/op 需要更深的 context/入口专用化或 VM intrinsic。
5. **方案 B 的适用方式明确：** 让粗粒度批处理和固定 Map 超过 GD，而不是承诺每个逐元素 API 都超过 GD。

## 正确性验证

- AngelScript 测试：178 passed，0 failed，3570 assertions。
- 三轮 after 基准：153 条数据，逐轮同负载三语言校验和一致。
- 全量 Godot 测试：1603 passed / 1 failed / 3 skipped；唯一失败为既有 GDScript 输出快照 `call_native_static_method.gd`，输出含 stray Node 行，与 AngelScript 容器断言无关。AngelScript 子集全部通过。

## 已知限制

- 当前原生调用 ABI按项目既有策略只在已验证的 MSVC x64 构建使用；其他平台仍需专门验证。
- Object 容器为弱引用，不保活对象。
- 首期没有为所有 Map 实现 Dictionary 批量导入导出；核心固定存储和 Packed/Array 转换已经落地。
- 动态 Array/Dictionary 原 API 保持不变，其性能不会因新增容器自动提升。

## 建议

- 数值逐项访问若必须超过 GD，下一步应添加 VM intrinsic 或脚本侧块访问，而不是继续增加普通 C++ thunk。
- 给 Int64Array 增加 `assign_range(start,count)`、`sum_masked(mask)` 等真实业务批处理，避免构建阶段逐项 push。
- Dispatch 若要低于 60 ns/op，应独立研究预 Prepare frame、专用零参执行入口或修改 AngelScript context；不可牺牲嵌套调用、Debugger、Profiler 语义。
