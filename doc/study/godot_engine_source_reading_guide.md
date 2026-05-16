# Godot 引擎源码导读：从启动到场景运行

## 导论

本导读面向第一次系统阅读 Godot 源码的学习者，目标不是替代 API 手册，而是建立一条能够反复使用的源码阅读路线。本文依据当前仓库源码撰写；`version.py` 显示本仓库为 Godot Engine `4.7.0 beta`。导读重点覆盖引擎启动、对象系统、资源与场景、主循环、Server 架构、模块与扩展、编辑器入口等学习 Godot 源码时最容易迷路的部分。

建议的阅读方式是：先理解“对象如何被注册和创建”，再理解“场景如何进入主循环”，最后理解“节点如何把工作委托给各类 Server”。这条路线能把 Godot 的宏、反射、资源加载、渲染和物理等看似分散的系统串在一起。

本导读主要参考的源码包括：

- 构建与模块：`SConstruct`、`methods.py`、`core/SCsub`、`scene/SCsub`、`modules/SCsub`、`modules/register_module_types.h`、`modules/register_module_types.gen.cpp`
- 启动与主循环：`platform/windows/godot_windows.cpp`、`platform/linuxbsd/godot_linuxbsd.cpp`、`platform/windows/os_windows.cpp`、`platform/linuxbsd/os_linuxbsd.cpp`、`main/main.cpp`
- 对象系统：`core/object/object.h`、`core/object/object.cpp`、`core/object/class_db.h`、`core/object/class_db.cpp`、`core/register_core_types.cpp`、`core/variant/variant.h`
- 资源与场景：`core/io/resource.h`、`core/io/resource_loader.h`、`core/io/resource_loader.cpp`、`scene/resources/packed_scene.h`、`scene/resources/packed_scene.cpp`
- 节点与 SceneTree：`scene/main/node.h`、`scene/main/node.cpp`、`scene/main/scene_tree.cpp`
- Server 架构：`servers/register_server_types.cpp`、`servers/display/display_server.cpp`、`servers/rendering/rendering_server.h`、`servers/rendering/rendering_server_default.cpp`、`servers/physics_2d/physics_server_2d.cpp`、`servers/physics_3d/physics_server_3d.cpp`、`servers/audio/audio_server.h`
- 2D/3D 节点到渲染：`scene/main/canvas_item.cpp`、`scene/2d/node_2d.cpp`、`scene/3d/node_3d.cpp`、`scene/3d/visual_instance_3d.cpp`
- 脚本、扩展与编辑器：`core/object/script_language.h`、`core/object/script_language.cpp`、`modules/gdscript/register_types.cpp`、`core/extension/gdextension.cpp`、`core/extension/gdextension_manager.cpp`、`editor/register_editor_types.cpp`、`editor/editor_node.cpp`、`editor/project_manager/project_manager.cpp`

## 学习目标

读完本文后，应能完成以下任务：

1. 说清 Godot 从平台入口进入 `Main::setup()`、`Main::setup2()`、`Main::start()`、`Main::iteration()`、`Main::cleanup()` 的总体过程。
2. 解释 `Object`、`ClassDB`、`Variant`、`ObjectDB` 分别解决什么问题，以及它们如何支撑脚本、编辑器和序列化。
3. 跟踪一个 `.tscn` 或 `.res` 文件从 `ResourceLoader::load()` 到 `PackedScene::instantiate()` 再到 `SceneTree` 的路径。
4. 理解 `Node` 的进入树、准备、处理、退出树和延迟删除的生命周期。
5. 区分场景层 API 与底层 Server API，理解节点为什么通过 `RenderingServer`、`PhysicsServer2D/3D`、`AudioServer` 等单例完成实际工作。
6. 理解模块和 GDExtension 的初始化层级，能判断某段扩展代码应该在 CORE、SERVERS、SCENE 还是 EDITOR 阶段注册。
7. 形成一条适合继续深入渲染、物理、脚本语言或编辑器的阅读路线。

## 第 1 章 源码总览与编译地图

### 1.1 顶层目录

Godot 的源码目录按“引擎基础层、运行时层、平台层、工具层”组织。新读者可以先记住以下地图：

| 目录 | 主要职责 |
| --- | --- |
| `core` | 基础类型、对象系统、Variant、资源抽象、IO、线程、OS 抽象、项目设置等。 |
| `main` | 启动流程、命令行解析、主循环调度、初始化和清理。 |
| `platform` | 各平台入口和 OS 实现，例如 Windows、LinuxBSD、macOS、Android。 |
| `servers` | 显示、渲染、音频、物理、导航、文本、XR 等底层服务接口和默认实现。 |
| `scene` | 场景树、节点、2D/3D/GUI 节点、资源类型、PackedScene。 |
| `editor` | 编辑器 UI、插件、导入导出、项目管理器、Inspector、文件系统等。 |
| `modules` | 可选模块，例如 GDScript、GodotPhysics、Jolt、文本后端、格式支持等。 |
| `drivers` | 一些底层驱动或第三方库接入。 |
| `thirdparty` | 第三方依赖源码。 |
| `doc/classes` | 类参考文档数据，既服务文档，也服务编辑器帮助。 |
| `tests` | 单元测试和集成测试入口。 |

一个实用原则是：如果某个问题涉及“语言无关的基础能力”，先看 `core`；涉及“游戏对象和节点”，先看 `scene`；涉及“实际渲染、音频、物理执行”，先看 `servers`；涉及“窗口、输入和平台差异”，看 `platform`；涉及“编辑器面板和工具行为”，看 `editor`。

### 1.2 SCons 构建入口

Godot 使用 SCons。根目录 `SConstruct` 是构建的总入口，它做几件关键事情：

1. 检查 SCons 和 Python 版本。当前源码要求 SCons 至少 `4.0`，Python 至少 `3.9`。
2. 扫描 `platform/*/detect.py`，收集可用平台及其构建选项。
3. 解析 `target` 等构建参数。常见目标包括 `editor`、`template_release`、`template_debug`。源码中会据此定义 `TOOLS_ENABLED`、`DEBUG_ENABLED`、`DEV_ENABLED` 等宏。
4. 调用 `methods.detect_modules()` 发现模块。一个模块至少需要 `register_types.h`、`SCsub`、`config.py`。
5. 依次加载 `core/SCsub`、`servers/SCsub`、`scene/SCsub`、`editor/SCsub`、`drivers/SCsub`、`platform/SCsub`、`modules/SCsub`、`tests/SCsub`、`main/SCsub` 和具体平台的 `platform/<platform>/SCsub`。

`methods.py` 中的 `detect_modules()`、`is_module()`、`module_add_dependencies()`、`module_check_dependencies()`、`sort_module_list()` 是理解模块构建顺序的关键函数。模块不只是把文件加入编译，还会参与初始化注册；这一点在后文的 `initialize_modules()` 中会再次出现。

### 1.3 生成代码与注册代码

Godot 的很多能力依赖生成代码。例如 `modules/SCsub` 会生成：

- `modules_enabled.gen.h`：记录哪些模块启用。
- `register_module_types.gen.cpp`：统一调用各模块的初始化和反初始化函数。

这意味着阅读 Godot 时要区分三类代码：

1. 手写核心代码，例如 `core/object/object.cpp`。
2. 构建期生成代码，例如 `modules/register_module_types.gen.cpp`。
3. 宏展开后的注册代码，例如 `GDCLASS`、`GDREGISTER_CLASS` 参与的 ClassDB 注册。

源码导读阶段不必立即理解所有生成细节，但必须知道“类如何进入 ClassDB”和“模块如何进入初始化序列”不是自然发生的，而是由构建和注册系统显式完成。

### 本章小结

Godot 源码不是单一主程序，而是一个由构建系统组合出来的多层运行时。`SConstruct` 决定编译哪些平台、模块和宏；`SCsub` 决定各目录如何生成库；注册代码决定类、资源和 Server 如何在运行时被发现。

### 习题

1. 在 `SConstruct` 中找到 `target` 的处理逻辑，说明 `editor` 与 `template_release` 在宏定义上至少有什么不同。
2. 在 `methods.py` 中阅读 `is_module()`，列出一个合法模块必须包含的文件。
3. 打开 `modules/SCsub`，观察 `register_module_types.gen.cpp` 是如何被加入编译的。

## 第 2 章 从平台入口到 Main 主循环

### 2.1 平台入口

在 Windows 上，入口位于 `platform/windows/godot_windows.cpp`。代码创建 `OS_Windows`，把 UTF-16 命令行参数转换为 UTF-8，然后执行：

```text
Main::setup(...)
Main::start()
os.run()
Main::cleanup()
```

LinuxBSD 的入口位于 `platform/linuxbsd/godot_linuxbsd.cpp`，总体结构相同，只是平台初始化和 CPU 能力检查不同。平台文件的核心职责不是运行游戏逻辑，而是把平台相关的命令行、窗口、OS 对象准备好，然后把控制权交给 `Main`。

平台的 `OS::run()` 实现在 `platform/windows/os_windows.cpp` 和 `platform/linuxbsd/os_linuxbsd.cpp` 中。它们会初始化 `main_loop`，循环处理显示事件，并反复调用 `Main::iteration()`。当 `Main::iteration()` 返回退出条件后，平台层结束主循环并进入清理阶段。

### 2.2 Main 的五个阶段

`main/main.cpp` 是理解 Godot 启动的第一主文件。可以把它拆成五个阶段：

| 阶段 | 主要职责 |
| --- | --- |
| `Main::setup()` | 建立底层单例和核心类型，解析命令行和项目设置，完成 core 层初始化。 |
| `Main::setup2()` | 初始化高层 Server、DisplayServer、RenderingServer、AudioServer、场景类型、脚本语言和编辑器类型。 |
| `Main::start()` | 根据命令行和项目设置选择运行模式，创建 `MainLoop`，加载主场景、编辑器或项目管理器。 |
| `Main::iteration()` | 每帧调度物理、处理、消息队列、渲染、音频、调试器和性能统计。 |
| `Main::cleanup()` | 按初始化的反方向释放脚本、编辑器、场景、Server、模块、扩展和 core 单例。 |

### 2.3 `Main::setup()`

`Main::setup()` 属于低层初始化。它会建立主线程标记，初始化 OS，创建 `Engine`，注册 core 类型和 driver 类型，创建 `InputMap`、`ProjectSettings`、`TranslationServer`、`Performance`、`PackedData`、`ZipArchive` 等基础对象，并解析命令行参数。

在这个阶段，`core/register_core_types.cpp` 中的 `register_core_types()` 非常重要。它会调用 `ObjectDB::setup()`、`StringName::setup()`，注册 `Object`、`RefCounted`、`Resource`、`MainLoop`、`ProjectSettings`、`Input`、`Engine` 等核心类，并初始化 `Variant::register_types()`、资源加载器、资源保存器和脚本抽象。

可以把 `Main::setup()` 理解为“让引擎具备描述自身和读取项目的能力”。此时还不是完整游戏运行时。

### 2.4 `Main::setup2()`

`Main::setup2()` 开始进入高层运行时。源码中可以看到它创建或初始化：

- `TextServerManager`
- `PhysicsServer2DManager` 和 `PhysicsServer3DManager`
- `DisplayServer`
- `RenderingServerDefault`
- `AudioServer`
- XR、导航、主题、场景类型、脚本语言
- 编辑器类型和编辑器模块，在 `TOOLS_ENABLED` 下启用

`DisplayServer::create()` 位于 `servers/display/display_server.cpp`，它根据注册的显示后端创建平台显示服务。`RenderingServerDefault` 位于 `servers/rendering/rendering_server_default.cpp`，它封装渲染命令队列、可选渲染线程、viewport/canvas/scene 的渲染调度。物理 Server 由 `PhysicsServer2DManager` 和 `PhysicsServer3DManager` 根据项目设置或默认值创建；若没有可用实现，则回退到 Dummy Server。

这一阶段还会按层级初始化模块和 GDExtension。后文会说明 `MODULE_INITIALIZATION_LEVEL_SERVERS`、`SCENE`、`EDITOR` 的含义。

### 2.5 `Main::start()`

`Main::start()` 决定本次进程到底做什么：

- 运行命令行工具，例如导出、文档生成、转换等。
- 创建自定义 `MainLoop`。
- 默认创建 `SceneTree`。
- 在编辑器模式下创建 `EditorNode` 并加入场景树。
- 在项目管理器模式下创建 `ProjectManager`。
- 在游戏运行模式下加载项目主场景。

游戏运行时，主场景通常通过 `ResourceLoader::load()` 加载为 `PackedScene`，然后调用 `PackedScene::instantiate()` 得到节点树，再加入 `SceneTree`。这条路径连接了资源系统、对象系统和节点生命周期，是本导读后半部分的核心。

### 2.6 `Main::iteration()`

`Main::iteration()` 是每帧调度中心。它大致执行：

1. 更新时间和帧步长。
2. 执行一个或多个物理步。
3. 调用 `SceneTree::iteration_prepare()`。
4. 同步并查询 `PhysicsServer3D`、`PhysicsServer2D`。
5. 调用 `main_loop->physics_process()`。
6. 处理导航、物理结束同步和消息队列。
7. 调用 `main_loop->process()`。
8. 刷新消息队列和导航处理。
9. 调用 `RenderingServer::sync()` 和 `RenderingServer::draw()`。
10. 更新 GDExtension、脚本语言、AudioServer、调试器、性能统计和帧延迟。

这个函数解释了为什么 Godot 的节点处理、物理、渲染、音频不是随意交错的。它们由主循环按固定顺序推进。

### 2.7 `Main::cleanup()`

清理顺序大体是初始化顺序的逆序。Godot 会先清理更高层的编辑器、场景、脚本、Server、模块和扩展，再释放底层 core 单例。这样可以避免高层对象在析构时访问已经不存在的基础服务。

读清理代码有一个实际价值：如果你在引擎退出时遇到泄漏、悬挂对象或单例访问崩溃，`Main::cleanup()` 和对应的 `unregister_*_types()` 往往是入口。

### 本章小结

Godot 的启动不是一个简单的 `main()` 调用链，而是平台入口、OS 主循环和 `Main` 初始化阶段共同组成。阅读 `main/main.cpp` 时，应优先抓住五个阶段，而不是被命令行选项和平台分支淹没。

### 习题

1. 在 `main/main.cpp` 中找到 `Main::start()` 创建 `SceneTree` 的位置，说明什么情况下会使用自定义 `MainLoop`。
2. 在 `Main::iteration()` 中标出物理、普通处理、渲染和音频更新的相对顺序。
3. 在平台 `OS::run()` 中确认 `Main::iteration()` 的返回值如何影响进程退出。

## 第 3 章 Object、ClassDB 与 Variant

### 3.1 Object 是 Godot 运行时对象的根

`core/object/object.h` 中的 `Object` 是 Godot 绝大多数引擎对象的根类。它提供：

- 实例 ID 和 `ObjectDB` 注册。
- 属性 `set()`、`get()` 和属性列表。
- 方法调用 `call()`、`callp()` 和延迟调用。
- 信号连接、断开和发射。
- 元数据。
- 脚本实例挂载。
- GDExtension instance binding。
- 通知系统 `notification()`。

`Object` 构造时通过 `_construct_object()` 注册到 `ObjectDB`。析构时会断开信号连接、移除 ObjectDB 记录并释放 instance binding。`ObjectDB` 使用带校验位的 `ObjectID` 管理对象引用，减少悬挂 ID 误用的风险。

### 3.2 先用 Node2D 串起 GDCLASS、ClassDB 和 `_bind_methods()`

理解 `GDCLASS` 和 `ClassDB` 时，最好不要从宏展开开始，而是先跟一个真实类。`Node2D` 是很好的例子，因为它会出现在场景、脚本、Inspector 和渲染链路中。

`scene/2d/node_2d.h` 中的类声明是：

```cpp
class Node2D : public CanvasItem {
	GDCLASS(Node2D, CanvasItem);
	...
};
```

这行 `GDCLASS(Node2D, CanvasItem)` 不是“标记给人看”的注释，而是 Godot 对象模型的入口。它让 `Node2D` 知道自己的 Godot 类名是 `Node2D`，父类是 `CanvasItem`，并提供 `get_class_static()`、`initialize_class()`、属性列表转发、`Object::cast_to()` 所需的类型检查入口等样板代码。

但是只有 `GDCLASS` 还不够。类还必须被注册。`scene/register_scene_types.cpp` 的 2D 类型注册区域里有：

```cpp
GDREGISTER_CLASS(Node2D);
GDREGISTER_CLASS(Sprite2D);
GDREGISTER_CLASS(Line2D);
```

`GDREGISTER_CLASS(Node2D)` 会调用 `ClassDB::register_class<Node2D>()`。注册时会做几件关键事：

1. 调用 `Node2D::initialize_class()`，确保父类先初始化。
2. 把 `Node2D` 和父类关系加入 `ClassDB::classes`。
3. 执行 `Node2D::_bind_methods()`，把方法、属性、常量等写进 `ClassDB`。
4. 给 `ClassInfo` 设置 `creation_func`，使 `ClassDB::instantiate("Node2D")` 能真正创建 C++ 对象。

`scene/2d/node_2d.cpp` 中的绑定代码说明了它具体暴露了什么：

```cpp
void Node2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_position", "position"), &Node2D::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &Node2D::get_position);
	ClassDB::bind_method(D_METHOD("rotate", "radians"), &Node2D::rotate);

	ADD_GROUP("Transform", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position", ...), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation", ...), "set_rotation", "get_rotation");
}
```

这里要注意两层含义：

1. `ClassDB::bind_method()` 暴露“可调用的方法”。脚本中的 `node.rotate(0.5)`、`Object::call("rotate", ...)`、编辑器自动补全和文档生成都依赖这张表。
2. `ADD_PROPERTY()` 暴露“可读写的属性”。`position` 不是直接绑定到 `Node2D::position` 成员变量，而是绑定到 `"set_position"` 和 `"get_position"`。Inspector、场景序列化和 `Object::set("position", value)` 都会通过这对 setter/getter 工作。

所以可以把一个 Godot C++ 类的接入过程记成三步：

```text
头文件声明：GDCLASS(Node2D, CanvasItem)
初始化注册：GDREGISTER_CLASS(Node2D)
反射内容：Node2D::_bind_methods()
```

少了第一步，类不具备 Godot 对象模型所需的静态类型入口；少了第二步，ClassDB 不知道这个类存在；少了第三步，类虽然能创建，但脚本、Inspector 和序列化看不到它的具体 API。

### 3.3 `GDCLASS` 到底解决了什么问题

C++ 本身没有 Godot 需要的运行时反射能力。引擎不能只靠 C++ 的 `typeid` 或虚函数表解决这些问题：

- 从字符串 `"Node2D"` 找到类并创建对象。
- 判断一个对象是否是某个 Godot 类或其子类。
- 不知道 C++ 静态类型时，按字符串调用 `"set_position"`。
- 遍历属性列表，让编辑器 Inspector 自动生成控件。
- 保存场景时只保存需要持久化的属性。
- 让脚本语言、GDExtension 和 C++ 内置类共享同一套类型系统。

`GDCLASS` 生成的样板代码正是这些能力的基础。以 `core/object/object.h` 中的宏为线索阅读，可以看到它主要提供：

- `self_type`、`super_type`：让模板注册代码知道“当前类”和“父类”。
- `get_class_static()`：返回稳定的 Godot 类名。
- `_get_typev()` 和 `get_gdtype_static()`：把对象连接到 Godot 自己的类型描述 `GDType`。
- `initialize_class()`：递归初始化父类，把当前类加入 `ClassDB`，再调用 `_bind_methods()`。
- `_get_property_listv()`、`_setv()`、`_getv()` 等转发入口：让属性查询、读写和动态属性能沿继承链工作。
- `get_class_ptr_static()` 与 `is_class_ptr()`：服务 `Object::cast_to<T>()` 这类高频类型判断。

这也是为什么 Godot 用宏，而不是要求每个类手写这些函数。对象系统要在上千个类中保持统一，宏把重复且容易写错的部分固定下来，把每个类真正需要写的内容集中到 `_bind_methods()` 和自身逻辑中。

### 3.4 ClassDB 是运行时类目录

`core/object/class_db.h` 和 `core/object/class_db.cpp` 是反射系统中心。它内部的 `ClassDB::ClassInfo` 可以理解为一张“类卡片”，记录：

- 类名、父类指针和 API 类型。
- `MethodBind` 表，也就是方法名到 C++ 调用封装的映射。
- `PropertyInfo` 列表、属性名到 setter/getter 的映射。
- `creation_func`，也就是如何创建这个类的对象。
- 是否抽象、是否虚类、是否暴露给脚本、是否编辑器专用、是否来自 GDExtension 等标志。

`GDREGISTER_CLASS`、`GDREGISTER_ABSTRACT_CLASS`、`GDREGISTER_VIRTUAL_CLASS` 的差异主要体现在这张类卡片上：

| 注册宏 | 常见用途 | 是否能 `ClassDB::instantiate()` |
| --- | --- | --- |
| `GDREGISTER_CLASS(T)` | 普通可创建类，如 `Node2D`、`Sprite2D` | 可以 |
| `GDREGISTER_ABSTRACT_CLASS(T)` | 抽象基类，如不能直接创建的接口型基类 | 不可以 |
| `GDREGISTER_VIRTUAL_CLASS(T)` | 暴露 API 但标记为 virtual 的类 | 有创建函数，但工具层会看到 `is_virtual` 标志 |
| `GDREGISTER_INTERNAL_CLASS(T)` | 引擎内部类，不暴露给普通 API | 默认公开实例化路径会拒绝未暴露类 |

创建实例时的路径在 `ClassDB::_instantiate_internal()` 中。简化后是：

```text
ClassDB::instantiate("Node2D")
  -> 查 ClassDB::classes["Node2D"]
  -> 检查类是否存在、是否禁用、是否暴露、是否能创建
  -> 如果是 GDExtension 类，走扩展创建函数
  -> 否则调用 ClassInfo::creation_func
  -> creator<Node2D>() new 出 C++ 对象并做初始化
```

这条路径解释了为什么场景文件只保存类型名也能恢复节点。场景里可以有类似这样的信息：

```text
[node name="Player" type="Node2D"]
position = Vector2(120, 80)
```

`PackedScene::instantiate()` 读到 `type="Node2D"` 后，会在 `scene/resources/packed_scene.cpp` 中调用 `ClassDB::instantiate(snames[n.type])`。创建出对象后再转成 `Node`，并继续恢复属性、子节点、owner、组和信号连接。

### 3.5 `_bind_methods()` 绑定的是“跨系统契约”

`_bind_methods()` 不是只给脚本用。它绑定出来的信息至少被这些系统消费：

| 消费者 | 具体依赖 |
| --- | --- |
| GDScript/C# 等脚本 | 根据 `MethodBind` 调用 C++ 方法，根据属性表读写属性。 |
| Inspector | 通过 `Object::get_property_list()` 得到 `PropertyInfo`，按类型和 hint 生成编辑控件。 |
| 场景保存 | `PackedScene` 遍历属性列表，只保存带 `PROPERTY_USAGE_STORAGE` 的值。 |
| 场景加载 | 根据保存的属性名调用 `Object::set()`，最终进入 ClassDB 记录的 setter。 |
| 文档与自动补全 | 读取方法、参数、默认值、属性、枚举和信号元数据。 |
| GDExtension | 外部类也要注册类似的类信息，才能和内置类使用同一套调用/属性系统。 |

一个属性写入的真实链路可以这样跟：

```text
object.set("position", Vector2(120, 80))
  -> Object::set()
  -> ClassDB::set_property(object, "position", value)
  -> 查 ClassInfo::property_setget["position"]
  -> 找到 ADD_PROPERTY 中登记的 "set_position"
  -> MethodBind::call()
  -> Node2D::set_position()
```

一个方法调用的链路类似：

```text
object.call("rotate", 0.5)
  -> Object::callp()
  -> 先尝试脚本实例方法
  -> ClassDB::get_method(get_class_name(), "rotate")
  -> MethodBind::call()
  -> Node2D::rotate()
```

这就是 Godot 设计里很重要的一点：编辑器、脚本、序列化不直接依赖某个 C++ 成员变量或具体类实现，而是依赖 ClassDB 中登记的元数据和调用入口。这样引擎可以用同一套机制处理内置类、脚本类和扩展类。

### 3.6 阅读 ClassDB 相关代码时的定位方法

遇到 `GDCLASS`、`ClassDB` 或 `_bind_methods()` 时，可以按下面顺序查：

1. 在头文件找 `GDCLASS(当前类, 父类)`，确认继承链。
2. 用 `rg "GDREGISTER_.*当前类"` 找注册位置，确认它在哪个初始化层级进入 ClassDB。
3. 找 `当前类::_bind_methods()`，确认脚本/编辑器能看到哪些方法、属性、信号和常量。
4. 如果关心创建，断点放在 `ClassDB::_instantiate_internal()` 和对应类构造函数。
5. 如果关心属性，断点放在 `ClassDB::set_property()`、`ClassDB::get_property()`、目标 setter/getter。
6. 如果关心脚本调用，断点放在 `Object::callp()` 和 `ClassDB::get_method()`。
7. 如果关心 Inspector 或保存，断点放在 `Object::get_property_list()` 和 `PackedScene` 的属性遍历逻辑。

这个顺序能避免一开始就陷进宏展开。先看“类在哪里注册、注册了什么、谁在消费这些信息”，再回头看宏的细节，会容易很多。

### 3.7 Variant 是统一值容器

`core/variant/variant.h` 中的 `Variant` 是脚本、属性、信号、序列化和动态调用的公共值类型。它支持 `NIL`、`BOOL`、`INT`、`FLOAT`、`STRING`、数学类型、`RID`、`OBJECT`、`CALLABLE`、`SIGNAL`、`DICTIONARY`、`ARRAY` 和各种 packed array。

源码注释说明，在 `real_t` 为 float 时 `Variant` 为 24 字节，double 时为 40 字节。它对部分大对象和集合类型使用动态分配。`Variant` 的存在让 C++ 方法绑定和脚本语言之间可以用统一 ABI 传参。

继续用 `Node2D::set_position(Vector2)` 举例：脚本、场景文件或 Inspector 写入 `position` 时，传给 `Object::set()` 的值先是一个 `Variant`。`ClassDB::set_property()` 找到 setter 后，`MethodBind` 再把这个 `Variant` 解包成 C++ 期望的 `Vector2` 参数。反过来，`get_position()` 返回的 `Vector2` 也会被装回 `Variant`，交给脚本、Inspector 或序列化系统。

这解释了为什么 `Variant` 必须出现在对象系统中心：没有它，动态属性、信号参数、脚本调用和资源保存就很难共用同一套通道。

### 3.8 MessageQueue 与延迟调用

`Object::call_deferredp()`、`set_deferred()` 会把调用推入 `MessageQueue`。主循环中多次刷新消息队列，使延迟调用在安全的时机执行。理解这一点可以解释 Godot 中常见的“当前帧不能立即删除/修改树结构，改为 deferred”的设计。

### 本章小结

`Object` 负责运行时对象能力，`GDCLASS` 让 C++ 类接入 Godot 对象模型，`GDREGISTER_CLASS` 把类放进 `ClassDB`，`_bind_methods()` 把方法和属性暴露给跨系统使用，`Variant` 负责统一动态值，`ObjectDB` 负责对象 ID 管理。它们共同支撑脚本、编辑器、序列化、GDExtension 和场景实例化。

### 习题

1. 跟踪 `Node2D`：从 `GDCLASS(Node2D, CanvasItem)` 找到 `GDREGISTER_CLASS(Node2D)`，再找到 `Node2D::_bind_methods()`。
2. 在 `ClassDB::_instantiate_internal()` 中追踪 `ClassDB::instantiate("Node2D")` 如何走到 `creator<Node2D>()`。
3. 对一个 `Node2D` 调用 `set("position", Vector2(...))`，断点观察 `Object::set()`、`ClassDB::set_property()` 和 `Node2D::set_position()` 的顺序。
4. 阅读 `Object::call_deferredp()`，说明它为什么不直接调用目标方法。

## 第 4 章 ResourceLoader 与 PackedScene

### 4.1 Resource 的职责

`core/io/resource.h` 中的 `Resource` 继承自 `RefCounted`。资源对象有路径缓存、名称、本地到场景标记、场景唯一 ID、复制方法和变更信号。它既可以代表纹理、材质、脚本等文件资源，也可以代表内嵌在场景中的子资源。

`ResourceCache` 维护路径到资源对象的映射。资源缓存的存在让同一路径的资源可以复用，也让热重载和编辑器状态管理成为可能。

### 4.2 ResourceLoader 的加载流程

`core/io/resource_loader.h` 和 `core/io/resource_loader.cpp` 定义了资源加载框架。`ResourceFormatLoader` 是具体格式加载器的抽象接口。`ResourceLoader::load()` 会处理：

- 路径本地化、UID 和相对路径。
- 缓存模式，如 `CACHE_MODE_REUSE`、`CACHE_MODE_REPLACE`、`CACHE_MODE_IGNORE`。
- 线程加载任务和加载 token。
- 自定义加载器。
- 工具模式下的时间戳和依赖信息。

因此，`ResourceLoader::load("res://main.tscn")` 并不是简单打开文件，而是在缓存、格式识别、线程任务和资源路径管理之间协调。

### 4.3 PackedScene 与 SceneState

`scene/resources/packed_scene.h` 和 `scene/resources/packed_scene.cpp` 定义了 `PackedScene`。它内部使用 `SceneState` 保存场景：

- 名称表。
- Variant 值表。
- NodePath 表。
- 节点数据，包括父节点、owner、类型名、节点名、实例信息、属性、分组。
- 信号连接数据。

实例化时，`PackedScene::instantiate()` 会根据 `SceneState` 创建节点。普通节点通过 `ClassDB::instantiate(type)` 创建；子场景实例则递归实例化 `PackedScene`；缺失类型或占位符会按源码中的策略处理。

随后它会设置属性、恢复脚本属性状态、复制 local-to-scene 资源、解析延迟的 NodePath 属性、添加子节点、设置 owner、加入组并恢复信号连接。

把上一章的 `Node2D` 放进这个流程，就能看到 ClassDB 的实际用途。文本场景中可能保存了类似信息：

```text
[node name="Player" type="Node2D"]
position = Vector2(120, 80)
```

加载后，`SceneState` 记录的不是 C++ 构造函数指针，而是类型名、属性名和值。实例化时大致发生：

```text
snames[n.type] == "Node2D"
  -> ClassDB::instantiate("Node2D")
  -> new Node2D
  -> Object::cast_to<Node>()
  -> 对保存的属性调用 node->set("position", Variant(Vector2(...)))
  -> ClassDB::set_property()
  -> Node2D::set_position()
```

这就是“为什么类名字符串、`GDCLASS`、`GDREGISTER_CLASS`、`_bind_methods()` 必须配套出现”的实际场景：场景文件不可能保存 C++ 函数地址，只能保存稳定的类型名和属性名；运行时再通过 ClassDB 把这些名字解析回真实的 C++ 对象与方法。

### 4.4 从主场景到节点树

游戏启动时，`Main::start()` 会读取项目主场景设置，调用资源系统加载 `PackedScene`，然后实例化为节点树并加入 `SceneTree`。这条链路可以概括为：

```text
ProjectSettings
  -> ResourceLoader::load()
  -> ResourceFormatLoader
  -> PackedScene
  -> PackedScene::instantiate()
  -> ClassDB::instantiate()
  -> Object::set() / ClassDB::set_property()
  -> Node::add_child()
  -> SceneTree
```

这个流程是学习 Godot 的关键主线。它把前一章的 Object/ClassDB 和下一章的 Node/SceneTree 连接起来。

### 4.5 打包场景

`PackedScene` 的 pack 逻辑会遍历由场景根节点拥有的节点，处理继承场景、实例化场景和属性存储规则。它不会简单保存整个内存树，而是保存符合 owner、属性 usage、场景实例差异等规则的数据。这也是编辑器中“保存场景”和运行时节点树不完全等价的原因。

### 本章小结

资源系统负责“从路径到 Resource”，PackedScene 负责“从 Resource 到节点树”。二者通过 `ResourceLoader` 和 `ClassDB` 与 Godot 的对象系统紧密连接。

### 习题

1. 在 `ResourceLoader::load()` 中观察缓存命中的判断，说明 `CACHE_MODE_REUSE` 的效果。
2. 在 `PackedScene::instantiate()` 中找到 `ClassDB::instantiate()` 的调用位置，说明类型名来自哪里。
3. 新建一个只包含一个 `Node2D` 的测试场景，断点跟踪它进入 `Node::add_child()` 的路径。

## 第 5 章 SceneTree 与 Node 生命周期

### 5.1 Node 是场景层基本单位

`scene/main/node.h` 中的 `Node` 继承自 `Object`。它增加了场景树关系、owner、组、处理模式、process/physics process、输入、线程处理组、ready 状态等能力。

`Node::Data` 保存父节点、子节点、owner、树指针、viewport、处理标志、组状态、ready 状态等运行时信息。阅读 `Node` 时要始终区分两种关系：

- 父子关系：决定树结构和通知传播。
- owner 关系：决定场景保存和 PackedScene 打包范围。

### 5.2 进入树

`Node::_propagate_enter_tree()` 是进入场景树的关键函数。它会设置树指针、深度、viewport 和组信息，发送 `NOTIFICATION_ENTER_TREE`，调用 `_enter_tree`，发出 `tree_entered` 信号，并递归处理子节点。

`CanvasItem`、`Node3D` 等派生类会在 `NOTIFICATION_ENTER_TREE` 中注册到自己的运行时结构。例如：

- `CanvasItem` 会进入 canvas，设置 `RenderingServer` 的 canvas item parent，处理可见性和纹理过滤继承。
- `Node3D` 会记录父级 `Node3D`，处理 world 进入、可见性和 transform dirty 状态。

这说明 Godot 的节点生命周期不是单纯的虚函数调用，而是由通知系统驱动的多层行为组合。

### 5.3 Ready 顺序

`Node::_propagate_ready()` 会先递归处理子节点，再处理当前节点的 `NOTIFICATION_READY` 和 `ready` 信号。也就是说 `_ready()` 的顺序通常是子节点先于父节点。这个顺序使父节点在 `_ready()` 中访问子节点时，子节点已经完成 ready。

### 5.4 处理与物理处理

`SceneTree::physics_process()` 和 `SceneTree::process()` 分别推进物理帧和普通帧。`Node::set_process()`、`Node::set_physics_process()` 会把节点加入或移出 SceneTree 的处理列表。

`Node::ProcessMode` 包括 `INHERIT`、`PAUSABLE`、`WHEN_PAUSED`、`ALWAYS`、`DISABLED`。SceneTree 会根据暂停状态和处理模式决定是否调用节点处理。

Godot 还支持 process thread group。`scene/main/scene_tree.cpp` 中可看到，处理组可以在主线程或 WorkerThreadPool 的子线程中运行。阅读多线程处理时，要特别关注线程安全守卫和 deferred 操作。

### 5.5 退出树与删除

`Node::_propagate_exit_tree()` 会按子节点反序退出，调用 `_exit_tree`，发出 `tree_exiting`，发送 `NOTIFICATION_EXIT_TREE`，并清理组、树指针和缓存。

`Node::queue_free()` 不会立即删除节点，而是把节点加入 SceneTree 的删除队列。SceneTree 在安全位置刷新删除队列。这样可以避免在遍历节点列表、处理信号或执行回调时立即破坏树结构。

### 5.6 SceneTree 的每帧工作

`scene/main/scene_tree.cpp` 中的主流程包括：

- 初始化时调用 `root->_set_tree(this)`。
- 物理帧中发出 `physics_frame`，处理 physics process、物理查询、导航、timer/tween、删除队列等。
- 普通帧中发出 `process_frame`，处理 process、multiplayer poll、消息队列、场景切换、timer/tween、删除队列、可访问性变更等。
- 组调用时会复制节点列表并检查节点是否仍在组内，避免遍历期间被移除造成错误。

### 本章小结

SceneTree 是运行时调度器，Node 是场景层对象。进入树、ready、process、physics process、退出树、删除队列共同定义了节点生命周期。大多数“节点为什么没有执行”或“为什么必须 deferred”的问题，都能在这一章的源码中找到答案。

### 习题

1. 在 `Node::_propagate_ready()` 中确认 ready 的递归顺序。
2. 在 `SceneTree::process()` 中标记消息队列刷新和删除队列刷新的位置。
3. 断点观察一个节点调用 `queue_free()` 后，真正析构发生在哪个阶段。

## 第 6 章 Server 架构

### 6.1 为什么有 Server

Godot 的节点 API 面向游戏逻辑，Server API 面向底层执行。节点不直接持有渲染后端对象或物理世界实现，而是通过 `RID` 和 Server 单例提交请求。

这种设计带来几个结果：

1. 场景层可以保持相对稳定，底层后端可以替换。
2. 渲染和物理资源可以用 `RID` 跨线程或跨层管理。
3. 脚本也可以直接访问部分 Server，实现低层控制。
4. 编辑器和运行时共用大量底层服务。

### 6.2 Server 注册

`servers/register_server_types.cpp` 注册了大量 Server 相关类和单例，包括：

- `TextServerManager`
- `AccessibilityServer`
- `DisplayServer`
- `RenderingServer`
- `AudioServer`
- `CameraServer`
- `RenderingDevice`
- 2D/3D 物理 Server 和 Manager
- 导航 Server
- XR Server

`Main::setup2()` 会在合适阶段创建这些服务，并把重要单例加入 `Engine`。

### 6.3 DisplayServer

`DisplayServer` 抽象窗口、显示器、输入法、鼠标、屏幕、子窗口等平台显示能力。`servers/display/display_server.cpp` 中的 `DisplayServer::register_create_function()` 用于注册显示后端，`DisplayServer::create()` 根据索引和渲染驱动创建具体实现。源码中默认还包含 headless 显示服务。

平台差异主要在 DisplayServer 和 OS 层解决，场景层通常不直接依赖平台窗口 API。

### 6.4 RenderingServer

`servers/rendering/rendering_server.h` 定义了渲染服务抽象，提供纹理、shader、材质、mesh、canvas、viewport、instance 等 RID API。

默认实现 `RenderingServerDefault` 位于 `servers/rendering/rendering_server_default.cpp`。它包含：

- `CommandQueueMT`
- 可选渲染线程
- `RendererCanvasCull`
- `RendererViewport`
- `RendererSceneCull`
- `RendererCompositor`
- 全局渲染存储指针 `RenderingServerGlobals`

`RenderingServerDefault::sync()` 会同步或刷新命令队列，`draw()` 会触发 `_draw()`。`_draw()` 中可以看到一帧渲染的大致步骤：begin frame、XR pre-render、scene/canvas 更新、particles、probe、viewport draw、canvas render update、end frame、XR end、可见性通知和 profiling。

### 6.5 PhysicsServer2D/3D

`servers/physics_2d/physics_server_2d.h` 和 `servers/physics_3d/physics_server_3d.h` 定义物理抽象接口，使用 RID 管理 shape、space、area、body、joint 和查询对象。

具体实现由模块注册。例如：

- `modules/godot_physics_2d/register_types.cpp` 注册 `GodotPhysics2D` 并设为默认 2D 物理。
- `modules/godot_physics_3d/register_types.cpp` 注册 `GodotPhysics3D` 并设为默认 3D 物理。
- `modules/jolt_physics/register_types.cpp` 注册 `Jolt Physics`。

`main/main.cpp` 的 `initialize_physics()` 会根据项目设置 `physics/2d/physics_engine` 和 `physics/3d/physics_engine` 创建指定 Server；如果指定失败则使用默认；如果没有默认则使用 Dummy Server。

### 6.6 AudioServer

`servers/audio/audio_server.h` 和 `servers/audio/audio_server.cpp` 管理音频总线、效果、混音数据和 playback。底层音频后端通过 `AudioDriver` 抽象。`Main::iteration()` 每帧会调用 `AudioServer::update()`，而驱动侧会请求音频服务进行混音处理。

### 本章小结

Server 是 Godot 运行时的执行层。节点描述“要做什么”，Server 管理“如何执行”。理解 RID 和 Server 单例，是继续阅读渲染、物理、音频和显示后端的前提。

### 习题

1. 在 `RenderingServer` 中找到 `canvas_item_create()` 和 `instance_create()`，说明它们分别服务 2D 和 3D 哪些节点。
2. 在 `initialize_physics()` 中追踪项目设置如何影响物理 Server 创建。
3. 在 `DisplayServer::register_create_function()` 中观察 headless 后端与平台后端的注册关系。

## 第 7 章 2D/3D 节点如何进入渲染

### 7.1 CanvasItem：2D 绘制根类

`scene/main/canvas_item.cpp` 是 2D 渲染链路的入口之一。`CanvasItem` 构造时调用：

```text
RenderingServer::get_singleton()->canvas_item_create()
```

析构时释放对应 RID。进入树时，`CanvasItem::_enter_canvas()` 会根据父级 `CanvasItem`、`CanvasLayer` 或 `Viewport` 找到 canvas，并调用 `canvas_item_set_parent()`。可见性变化通过 `canvas_item_set_visible()` 下发到 `RenderingServer`。

`CanvasItem::queue_redraw()` 会通过 deferred callback 安排 `_redraw_callback()`。真正绘制时 `_redraw_callback()` 清理旧绘制命令，发送 `NOTIFICATION_DRAW`，发出 `draw` 信号，并调用脚本虚方法 `_draw`。这解释了为什么 Godot 要求绘制命令只在 `_draw()`、draw 信号或 `NOTIFICATION_DRAW` 中调用。

### 7.2 Node2D：变换下发

`scene/2d/node_2d.cpp` 中的 `Node2D::_update_transform()` 会计算本地 `Transform2D`，并调用：

```text
RenderingServer::canvas_item_set_transform(get_canvas_item(), transform)
```

随后调用 `_notify_transform()`，把变换变化通知给依赖它的系统。也就是说，`Node2D` 自己不绘制，它继承 `CanvasItem` 的 RID，并把变换同步给渲染 Server。

典型 2D 链路为：

```text
Node2D::set_position()
  -> Node2D::_update_transform()
  -> RenderingServer::canvas_item_set_transform()
  -> RendererCanvasCull
  -> viewport/canvas 渲染
```

具体绘制节点如 `Sprite2D`、`Polygon2D`、GUI `Control` 会在 `NOTIFICATION_DRAW` 中向同一个 canvas item RID 添加绘制命令。

### 7.3 Node3D：空间变换与世界

`scene/3d/node_3d.cpp` 管理 3D 节点的本地和全局变换。它有 dirty bit 机制，避免每次访问都重新计算全局变换。变换改变时，`_propagate_transform_changed()` 会向子树传播 dirty 状态，并在需要时把节点加入 `SceneTree::xform_change_list`，之后发送 `NOTIFICATION_TRANSFORM_CHANGED`。

进入树时，`Node3D` 会处理父级 `Node3D`、world、viewport、可见性和物理插值相关状态。它本身还不是可渲染对象，可渲染能力由 `VisualInstance3D` 及其派生类提供。

### 7.4 VisualInstance3D：3D 可见实例

`scene/3d/visual_instance_3d.cpp` 中的 `VisualInstance3D` 构造时调用：

```text
RenderingServer::instance_create()
RenderingServer::instance_attach_object_instance_id()
set_notify_transform(true)
```

进入 world 时，它把 instance 绑定到当前 `World3D` 的 scenario：

```text
RenderingServer::instance_set_scenario(instance, get_world_3d()->get_scenario())
```

变换变化时，如果可见且未使用 identity transform，它调用：

```text
RenderingServer::instance_set_transform(instance, get_global_transform())
```

几何派生类再通过 `instance_set_base()`、材质、可见范围等接口把 mesh 或其他几何资源绑定到这个 instance。典型 3D 链路为：

```text
MeshInstance3D
  -> VisualInstance3D::instance
  -> RenderingServer::instance_set_base()
  -> RenderingServer::instance_set_transform()
  -> RendererSceneCull
  -> viewport 渲染
```

### 本章小结

2D 节点通过 `CanvasItem` 持有 canvas item RID，3D 可见节点通过 `VisualInstance3D` 持有 instance RID。节点层负责变换、可见性和生命周期，渲染 Server 负责保存和执行底层渲染命令。

### 习题

1. 在 `CanvasItem::_redraw_callback()` 中标出 `_draw()` 被调用的位置。
2. 修改一个 `Node2D` 的 position，断点观察 `canvas_item_set_transform()` 是否被调用。
3. 在 `VisualInstance3D::_notification()` 中说明 `NOTIFICATION_ENTER_WORLD` 和 `NOTIFICATION_TRANSFORM_CHANGED` 分别下发什么渲染状态。

## 第 8 章 脚本、模块与 GDExtension

### 8.1 ScriptServer

`core/object/script_language.h` 和 `core/object/script_language.cpp` 定义脚本语言抽象。`ScriptServer::register_language()` 注册脚本语言，要求扩展名、语言名和类型不能重复。如果脚本系统已经初始化，新注册语言会立即调用 `init()`。

`ScriptServer::init_languages()` 会加载全局脚本类并初始化所有注册语言。`ScriptServer::finish_languages()` 会结束脚本语言并移除相关日志器。`Main::iteration()` 中还会调用脚本语言的逐帧入口。

脚本语言不是附着在编辑器上的功能，而是引擎运行时的一等扩展点。

### 8.2 模块初始化层级

`modules/register_module_types.h` 定义了模块初始化层级：

```text
MODULE_INITIALIZATION_LEVEL_CORE
MODULE_INITIALIZATION_LEVEL_SERVERS
MODULE_INITIALIZATION_LEVEL_SCENE
MODULE_INITIALIZATION_LEVEL_EDITOR
```

生成文件 `modules/register_module_types.gen.cpp` 中的 `initialize_modules(p_level)` 会按层级调用启用模块的初始化函数。`Main` 在不同阶段调用不同层级：

- core 层：基础类型和最低层能力。
- servers 层：Server 和运行时服务。
- scene 层：节点、资源和场景相关类型。
- editor 层：编辑器专用类型和插件。

模块必须把注册代码放在正确层级。例如 GDScript 的 `modules/gdscript/register_types.cpp` 会在 SERVERS 层注册语言和资源加载器，在 EDITOR 层注册语法高亮、语言服务器和编辑器插件。

### 8.3 GDExtension

`core/extension/gdextension.cpp` 和 `core/extension/gdextension_manager.cpp` 管理 GDExtension。`GDExtension` 是一个资源对象，包装原生动态库加载、接口函数、类注册和初始化回调。

`GDExtensionManager::initialize_extensions()` 会按初始化层级初始化扩展。如果扩展在引擎已经初始化后被加载，管理器会从最低需要层级补跑到当前层级。反初始化时按相反层级清理。

GDExtension 的层级与模块层级对应，这让内置模块和外部扩展可以遵循同一套生命周期模型。

### 本章小结

脚本语言、内置模块和 GDExtension 都依赖 Godot 的注册系统。区别在于：脚本语言通过 `ScriptServer` 接入，内置模块通过构建生成的 `initialize_modules()` 接入，GDExtension 通过动态库和 `GDExtensionManager` 接入。

### 习题

1. 在 `modules/gdscript/register_types.cpp` 中找出 GDScript 语言对象何时创建、何时注册到 `ScriptServer`。
2. 在 `modules/register_module_types.gen.cpp` 中观察模块初始化顺序。
3. 阅读 `GDExtensionManager::initialize_extensions()`，说明运行时加载扩展时为什么需要考虑当前初始化层级。

## 第 9 章 编辑器也是场景树程序

### 9.1 编辑器类型注册

`editor/register_editor_types.cpp` 注册编辑器相关类型，包括 `EditorPlugin`、`EditorSettings`、`EditorFileSystem`、`ScriptEditor`、`EditorInterface`、导入导出类、Inspector 类和大量编辑器插件。

当 `TOOLS_ENABLED` 启用时，`Main::setup2()` 会初始化 editor 层模块和扩展，并注册编辑器类型。非编辑器模板不会包含这些能力。

### 9.2 EditorNode

`editor/editor_node.cpp` 和 `editor/editor_node.h` 中的 `EditorNode` 是编辑器主节点。`Main::start()` 在编辑器模式下创建 `EditorNode` 并加入 `SceneTree` root。此后编辑器 UI、场景编辑、Inspector、资源文件系统、导入导出和插件系统都作为运行在 SceneTree 中的对象协作。

这意味着 Godot 编辑器不是独立于引擎运行时的另一个程序，而是大量复用 Object、SceneTree、Resource、Server 和 UI 节点的 Godot 应用。

### 9.3 ProjectManager

项目管理器入口在 `editor/project_manager/project_manager.cpp`。`Main::start()` 在项目管理器模式下创建 `ProjectManager`。它同样是 `Control` 派生的界面节点，运行在 SceneTree 中。

### 本章小结

Godot 编辑器本身就是使用 Godot 运行时构建的大型工具应用。理解运行时对象、节点生命周期和 Server 后，再阅读编辑器源码会清晰很多。

### 习题

1. 在 `Main::start()` 中找到创建 `EditorNode` 和 `ProjectManager` 的分支。
2. 在 `editor/register_editor_types.cpp` 中挑选一个编辑器插件，追踪它如何注册。
3. 阅读 `EditorNode` 的初始化代码，列出它依赖的三个核心子系统。

## 第 10 章 推荐阅读路线

### 10.1 第一阶段：建立启动骨架

阅读文件：

- `platform/windows/godot_windows.cpp` 或 `platform/linuxbsd/godot_linuxbsd.cpp`
- `platform/windows/os_windows.cpp` 或 `platform/linuxbsd/os_linuxbsd.cpp`
- `main/main.cpp`
- `core/register_core_types.cpp`

目标：

- 画出 `main -> OS -> Main -> SceneTree` 的流程图。
- 能解释 `setup`、`setup2`、`start`、`iteration`、`cleanup` 的边界。
- 能找到某个单例在哪里创建、在哪里销毁。

### 10.2 第二阶段：掌握 Object 与 ClassDB

阅读文件：

- `core/object/object.h`
- `core/object/object.cpp`
- `core/object/class_db.h`
- `core/object/class_db.cpp`
- `core/variant/variant.h`
- `scene/2d/node_2d.h`
- `scene/2d/node_2d.cpp`
- `scene/register_scene_types.cpp`

目标：

- 理解类注册和动态实例化。
- 理解属性、方法、信号为什么能被脚本和编辑器访问。
- 能追踪一个 `ClassDB::instantiate("Node2D")` 的路径。
- 能从 `GDCLASS(Node2D, CanvasItem)`、`GDREGISTER_CLASS(Node2D)`、`Node2D::_bind_methods()` 解释一个类如何进入脚本、Inspector 和场景序列化。

### 10.3 第三阶段：资源与场景

阅读文件：

- `core/io/resource.h`
- `core/io/resource_loader.cpp`
- `scene/resources/packed_scene.cpp`
- `scene/main/node.cpp`
- `scene/main/scene_tree.cpp`

目标：

- 跟踪主场景加载和实例化。
- 理解 owner 与 parent 的区别。
- 理解 ready 顺序和删除队列。

### 10.4 第四阶段：Server 与具体子系统

阅读文件：

- `servers/register_server_types.cpp`
- `servers/rendering/rendering_server.h`
- `servers/rendering/rendering_server_default.cpp`
- `scene/main/canvas_item.cpp`
- `scene/2d/node_2d.cpp`
- `scene/3d/visual_instance_3d.cpp`
- `servers/physics_2d/physics_server_2d.cpp`
- `servers/physics_3d/physics_server_3d.cpp`

目标：

- 理解 RID。
- 理解节点状态如何下发到 Server。
- 能选择一个方向继续深入，例如 Renderer、Physics、GUI、GDScript 或 Editor。

### 10.5 一条完整跟踪练习

建议用调试器跟踪以下路径：

```text
启动可执行文件
  -> platform main
  -> Main::setup()
  -> Main::setup2()
  -> Main::start()
  -> ResourceLoader::load(main scene)
  -> PackedScene::instantiate()
  -> ClassDB::instantiate(Node type)
  -> Object::set(saved property)
  -> ClassDB::set_property()
  -> Node::add_child()
  -> Node::_propagate_enter_tree()
  -> Node::_propagate_ready()
  -> Main::iteration()
  -> SceneTree::physics_process()
  -> SceneTree::process()
  -> RenderingServer::sync()
  -> RenderingServer::draw()
```

这条路径覆盖了 Godot 源码学习的最小闭环。

## 附录 A 常见概念速查

| 概念 | 含义 |
| --- | --- |
| `Object` | Godot 运行时对象根类，提供属性、方法、信号、脚本实例和 ObjectID。 |
| `RefCounted` | 引用计数对象基类，常用于资源。 |
| `Resource` | 可加载、保存、缓存和复用的数据对象。 |
| `Node` | 场景树节点基类。 |
| `SceneTree` | 默认 MainLoop，负责节点生命周期和每帧调度。 |
| `GDCLASS` | C++ 类接入 Godot 对象模型的宏，生成静态类名、父类关系、绑定入口和属性转发等样板代码。 |
| `GDREGISTER_CLASS` | 把 C++ 类注册进 `ClassDB`，并设置创建函数和暴露状态。 |
| `ClassDB` | 类注册、反射、方法绑定和动态实例化中心。 |
| `_bind_methods()` | 类向 `ClassDB` 登记方法、属性、信号、常量和虚方法的入口。 |
| `Variant` | 动态值容器，连接 C++、脚本、属性、信号和序列化。 |
| `RID` | Resource ID，Server 层资源句柄。 |
| `Server` | 底层执行服务，如 RenderingServer、PhysicsServer、AudioServer。 |
| `PackedScene` | 序列化场景资源，可实例化为节点树。 |
| `GDExtension` | 动态库扩展机制。 |
| `Module` | 编译期集成的可选功能模块。 |
| `TOOLS_ENABLED` | 编辑器构建相关宏。 |
| `DEBUG_ENABLED` | 调试构建相关宏。 |

## 附录 B 建议断点

初学源码时，建议设置这些断点：

- `platform/windows/godot_windows.cpp` 或 `platform/linuxbsd/godot_linuxbsd.cpp` 的平台入口。
- `main/main.cpp`：`Main::setup()`、`Main::setup2()`、`Main::start()`、`Main::iteration()`、`Main::cleanup()`。
- `core/register_core_types.cpp`：`register_core_types()`。
- `scene/register_scene_types.cpp`：`GDREGISTER_CLASS(Node2D)` 附近。
- `scene/2d/node_2d.cpp`：`Node2D::_bind_methods()`、`Node2D::set_position()`、`Node2D::_update_transform()`。
- `core/object/class_db.cpp`：`ClassDB::_instantiate_internal()`、`ClassDB::set_property()`、`ClassDB::get_property()`。
- `core/object/object.cpp`：`Object::_construct_object()`、`Object::~Object()`、`Object::set()`、`Object::callp()`。
- `core/io/resource_loader.cpp`：`ResourceLoader::load()`。
- `scene/resources/packed_scene.cpp`：`PackedScene::instantiate()`。
- `scene/main/node.cpp`：`Node::_propagate_enter_tree()`、`Node::_propagate_ready()`、`Node::_propagate_exit_tree()`、`Node::queue_free()`。
- `scene/main/scene_tree.cpp`：`SceneTree::physics_process()`、`SceneTree::process()`。
- `scene/main/canvas_item.cpp`：`CanvasItem::_redraw_callback()`。
- `scene/3d/visual_instance_3d.cpp`：`VisualInstance3D::_notification()`。
- `servers/rendering/rendering_server_default.cpp`：`RenderingServerDefault::sync()`、`RenderingServerDefault::draw()`。

## 附录 C 阅读时的判断方法

阅读 Godot 源码时，遇到一个类或函数可以按下面顺序判断：

1. 它是不是 `Object` 派生类？如果是，先找 `GDCLASS(当前类, 父类)`，再找 `GDREGISTER_*` 注册位置，最后看 `_bind_methods()`。
2. 它是不是 `Resource`？如果是，找加载器、保存器和缓存行为。
3. 它是不是 `Node`？如果是，找生命周期通知和 SceneTree 交互。
4. 它是否持有 `RID`？如果是，继续追踪对应 Server。
5. 它是否只在 `TOOLS_ENABLED` 下存在？如果是，它属于编辑器或工具链。
6. 它是否在模块中？如果是，找 `register_types.cpp` 和初始化层级。
7. 它是否由构建生成？如果是，回到 `SConstruct`、`SCsub` 或生成脚本。

按照这个顺序阅读，通常可以较快定位一段源码在 Godot 架构中的位置。
