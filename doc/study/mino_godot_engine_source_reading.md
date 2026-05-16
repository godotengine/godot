# Godot Engine 4.7 源码导读

> 版本：Godot 4.7.0-beta | 源码位置：godot/
> 面向第一次系统阅读 Godot 源码的学习者

---

## 目录

1. [代码库概览](#1-代码库概览)
2. [构建系统与入口](#2-构建系统与入口)
3. [推荐阅读路线](#3-推荐阅读路线)
4. [核心子系统详解](#4-核心子系统详解)
   - [4.1 类型系统：Variant](#41-类型系统variant)
   - [4.2 对象系统：Object & ClassDB](#42-对象系统object--classdb)
   - [4.3 引用计数：RefCounted & Ref\<T\>](#43-引用计数refcounted--reft)
   - [4.4 信号系统：Signal & Callable](#44-信号系统signal--callable)
   - [4.5 StringName 字符串驻留](#45-stringname-字符串驻留)
   - [4.6 节点系统：Node & SceneTree](#46-节点系统node--scenetree)
   - [4.7 图形界面：Control 体系](#47-图形界面control-体系)
   - [4.8 渲染服务：RenderingServer](#48-渲染服务renderingserver)
   - [4.9 物理系统：PhysicsServer2D/3D](#49-物理系统physicsserver2d3d)
   - [4.10 动画系统：AnimationMixer & Tween](#410-动画系统animationmixer--tween)
   - [4.11 资源系统：Resource & ResourceLoader](#411-资源系统resource--resourceloader)
   - [4.12 输入系统：Input & InputMap](#412-输入系统input--inputmap)
   - [4.13 导航系统：NavigationServer2D/3D](#413-导航系统navigationserver2d3d)
   - [4.14 音频系统：AudioServer](#414-音频系统audioserver)
   - [4.15 GDScript 语言](#415-gdscript-语言)
   - [4.16 C# / Mono 模块](#416-c--mono-模块)
   - [4.17 多线程与并发](#417-多线程与并发)
   - [4.18 编辑器架构](#418-编辑器架构)
   - [4.19 资源导入管道](#419-资源导入管道)
   - [4.20 平台抽象层](#420-平台抽象层)
   - [4.21 网络/多人系统](#421-网络多人系统)
   - [4.22 文件 I/O 系统](#422-文件-io-系统)
5. [关键宏与模式总结](#5-关键宏与模式总结)
6. [核心类继承关系](#6-核心类继承关系)
7. [附录：小技巧](#7-附录小技巧)

---

## 1. 代码库概览

### 顶层目录

```
godot/
  SConstruct                # SCons 构建文件（Python）
  version.py                # 版本信息：4.7.0-beta
  methods.py                # 共享构建方法
  glsl_builders.py          # GLSL 着色器编译为 C++ 头文件

  core/                     # 核心引擎：类型系统、对象、数学、IO、模板
  scene/                    # 场景树、UI控件、2D/3D 节点
  servers/                  # 抽象服务接口（渲染、物理、音频、导航等）
  editor/                   # 编辑器专用代码
  main/                     # 入口点、启动流程、主循环
  drivers/                  # 平台具体驱动（Vulkan、GLES3、D3D12等）
  platform/                 # 平台入口（windows、linuxbsd、android 等）
  modules/                  # 可选模块（gdscript、mono、physics 等）
  thirdparty/               # 第三方库（69个）
  doc/                      # 文档
  tests/                    # 单元测试（doctest 框架）
  misc/                     # 杂项工具脚本
  bin/                      # 构建输出
```

### 分层架构概览

```
┌─────────────────────────────────────────────────────────────┐
│  platform/ (OS_Windows, OS_LinuxBSD...)                     │
│  平台入口、WinMain、消息循环                                  │
├─────────────────────────────────────────────────────────────┤
│  main/ (Main::setup -> setup2 -> start -> iteration)        │
│  引擎初始化与主循环                                           │
├─────────────────────────────────────────────────────────────┤
│  editor/ (EditorNode, EditorPlugin...)                      │
│  编辑器逻辑（仅 TOOLS_ENABLED 构建）                          │
├─────────────────────────────────────────────────────────────┤
│  scene/ (Node, SceneTree, Control, Node2D, Node3D...)       │
│  节点、UI、动画、音频播放器                                    │
├─────────────────────────────────────────────────────────────┤
│  servers/ (RenderingServer, PhysicsServer, AudioServer...)  │
│  抽象服务接口（纯虚类）                                        │
├─────────────────────────────────────────────────────────────┤
│  core/ (Object, Variant, ClassDB, FileAccess...)            │
│  基础类型系统、对象模型、IO、线程、容器                          │
├─────────────────────────────────────────────────────────────┤
│  drivers/ + modules/ + thirdparty/                          │
│  具体实现（Vulkan/GLES3 渲染器、GodotPhysics、GDScript VM...）  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 构建系统与入口

### 2.1 SCons 构建 (`SConstruct`)

Godot 使用 **SCons** (>=4.0) + **Python >=3.9**，标准为 **C++17**。关键特征：

```python
# 构建目标
target = "editor"          # 编辑器构建（含 TOOLS_ENABLED）
target = "template_release"  # 发布模板
target = "template_debug"    # 调试模板

# 平台
platform = "windows"       # 也支持 linuxbsd, macos, android, ios, web, visionos

# 关键选项
precision = "single"       # 浮点精度（也支持 double）
dev_build = "yes"          # 开发者构建（DEV_ENABLED）
optimize = "speed"         # 优化级别
```

版本定义在 `version.py`：
```python
major = 4; minor = 7; patch = 0; status = "beta"
```

**构建入口阅读路线：**
1. `SConstruct` → 加载 `methods.py`、检测平台 → 设置环境
2. 扫描 `platform/*/detect.py` → 编译器配置（MSVC/MinGW）
3. 扫描 `modules/*/config.py` → 启用/禁用模块
4. 调用 `env.SConscript("core/SCsub")`、`scene/SCsub`、`servers/SCsub`...

### 2.2 启动流程 (`main/main.cpp`)

引擎初始化分 **三个阶段**，由每个平台的 `main()` 调用：

```
平台 main()
  └─> Main::setup(execpath, argc, argv, p_second_phase=true)
       └─> Main::setup2()
            └─> Main::start()
                 └─> Main::iteration() 循环
```

#### 第一阶段：`Main::setup()` (~第1027行)

核心调用顺序：
```cpp
Thread::make_main_thread();                     // 设置主线程ID
OS::get_singleton()->initialize();              // 平台初始化

register_core_types();                          // 注册基本类型
register_core_driver_types();

// 创建核心单例
input_map = memnew(InputMap);
globals = memnew(ProjectSettings);
translation_server = memnew(TranslationServer);
performance = memnew(Performance);

// 解析命令行参数
// 初始化 PackedData (PCK 文件系统), ZipArchive
```

#### 第二阶段：`Main::setup2()` (~第3008行)

```cpp
// 创建服务管理器
TextServerManager, PhysicsServer2DManager, PhysicsServer3DManager
NavigationServer2DManager, NavigationServer3DManager

register_server_types();            // 注册服务类型
initialize_modules(MODULE_INITIALIZATION_LEVEL_SERVERS);

DisplayServer::create();            // 创建显示服务（窗口）
RenderingServer::create();          // 创建渲染服务
AudioServer::init();

register_scene_types();             // 注册场景类型
register_driver_types();

// 编辑器额外注册
register_editor_types();            // (仅 TOOLS_ENABLED)
```

#### 第三阶段：`Main::start()` (~第3988行)

```cpp
// 创建 SceneTree（主循环）
SceneTree *scene_tree = memnew(SceneTree);
scene_tree->initialize();

// 游戏模式：加载 autoloads + 主场景
// 编辑器模式：创建 EditorNode
// 项目管理器：创建 ProjectManager
```

#### 主循环：`Main::iteration()` (~第4896行)

```cpp
// 1. 计算帧时间
main_timer_sync.advance(delta);

// 2. 固定物理步长循环
main_loop->iteration_prepare()
main_loop->physics_process(step)    // SceneTree::physics_process
    ├── NavigationServer physics process
    ├── PhysicsServer2D::step()
    └── PhysicsServer3D::step()
main_loop->iteration_end()

// 3. 可变渲染帧处理
main_loop->process(step)            // SceneTree::process

// 4. 渲染
RenderingServer::sync()
RenderingServer::draw()             // 实际渲染调用

// 5. 音频
AudioServer::update()
```

---

## 3. 推荐阅读路线

对于第一次系统阅读 Godot 源码的学习者，建议按以下顺序逐步深入：

### 第一遍：理解核心机制（阅读时间 ~2天）

| 顺序 | 文件 | 掌握要点 |
|------|------|---------|
| 1 | `core/typedefs.h` | 基本类型别名、`_FORCE_INLINE_`、`CLAMP`/`MIN`/`MAX` |
| 2 | `core/variant/variant.h` | Variant 类型系统、40种数据类型 |
| 3 | `core/string/string_name.h` | 字符串驻留机制 |
| 4 | `core/object/object.h` | Object 基类、信号系统、cast_to |
| 5 | `core/object/class_db.h` | ClassDB 类注册 |
| 6 | `core/object/ref_counted.h` | 引用计数 + Ref\<T\> 智能指针 |
| 7 | `core/templates/` | Vector, HashMap, List, RID 等容器 |
| 8 | `core/io/resource.h` | Resource 基类 |

### 第二遍：理解运行框架（~2天）

| 顺序 | 文件 | 掌握要点 |
|------|------|---------|
| 1 | `main/main.cpp` | 引擎启动三阶段 |
| 2 | `scene/main/node.h` | Node 体系 |
| 3 | `scene/main/scene_tree.h` | SceneTree 管理 |
| 4 | `scene/main/viewport.h` | 视口系统 |
| 5 | `scene/main/canvas_item.h` | 2D 绘制基类 |
| 6 | `servers/rendering/rendering_server.h` | 渲染服务 API |
| 7 | `servers/display/display_server.h` | 显示服务 |
| 8 | `core/os/main_loop.h` | 主循环抽象 |

### 第三遍：理解服务层（~2天）

| 顺序 | 文件 | 掌握要点 |
|------|------|---------|
| 1 | `servers/physics_2d/physics_server_2d.h` | 物理服务 API |
| 2 | `servers/physics_3d/physics_server_3d.h` | 物理服务 API |
| 3 | `servers/audio/audio_server.h` | 音频服务 |
| 4 | `servers/text/text_server.h` | 文本/字体服务 |
| 5 | `scene/gui/control.h` | UI Control 基类 |
| 6 | `scene/animation/animation_mixer.h` | 动画混音器 |
| 7 | `modules/gdscript/gdscript.h` | GDScript 语言 |
| 8 | `editor/editor_node.h` | 编辑器节点 |

### 第四遍：深入具体实现（按需）

| 领域 | 关注文件 |
|------|---------|
| Vulkan 渲染 | `drivers/vulkan/rendering_device_driver_vulkan.h` |
| GLES3 渲染 | `drivers/gles3/rasterizer_gles3.h` |
| Godot 物理 2D | `modules/godot_physics_2d/godot_physics_server_2d.h` |
| Godot 物理 3D | `modules/godot_physics_3d/godot_physics_server_3d.h` |
| GDScript VM | `modules/gdscript/gdscript_vm.cpp` |
| GDScript 解析器 | `modules/gdscript/gdscript_parser.h` |
| 资源格式 | `core/io/resource_format_binary.h` |
| Windows 平台 | `platform/windows/os_windows.h` |
| 导航系统 | `modules/navigation_2d/nav_map_2d.h` |
| 多人网络 | `modules/multiplayer/scene_multiplayer.h` |

---

## 4. 核心子系统详解

### 4.1 类型系统：Variant

**文件：** `core/variant/variant.h` (984行), `core/variant/variant.cpp`

Variant 是 Godot 的** tagged union**（带标签的联合体），可以持有任意 Godot 内建类型。它是一切动态类型交互的基础——GDScript 变量、信号参数、属性系统都依赖它。

#### 支持的类型

```cpp
enum Type {
    NIL,
    BOOL, INT, FLOAT, STRING,               // 原子类型
    VECTOR2, VECTOR2I, RECT2, VECTOR3, ...   // 数学类型 (12个)
    COLOR, STRING_NAME, NODE_PATH, RID,      // 杂项类型
    OBJECT, CALLABLE, SIGNAL,
    DICTIONARY, ARRAY,
    PACKED_BYTE_ARRAY, PACKED_INT32_ARRAY,   // 打包数组 (10个)
    ... 
    VARIANT_MAX  // = 40
};
```

#### 内部存储

Variant 大小为 24 字节（单精度）或 40 字节（双精度）。使用 union 实现：

```cpp
struct ObjData { ObjectID id; Object *obj; };  // 对象引用
struct PackedArrayRefBase { SafeRefCount refcount; }; // 打包数组共享数据

// 核心存储：union 包含 bool, int64_t, double, 指针, 以及内联缓冲区
// 复杂类型（AABB, Transform3D, Projection 等）在堆上分配
```

#### 操作分发表

Variant 有四个操作分发表，实现了动态类型操作：

- `variant_op.cpp`：运算符分发表（`OP_EQUAL`, `OP_ADD`, `OP_NEGATE` 等）
- `variant_call.cpp`：内建方法调用表（如 `String::length()`）
- `variant_construct.cpp`：构造函数表
- `variant_utility.cpp`：工具函数表

**阅读要点：**
- 关注 `get_type()` 和类型转换方法
- 理解 `operator==` 和 `hash()` 如何工作
- 注意 `OBJECT` 类型的 `ObjData` 包含 `ObjectID` + 裸指针

---

### 4.2 对象系统：Object & ClassDB

**文件：** `core/object/object.h` (1168行), `core/object/class_db.h` (589行)

Object 是所有 Godot 引擎对象的**终极基类**。它实现了属性系统、信号系统、脚本绑定、RTTI（运行时类型识别）。

#### Object 核心结构

```cpp
class Object {
    // 信号系统
    SignalData { Slot { Connection conn; int reference_count; } }
    HashMap<StringName, SignalData> signal_map;
    List<Connection> connections;

    // 类型信息
    const GDType *_gdtype_ptr;
    uint32_t _ancestry : 15;  // 快速祖先检测位域

    // 脚本绑定
    ScriptInstance *script_instance;
    ObjectGDExtension *_extension;

    // GDExtension 实例绑定
    InstanceBinding *_instance_bindings;

    // 元数据
    HashMap<StringName, Variant> metadata;
};
```

#### 祖先进程位域 (AncestralClass)

```cpp
enum class AncestralClass : unsigned int {
    REF_COUNTED = 1 << 0,
    NODE = 1 << 1,
    RESOURCE = 1 << 2,
    SCRIPT = 1 << 3,
    CANVAS_ITEM = 1 << 4,
    CONTROL = 1 << 5,
    NODE_2D = 1 << 6,
    NODE_3D = 1 << 9,
    MESH_INSTANCE_3D = 1 << 14,
    // ...
};
```

这是一个精巧的优化：`cast_to<T>()` 先检查位域，如果 `T` 不是祖先就直接返回 `nullptr`，无需遍历继承链。

#### GDCLASS 宏

```cpp
#define GDCLASS(m_class, m_inherits) \
    GDSOFTCLASS(m_class, m_inherits)  // 提供 cast_to 功能
    // 生成:
    //   get_class_static() -> StringName
    //   get_gdtype_static() -> GDType&
    //   initialize_class() -> 注册到 ClassDB
    //   _bind_methods() 自动调用
```

**阅读要点：**
- `cast_to<T>()` 取代 `dynamic_cast`，更快速
- `_set`/`_get`/`_notification`/`_bind_methods` 是子类可以重写的关键虚方法
- `initialize_class()` 是线程安全的双检锁模式

#### ClassDB

ClassDB 维护了全局的类注册表：

```cpp
struct ClassInfo {
    MethodBind *creation_func;         // 工厂函数
    ClassInfo *inherits_ptr;           // 父类
    HashMap<StringName, MethodBind*> method_map;
    List<PropertyInfo> property_list;
    AHashMap<StringName, PropertySetGet> property_setget;
    // ...
};

static HashMap<StringName, ClassInfo> classes;
```

**注册流程：**
1. 子类调用 `GDCLASS(MyClass, Parent)`
2. `initialize_class()` 调用父类的 `initialize_class()`
3. 调用 `_bind_methods()` 使用 `BIND_METHOD`, `ADD_PROPERTY`, `ADD_SIGNAL` 等宏
4. ClassDB 存储方法/属性/信号元信息

**阅读要点：**
- `ClassDB::classes` 是 `HashMap<StringName, ClassInfo>`
- `APIType` 区分 `API_CORE`, `API_EDITOR`, `API_EXTENSION` 等
- `is_class_enabled<T>` 模板结构体配合 `disabled_classes.gen.h` 支持按构建配置禁用类

---

### 4.3 引用计数：RefCounted & Ref\<T\>

**文件：** `core/object/ref_counted.h` (262行)

#### RefCounted

```cpp
class RefCounted : public Object {
    GDCLASS(RefCounted, Object);
    SafeRefCount refcount;       // 引用计数
    SafeRefCount refcount_init;  // 初始化计数（检测是否被Ref管理）
    SafeNumeric<uint32_t> dereference_count;

    bool init_ref();      // 初始引用（由 Ref 首次赋值时调用）
    bool reference();     // 增加引用计数
    bool unreference();   // 减少引用计数，返回 true 表示降为 0
};
```

#### Ref\<T\> 智能指针

```cpp
template <typename T>
class Ref {
    T *reference = nullptr;

    // 核心逻辑：ref_pointer
    //   ref() -> 调用 reference()
    //   析构 -> 调用 unreference()，如果返回 true 则 memdelete
    //   拷贝 -> 先 unref 旧的，再 ref 新的
    //   move -> 直接转移指针
    
    void unref();          // 安全析构
    void instantiate();    // memnew + 自动 ref
    bool is_valid();       // 检查是否非空
    T *ptr();              // 获取裸指针
    T *operator->();       // 指针解引用
};
```

**关键设计：** 模板类型 `T` 可以**前置声明**——`Ref<T>` 不要求 `T` 完整定义，因为 `unref()` 使用 `reinterpret_cast<RefCounted *>` 转换，利用了 `RefCounted` 单一继承（无虚继承、无多重继承）的保证。

#### 自动 `memnew` 返回值处理

```cpp
template <typename T>
struct memnew_result<T, enable_if_t<is_base_of_v<RefCounted, T>>> {
    using class_name = Ref<T>;
};
```

这意味着 `memnew(MyResource)` 返回 `Ref<MyResource>` 而不是裸指针。

**阅读要点：**
- `init_ref()` vs `reference()`：前者只调用一次（首次绑定到 Ref），后者每次拷贝都调用
- `RefCounted` 的子类包括 `Resource`、`SceneTreeTimer`、`ResourceFormatLoader` 等
- `WeakRef` 使用 `ObjectID` 实现弱引用

---

### 4.4 信号系统：Signal & Callable

**核心文件：** `core/object/object.h` (信号存储), `core/variant/callable.h`

#### 内部数据结构

每个 Object 持有：
```cpp
struct SignalData {
    struct Slot {
        int reference_count = 0;
        Connection conn;                    // 连接信息
        List<Connection>::Element *cE = nullptr;
    };
    MethodInfo user;                        // 信号签名
    HashMap<Callable, Slot> slot_map;       // Callable -> Slot
    bool removable = false;
};

HashMap<StringName, SignalData> signal_map;  // 信号名 -> 数据
List<Connection> connections;                // 全局连接列表
```

**Connection** 结构：
```cpp
struct Connection {
    Signal signal;       // 哪个信号
    Callable callable;   // 调用什么（方法/函数/lamdba）
    uint32_t flags;      // CONNECT_DEFERRED, CONNECT_ONE_SHOT 等
};
```

#### 连接标志

```cpp
CONNECT_DEFERRED = 1,          // 下个闲置帧调用
CONNECT_PERSIST = 2,           // 保存在场景文件中
CONNECT_ONE_SHOT = 4,          // 触发一次后自动断开
CONNECT_REFERENCE_COUNTED = 8, // 引用计数
```

#### 信号生命周期

```
emit_signal("my_signal", args)
  └─> emit_signalp(name, args, argc)
       └─> 从 signal_map 查找 SignalData
            └─> 遍历 slot_map
                 ├─> CONNECT_DEFERRED: 推入 MessageQueue
                 └─> 同步调用 Callable::call()
```

**Callable** 可以包装：
- `MethodBind`（C++ 方法）
- GDScript 方法
- Lambda 函数
- 自定义 Callable（GDExtension）

---

### 4.5 StringName 字符串驻留

**文件：** `core/string/string_name.h` (212行)

StringName 是 Godot 的**字符串驻留**机制——相同的字符串共享同一份内存，比较时只需比较指针。

```cpp
class StringName {
    struct _Data {
        SafeRefCount refcount;
        String name;
        uint32_t hash;
        StringName *cname;  // 指向 C 字符串常量
    };
    _Data *_data = nullptr;
};
```

**关键特性：**
- 全局哈希表存储所有 `_Data` 节点
- `SNAME("foo")` 宏创建静态持久的 StringName
- 比较使用直接的指针比较（`_data == other._data`），极快
- 主要用于属性名、信号名、方法名等频繁比较的字符串

---

### 4.6 节点系统：Node & SceneTree

**文件：** `scene/main/node.h` (951行), `scene/main/scene_tree.h` (477行)

#### Node 结构

```cpp
class Node : public Object {
    GDCLASS(Node, Object);
    // 祖先标记用于快速类型识别
    static constexpr AncestralClass static_ancestral_class = AncestralClass::NODE;

    struct Data {
        String scene_file_path;
        Node *parent, *owner;
        HashMap<StringName, Node*> children;     // 名字 -> 子节点
        mutable LocalVector<Node*> children_cache; // 有序缓存
        StringName name;
        SceneTree *tree;
        Viewport *viewport;
        HashMap<StringName, GroupData> grouped;  // 分组
        int multiplayer_authority = 1;
        ProcessMode process_mode : 3;
        // ... 位域标志：physics_process, process, input 等
    } data;
};
```

**关键方法：**
- `add_child()`, `remove_child()`：树操作
- `_ready()`, `_process(delta)`, `_physics_process(delta)`：生命周期回调
- `enter_tree()`, `exit_tree()`：树进入/退出通知
- `queue_free()`：安全延迟删除
- `rpc()`：远程过程调用

#### 通知 (Notification) 系统

Node 使用整数 ID 标识通知：
```cpp
NOTIFICATION_ENTER_TREE = 10
NOTIFICATION_EXIT_TREE = 11
NOTIFICATION_READY = 13
NOTIFICATION_PROCESS = 17
NOTIFICATION_PHYSICS_PROCESS = 16
NOTIFICATION_WM_CLOSE_REQUEST = 1006
```

通知分**正向**和**反向**传播：
- `_notification_forward()`：先通知子类，再通知脚本（从基类到子类）
- `_notification_backward()`：先通知脚本，再通知子类（从子类到基类）

#### SceneTree

```cpp
class SceneTree : public MainLoop {
    Window *root;                    // 根窗口
    Node *current_scene;             // 当前场景
    HashMap<StringName, SceneTreeGroup> group_map; // 组管理
    
    // 处理组（多线程支持）
    struct ProcessGroup {
        CallQueue call_queue;
        Vector<Node*> nodes;         // process 节点
        Vector<Node*> physics_nodes; // physics_process 节点
        Node *owner;
    };
    PagedAllocator<ProcessGroup> group_allocator;
    
    // 时间管理
    List<Ref<SceneTreeTimer>> timers;
    List<Ref<Tween>> tweens;
    
    double physics_process_time, process_time;
    bool paused;
};
```

**关键特点：**
- `ProcessGroup` 支持多线程节点处理
- `SceneTreeFTI` 管理客户端物理插值
- `group_map` 实现节点分组广播
- `delete_queue` 实现安全延迟删除

---

### 4.7 图形界面：Control 体系

**文件：** `scene/gui/control.h` (873行)

#### Control 继承链

```
Object -> Node -> CanvasItem -> Control
                                ├── BaseButton -> Button
                                ├── Label
                                ├── LineEdit
                                ├── Container -> BoxContainer, GridContainer...
                                ├── Popup -> PopupMenu
                                ├── Panel
                                ├── Tree, ItemList, TabBar...
                                └── ...
```

#### Control 核心结构

```cpp
class Control : public CanvasItem {
    struct Data {
        // 定位
        real_t offset[4], anchor[4];     // 锚点 + 偏移
        Point2 pos_cache, size_cache;
        Size2 minimum_size_cache, maximum_size_cache;
        real_t rotation; Vector2 scale;
        GrowDirection h_grow, v_grow;

        // 容器尺寸标志
        BitField<SizeFlags> h_size_flags, v_size_flags;
        real_t expand;

        // 输入
        MouseFilter mouse_filter;
        CursorShape default_cursor;
        FocusMode focus_mode;

        // 主题
        Ref<Theme> theme;
        // 6种主题覆盖映射表
        Theme::ThemeIconMap theme_icon_override;
        Theme::ThemeStyleMap theme_style_override;
        // ...

        // 无障碍
        String accessibility_name, accessibility_description;
    };
};
```

#### 布局系统

Control 使用**锚点 (Anchor)** + **偏移 (Offset)** 布局模型：
- 四个方向 (LEFT, TOP, RIGHT, BOTTOM) 各有一个锚点值 (0.0-1.0) 和偏移值
- `LayoutPreset` 提供预设布局（居中、铺满等）
- `Container` 子类实现自动布局（`HBoxContainer`、`GridContainer` 等）

#### 事件处理

```cpp
// 三种鼠标过滤模式
MOUSE_FILTER_STOP    // 接收事件并阻止传递
MOUSE_FILTER_PASS    // 接收后继续传递给下层
MOUSE_FILTER_IGNORE  // 完全忽略

// 拖放系统
get_drag_data()      // 拖拽源
can_drop_data()      // 放置验证
drop_data()          // 放置处理

// 输入回调
virtual void gui_input(const Ref<InputEvent> &p_event);
```

---

### 4.8 渲染服务：RenderingServer

**文件：** `servers/rendering/rendering_server.h` (1175行)

#### 架构

```
RenderingServer (抽象单例)
  ├── RendererCompositor (渲染合成器)
  │    ├── RendererCompositorRD      (Forward+ / Mobile 渲染器，基于 Vulkan/D3D12/Metal)
  │    └── RasterizerGLES3           (兼容渲染器，基于 OpenGL ES 3.0)
  ├── RenderingDevice (抽象设备接口)
  │    ├── RenderingDeviceDriverVulkan  (Vulkan 驱动)
  │    ├── RenderingDeviceDriverD3D12   (Direct3D 12 驱动)
  │    └── RenderingDeviceDriverMetal   (Metal 驱动)
  └── RendererCanvasRender (2D 渲染器)
  └── RendererSceneRender (3D 场景渲染器)
```

#### RenderingServer API 分类（约200个纯虚方法）

| API 分组 | 关键方法 |
|---------|---------|
| 纹理 (TEXTURE) | `texture_2d_create()`, `texture_2d_layered_create()` |
| 着色器 (SHADER) | `shader_create()`, `shader_set_code()` |
| 材质 (MATERIAL) | `material_create()`, `material_set_param()` |
| 网格 (MESH) | `mesh_create()`, `mesh_add_surface()` |
| 光照 (LIGHT) | `directional_light_create()`, `omni_light_create()` |
| 视口 (VIEWPORT) | `viewport_create()`, `viewport_set_size()` |
| 环境 (ENVIRONMENT) | `environment_create()`, glow, SSAO, SDFGI 等 |
| 实例 (INSTANCING) | `instance_create()`, `instance_set_base()` |
| 画布 (CANVAS) | `canvas_create()`, `canvas_item_add_*` |
| 粒子 (PARTICLES) | `particles_create()`, `particles_set_emitting()` |

#### RID (资源标识符)

渲染服务使用 **RID**（32/64位整数）作为资源句柄，而不是直接返回指针：
```cpp
struct RID {
    uint64_t _data = 0;  // 高 x 位 = 验证器，低 y 位 = 索引
};
```

RID 提供了：
- 类型安全（不能混淆纹理 RID 和网格 RID）
- 验证（防止使用已释放资源）
- O(1) 查找

#### Vulkan 渲染驱动

```
drivers/vulkan/
  rendering_context_driver_vulkan.h    // VkInstance, VkPhysicalDevice, VkDevice
  rendering_device_driver_vulkan.h     // VkBuffer, VkImage, VkPipeline, 描述符
  rendering_shader_container_vulkan.h  // SPIR-V 着色器管理
  godot_vulkan.h                       // Vulkan 头文件封装 + VMA
```

关键设计：
- 内存管理使用 **VMA** (Vulkan Memory Allocator)
- 缓冲区使用**环形缓冲** (ring-buffered) 避免同步
- 描述符池使用**每个布局自定义池**策略
- 支持计算管线、光线追踪 (BLAS/TLAS)

---

### 4.9 物理系统：PhysicsServer2D/3D

**文件：** `servers/physics_2d/physics_server_2d.h` (863行), `servers/physics_3d/physics_server_3d.h` (1074行)

#### 抽象 API

```cpp
class PhysicsServer2D : public Object {
    // 空间 (Space)
    virtual RID space_create() = 0;
    virtual PhysicsDirectSpaceState2D *space_get_direct_state(RID p_space) = 0;

    // 区域 (Area)
    virtual RID area_create() = 0;
    virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) = 0;

    // 刚体 (Body)
    virtual RID body_create() = 0;
    virtual void body_set_mode(RID p_body, BodyMode p_mode) = 0;
    virtual void body_apply_force(RID p_body, const Vector2 &p_force, const Vector2 &p_position = Vector2()) = 0;

    // 形状 (Shape)
    virtual RID shape_create(ShapeType p_type) = 0;

    // 关节 (Joint)
    virtual RID joint_create(JointType p_type) = 0;

    // 生命周期
    virtual void init() = 0;
    virtual void step(real_t p_step) = 0;
    virtual void sync() = 0;
    virtual void flush_queries() = 0;
    virtual void end_sync() = 0;
    virtual void finish() = 0;
};
```

#### 物理查询

```cpp
class PhysicsDirectSpaceState2D {
    virtual bool intersect_ray(const RayParameters &p_parameters, RayResult &r_result) = 0;
    virtual bool intersect_point(const PointParameters &p_parameters, ...) = 0;
    virtual bool intersect_shape(const ShapeParameters &p_parameters, ...) = 0;
    virtual bool cast_motion(const ShapeParameters &p_parameters, real_t &r_closest_safe, ...) = 0;
    virtual bool collide_shape(const ShapeParameters &p_parameters, ...) = 0;
};
```

#### Godot Physics 实现（默认物理引擎）

```
modules/godot_physics_2d/
  godot_physics_server_2d.h      // 主实现
  godot_space_2d.h               // 空间管理（宽相位+窄相位）
  godot_body_2d.h                // 刚体
  godot_area_2d.h                // 区域（重力/阻尼覆盖）
  godot_shape_2d.h               // 所有形状类型
  godot_step_2d.h                // 物理步进器
  godot_collision_solver_2d.h    // 碰撞检测分发
  godot_collision_solver_2d_sat.h // SAT 分离轴定理
  godot_broad_phase_2d_bvh.h     // BVH 宽相位
  godot_body_pair_2d.h           // 体-体约束求解
```

**宽相位** (Broad Phase)：使用 BVH（包围体层次结构）快速排除不相交的对
**窄相位** (Narrow Phase)：使用 SAT 算法检测精确碰撞
**约束求解**：使用迭代的"约束冲刷" (Constraint Sweep) 算法

---

### 4.10 动画系统：AnimationMixer & Tween

**文件：** `scene/animation/animation_mixer.h` (507行), `scene/animation/tween.h` (359行)

#### 双层架构

```
AnimationMixer (基类)
  ├── AnimationPlayer      — 轨迹动画播放器
  └── AnimationTree        — 混合树（状态机、混合空间）

Tween (独立于 AnimationMixer)
  — 属性插值，流畅 API 链式调用
```

#### AnimationMixer 混音管线

```
1. _blend_init()          — 初始化累加器
2. _blend_pre_process()   — 子类预处理
3. _blend_calc_total_weight() — 计算总权重（不确定模式）
4. _blend_process()       — 遍历动画实例，混合轨迹
5. _blend_apply()         — 将最终值写入对象
6. _blend_post_process()  — 子类清理
```

**轨迹缓存类型：**
- `TrackCacheTransform`：位置/旋转/缩放
- `TrackCacheValue`：通用 Variant 属性
- `TrackCacheMethod`：方法调用轨迹
- `TrackCacheAudio`：音频轨迹
- `TrackCacheAnimation`：嵌套动画轨迹

#### AnimationTree 节点体系

```
AnimationNode (资源基类)
  ├── AnimationNodeAnimation     — 播放特定动画
  ├── AnimationNodeBlend2/3      — 混合 N 个输入
  ├── AnimationNodeBlendTree     — 节点图
  ├── AnimationNodeStateMachine  — 状态机
  ├── AnimationNodeBlendSpace1D  — 一维混合空间
  ├── AnimationNodeBlendSpace2D  — 二维混合空间
  ├── AnimationNodeOneShot       — 一次性触发
  └── AnimationNodeTransition    — 过渡
```

#### Tween 系统

```cpp
class Tween : public RefCounted {
    // 168 个插值函数表 (12种过渡 x 7种缓动 x 2方向)
    static interpolater interpolaters[TRANS_MAX][EASE_MAX];

    // 流畅 API 链式调用
    tween_property(object, "position", target_pos, 1.0)
        ->set_trans(Tween::TRANS_BOUNCE)
        ->set_ease(Tween::EASE_OUT)
    tween_callback(my_callable)
    tween_interval(0.5)
    parallel()  // 并行执行后续动画
    chain()     // 串行执行
};
```

---

### 4.11 资源系统：Resource & ResourceLoader

**文件：** `core/io/resource.h` (215行), `core/io/resource_loader.h` (316行)

#### Resource 基类

```cpp
class Resource : public RefCounted {
    String name;
    String path_cache;           // 资源路径
    String scene_unique_id;
    bool local_to_scene;         // 场景内独立副本
    ObjectID local_scene;        // 所属场景

    void emit_changed();         // 改变通知
    virtual Ref<Resource> duplicate(bool p_deep = false) const;
};
```

**资源缓存**：
```cpp
class ResourceCache {
    static HashMap<String, Resource*> resources;  // 路径 -> Resource
    static Mutex lock;
};
```

#### ResourceLoader

```cpp
class ResourceLoader {
    static Ref<Resource> load(const String &p_path, const String &p_type_hint = "",
                              CacheMode p_cache_mode = CACHE_MODE_REUSE);
    
    // 异步加载
    static Error load_threaded_request(const String &p_path);
    static ThreadLoadStatus load_threaded_get_status(const String &p_path, Vector<String> *r_progress = nullptr);
    static Ref<Resource> load_threaded_get(const String &p_path);
    
    // 内部：遍历已注册的 loader
    static Ref<ResourceFormatLoader> loader[MAX_LOADERS=64];
    static int loader_count;
};
```

#### ResourceFormatLoader

```cpp
class ResourceFormatLoader : public RefCounted {
    virtual Ref<Resource> load(const String &p_path, const String &p_original_path,
                               Error *r_error, bool p_use_sub_threads = false,
                               real_t *r_progress = nullptr,
                               CacheMode p_cache_mode = CACHE_MODE_REUSE) = 0;
    virtual void get_recognized_extensions(List<String> *p_extensions) const = 0;
    virtual bool handles_type(const String &p_type) const = 0;
};
```

**已注册的加载器：**
- `ResourceFormatLoaderBinary` — `.res` / `.tscn` 二进制格式
- `ResourceFormatLoaderText` — `.tres` 文本格式
- `ResourceFormatImporter` — 导入资源（纹理、音频等）
- GDScript/C# 的资源格式加载器

---

### 4.12 输入系统：Input & InputMap

**文件：** `core/input/input.h` (493行), `core/input/input_map.h` (117行), `core/input/input_event.h` (600行)

#### 三层架构

```
InputEvent (资源基类)
  ├── InputEventKey
  ├── InputEventMouseButton
  ├── InputEventMouseMotion
  ├── InputEventJoypadButton
  ├── InputEventJoypadMotion
  ├── InputEventScreenTouch/Drag
  ├── InputEventAction (程序化动作)
  └── InputEventMIDI

InputMap (动作映射)
  └── StringName -> Action { deadzone, List<InputEvent> inputs }

Input (单例，运行时状态)
  ├── is_key_pressed(), is_mouse_button_pressed()
  ├── is_action_pressed(), is_action_just_pressed()
  ├── get_action_strength(), get_vector()
  └── parse_input_event() — 事件注入点
```

#### InputEvent 与动作匹配

```cpp
class InputEvent : public Resource {
    // 每个子类重写 action_match() 方法
    virtual bool action_match(const Ref<InputEvent> &p_event,
                              bool &r_pressed, float &r_strength,
                              float &r_raw_strength, float p_deadzone) const;
};
```

`InputMap::event_is_action()` 遍历动作的所有绑定输入事件，调用 `action_match()` 检查匹配。

---

### 4.13 导航系统：NavigationServer2D/3D

**文件：** `servers/navigation_2d/navigation_server_2d.h`, `modules/navigation_2d/nav_map_2d.h`

#### 抽象层

```cpp
class NavigationServer2D : public Object {
    // 地图
    virtual RID map_create() = 0;
    virtual void map_set_active(RID p_map, bool p_active) = 0;
    virtual Vector<Vector2> map_get_path(RID p_map, const Vector2 &p_origin,
                                          const Vector2 &p_destination,
                                          bool p_optimize, uint32_t p_navigation_layers = 1) const = 0;
    
    // 区域（导航网格）
    virtual RID region_create() = 0;
    virtual void region_set_map(RID p_region, RID p_map) = 0;
    virtual void region_set_navigation_polygon(RID p_region, const Ref<NavigationPolygon> &p_polygon) = 0;
    
    // 链接（区域间连接）
    virtual RID link_create() = 0;
    
    // 代理（RVO 避障）
    virtual RID agent_create() = 0;
    virtual void agent_set_velocity(RID p_agent, const Vector2 &p_velocity) = 0;
};
```

#### 模块实现

```cpp
// modules/navigation_2d/nav_map_2d.h
class NavMap2D {
    RBMap<ObjectID, NavRegion2D> regions;
    RBMap<ObjectID, NavLink2D> links;
    RVO2D::RVOSimulator2D rvo_simulator;  // RVO 避障模拟器
    
    // 异步迭代（WorkerThreadPool）
    mutable WorkerThreadPool::TaskID nav_thread;
    mutable bool nav_thread_running;
    
    // 路径查询
    struct NavMeshPathQueryTask2D { ... };
    Vector2 map_get_path(...);
};
```

---

### 4.14 音频系统：AudioServer

**文件：** `servers/audio/audio_server.h` (563行)

```cpp
class AudioServer : public Object {
    Vector<Bus *> buses;                    // 总线系统（含效果器链）
    SafeList<AudioStreamPlaybackListNode *> playbacks; // 播放列表
    
    // 总线管理
    void set_bus_count(int count);
    void add_bus_effect(int bus_idx, Ref<AudioEffect> effect);
    
    // 播放控制
    float get_mix_rate();
    float get_output_latency();
    
    // 全局
    void set_playback_speed_scale(float scale);
};

class AudioDriver {
    virtual const char *get_name() = 0;
    virtual Error init() = 0;
    virtual int get_mix_rate() const = 0;
    void audio_server_process(int frames, int32_t *buffer, bool update_mix_time = true);
};
```

播放状态机：
```
PLAYING -> FADE_OUT_TO_PAUSE -> PAUSED
       └-> FADE_OUT_TO_DELETION -> AWAITING_DELETION
```

---

### 4.15 GDScript 语言

**模块目录：** `modules/gdscript/`

#### 四阶段执行模型

```
源码 (.gd)
  └─> Tokenizer (gdscript_tokenizer.h)       — 词法分析
       └─> Parser (gdscript_parser.h)         — 语法分析 -> AST
            └─> Analyzer (gdscript_analyzer.h) — 语义分析/类型检查
                 └─> Compiler (gdscript_compiler.h) — 生成字节码
                      └─> VM (gdscript_vm.cpp) — 执行字节码
```

#### 核心类

```cpp
class GDScript : public Script {
    HashMap<StringName, GDScriptFunction*> member_functions;   // 成员函数
    HashMap<StringName, Ref<GDScript>> subclasses;              // 子类
    
    GDScriptFunction *initializer;          // _init()
    GDScriptFunction *implicit_ready;       // _ready()
    Vector<uint8_t> binary_tokens;           // 序列化字节码
};

class GDScriptInstance : public ScriptInstance {
    Object *owner;
    Ref<GDScript> script;
    Vector<Variant> members;                // 实例成员变量
};

class GDScriptLanguage : public ScriptLanguage {
    Vector<Variant> global_array;   // 全局变量
    struct CallLevel {               // 调用栈
        void *sp; GDScriptFunction *function;
        GDScriptInstance *instance;
        int ip, line;
        CallLevel *previous;
    };
};
```

#### 解析器 (Parser)

使用 **Pratt 解析**（自顶向下运算符优先级解析）：
- 40+ 种 AST 节点类型
- 19 级运算符优先级 (`PREC_NONE` ~ `PREC_PRIMARY`)
- `parse_precedence()` 结合前缀/中缀函数表

#### 虚拟机的关键特征

`gdscript_vm.cpp`（4042行）包含 `GDScriptFunction::call()`——主 VM 执行入口：
- 字节码分派循环
- 内置函数调用
- 类型检查与错误报告
- 分析器集成

---

### 4.16 C# / Mono 模块

**模块目录：** `modules/mono/`

```cpp
class CSharpScript : public Script {
    struct TypeInfo {
        StringName class_name;
        StringName native_base_name;
        bool is_tool, is_abstract, is_global_class;
        bool is_constructed_generic_type;
    };
    TypeInfo type_info;
    HashSet<Object *> instances;       // 所有实例
    MonoGCHandleData gchandle;         // GC 句柄
};

class CSharpLanguage : public ScriptLanguage {
    GDMono *gdmono;                                    // Mono 运行时桥接
    RBMap<Object*, CSharpScriptBinding> script_bindings; // 本地-托管绑定
    void tie_native_managed_to_unmanaged(Object *p_owner, Object *p_native);
};
```

**MonoGCHandleData** (`mono_gc_handle.h`)：
```cpp
struct MonoGCHandleData {
    GCHandleIntPtr handle;
    GCHandleType type;  // NIL, STRONG_HANDLE, WEAK_HANDLE
    void release();     // 释放 GC 句柄
};
```

**胶水代码生成：** `modules/mono/glue/` 包含完整的 C# Godot API 项目：
- `GodotSharp/` — 核心 C# API
- `GodotSharpEditor/` — 编辑器 C# API
- `Godot.SourceGenerators.Internal/` — Roslyn 源码生成器

---

### 4.17 多线程与并发

**文件：** `core/os/thread.h`, `core/os/mutex.h`, `core/object/worker_thread_pool.h`, `core/templates/safe_refcount.h`

#### Thread

```cpp
class Thread {
    using Callback = void (*)(void*);
    using ID = uint64_t;
    
    static ID MAIN_ID = 1;
    static thread_local ID caller_id;
    
    void start(Callback, void *userdata, Settings = {});
    void wait_to_finish();
    static ID get_caller_id();
    static bool is_main_thread();
};
```

**线程 ID 分配：**
- `MAIN_ID = 1`
- `UNASSIGNED_ID = 0`
- 后续 ID 从 `SafeNumeric<uint64_t>` 递增

#### Mutex 体系

```cpp
using Mutex = MutexImpl<std::recursive_mutex>;       // 可重入
using BinaryMutex = MutexImpl<std::mutex>;            // 不可重入
using MutexLock = MutexLock<MutexT>;                  // RAII 锁 ([[nodiscard]])
```

#### WorkerThreadPool

```cpp
class WorkerThreadPool : public Object {
    // 任务类型
    TaskID add_task(const Callable &p_callable);
    void add_native_task(void (*p_func)(void*), void *p_userdata);
    
    // 组任务（类似 parallel-for）
    void add_group_task(...);      // 并行处理多个元素
    
    // 同步
    void wait_for_task_completion(TaskID);
    bool is_task_completed(TaskID);
    
    // 运行级别控制
    enum RunLevel { NORMAL, PRE_EXIT_LANGUAGES, EXIT_LANGUAGES, EXIT };
    void init(int thread_count, float low_priority_ratio);
    void finish();
};
```

#### 安全原子操作

```cpp
template<typename T>
class SafeNumeric {
    std::atomic<T> value;
    T increment();    // acquire-release
    T decrement();
    bool conditional_increment();  // CAS 循环，0 时不递增
};

class SafeFlag {
    std::atomic_bool flag;
    bool is_set(), set(), clear();
};

class SafeRefCount {
    SafeNumeric<uint32_t> refcount;
    bool ref();       // 条件递增
    bool unref();     // 递减，返回 true 表示到 0
};
```

---

### 4.18 编辑器架构

**核心文件：** `editor/editor_node.h` (1061行)

#### EditorNode

```cpp
class EditorNode : public Node {
    EditorData editor_data;
    EditorSelectionHistory editor_history;
    Vector<EditorPlugin*> editor_plugins;
    EditorMainScreen *editor_main_screen;
    EditorResourcePreview *resource_preview;
    SubViewport *scene_root;
    
    // 菜单系统
    enum MenuOptions {
        FILE_NEW, FILE_OPEN, FILE_SAVE, FILE_QUIT,
        PROJECT_SETTINGS, PROJECT_EXPORT,
        EDITOR_SETTINGS, COMMAND_PALETTE,
        HELP_SEARCH, HELP_DOCS, HELP_ABOUT
    };
};
```

#### EditorPlugin 插件系统

```cpp
class EditorPlugin : public Node {
    // 虚拟方法
    virtual bool _handles(Object *p_object) const;
    virtual void _edit(Object *p_object);
    virtual bool _forward_canvas_gui_input(const Ref<InputEvent> &p_event);
    virtual void _forward_canvas_draw_over_viewport(Control *p_overlay);
    
    // 注册点
    void add_control_to_container(CustomControlContainer, Control*);
    void add_import_plugin(Ref<EditorImportPlugin>);
    void add_export_plugin(Ref<EditorExportPlugin>);
    void add_inspector_plugin(Ref<EditorInspectorPlugin>);
    void add_custom_type(const StringName &type, const StringName &base, Ref<Script> script, const Ref<Texture2D> &icon);
};
```

#### 主题系统

`editor/themes/editor_theme_manager.h`：
```cpp
class EditorThemeManager {
    struct ThemeConfiguration {
        Color base_color, accent_color, contrast;
        float contrast_variation;
        // ~50 个颜色/样式属性
    };
    Ref<Theme> generate_theme();
};
```

编辑器子目录：
```
editor/animation/      — 动画编辑器
editor/asset_library/  — 资源库
editor/debugger/       — 调试器
editor/docks/          — 停靠面板
editor/export/         — 导出系统
editor/filesystem/     — 文件系统
editor/import/         — 导入配置
editor/inspector/      — 检查器
editor/plugins/        — 插件
editor/scene/          — 场景编辑器
editor/script/         — 脚本编辑器
editor/themes/         — 主题
editor/version_control/— 版本控制
```

---

### 4.19 资源导入管道

**文件：** `core/io/resource_importer.h` (176行)

```cpp
class ResourceImporter : public RefCounted {
    virtual String get_importer_name() const = 0;
    virtual String get_recognized_extensions() const = 0;
    virtual String get_save_extension() const = 0;
    virtual String get_resource_type() const = 0;
    virtual List<ImportOption> get_import_options() const = 0;
    virtual Error import(const String &p_source_file, const String &p_save_path,
                         const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants,
                         List<String> *r_gen_files, Variant *r_metadata) = 0;
};

class ResourceFormatImporter : public ResourceFormatLoader {
    static ResourceFormatImporter *singleton;
    Vector<Ref<ResourceImporter>> importers;  // 所有导入器
};
```

**导入流程：**
1. 资源文件（如 `.png`）放入项目
2. 编辑器检测到新文件，调用匹配的 `ResourceImporter`
3. 导入器生成 `.ctex`（导入后格式）在 `.godot/imported/` 下
4. 运行时 `ResourceLoader` 加载 `.ctex` 文件

---

### 4.20 平台抽象层

**示例：Windows 平台** (`platform/windows/`)

```
platform/windows/
  detect.py                              — SCons 检测和配置
  godot_windows.cpp                      — WinMain 入口
  os_windows.h/cpp                       — OS_Windows (2936行)
  display_server_windows.h/cpp           — 窗口管理
  crash_handler_windows.h/cpp            — SEH 异常处理
  key_mapping_windows.h/cpp              — 虚拟键映射
  gl_manager_windows_native.h/cpp        — 原生 OpenGL
  gl_manager_windows_angle.h/cpp         — ANGLE (OpenGL over D3D11)
  rendering_context_driver_vulkan_windows.h/cpp — Vulkan surface
  tts_windows.h/cpp                      — 文本转语音
  native_menu_windows.h/cpp              — 原生菜单
  drop_target_windows.h/cpp              — 拖放支持
```

**OS_Windows 核心方法：**
```cpp
void initialize();            // 音频、渲染初始化
void run();                   // 消息循环 (PeekMessage/DispatchMessage)
void delay_usec(int p_usec);  // 高精度延时 (QueryPerformanceCounter)
Error execute(const String &p_path, const List<String> &p_args, ...);  // 进程创建 (CreateProcess)
String get_system_fonts();    // DirectWrite 字体查询
void move_to_trash(const String &p_path);  // 回收站删除 (IFileOperation)
Vector<uint8_t> get_entropy(int p_count);  // 加密随机数 (BCryptGenRandom)
```

**detect.py 关键配置：**
- MSVC: `/ENTRY:mainCRTStartup`, `/fp:strict`, `/utf-8`, `/bigobj`
- MinGW: `-Wa,-mbig-obj`, `-ffp-contract=off`
- 定义：`WINDOWS_ENABLED`, `WASAPI_ENABLED`, `VULKAN_ENABLED`, `D3D12_ENABLED`

---

### 4.21 网络/多人系统

**文件：** `scene/main/multiplayer_api.h`, `modules/multiplayer/scene_multiplayer.h`

#### 多层架构

```
MultiplayerAPI (抽象基类)
  └── SceneMultiplayer (默认实现)
       ├── SceneRPCInterface    — RPC 分发
       ├── SceneCacheInterface  — 路径缓存
       └── SceneReplicationInterface — 状态同步

MultiplayerPeer (传输层抽象)
  ├── ENetMultiplayerPeer
  ├── WebSocketMultiplayerPeer
  └── OfflineMultiplayerPeer (单机)
```

#### RPC 调用流程

```cpp
// Node 上调用
node->rpc("some_method", arg1, arg2);

// 内部流程
Node::rpc()
  └─> MultiplayerAPI::rpcp()
       └─> SceneMultiplayer::rpcp()
            └─> SceneRPCInterface::_process_rpc()
                 └─> 序列化参数，发送到 MultiplayerPeer
```

---

### 4.22 文件 I/O 系统

**文件：** `core/io/file_access.h`, `core/io/dir_access.h`, `core/io/compression.h`

#### FileAccess

```cpp
class FileAccess : public RefCounted {
    enum AccessType { ACCESS_RESOURCES, ACCESS_USERDATA, ACCESS_FILESYSTEM, ACCESS_PIPE };
    enum ModeFlags { READ=1, WRITE=2, READ_WRITE=3, WRITE_READ=7, SKIP_PACK=16 };
    
    // 工厂方法
    static Ref<FileAccess> create(AccessType p_access);
    static Ref<FileAccess> open(const String &p_path, int p_mode_flags);
    
    // 读
    uint8_t get_8(); uint16_t get_16(); uint32_t get_32(); uint64_t get_64();
    float get_float(); double get_double(); real_t get_real();
    String get_line(); String get_var();  // 读取 Variant
    Vector<uint8_t> get_buffer(int p_length);
    
    // 写
    void store_8(uint8_t); void store_16(uint16_t);
    void store_float(float); void store_double(double);
    void store_string(const String &p_string);
    void store_var(const Variant &p_var);  // 写入 Variant
    
    // 静态工具
    static String get_file_as_string(const String &p_path);
    static Vector<uint8_t> get_file_as_bytes(const String &p_path);
};
```

**平台注册：** `FileAccess::make_default<T>(AccessType)` 注册平台实现

#### DirAccess

```cpp
class DirAccess : public RefCounted {
    Error list_dir_begin();
    String get_next();
    bool current_is_dir();
    void list_dir_end();
    
    Error make_dir(const String &p_path);
    Error make_dir_recursive(const String &p_path);
    bool exists(const String &p_path);
    Error copy(const String &p_from, const String &p_to);
    Error rename(const String &p_from, const String &p_to);
    Error remove(const String &p_path);
};
```

#### Compression

```cpp
class Compression {
    enum Mode { MODE_FASTLZ, MODE_DEFLATE, MODE_ZSTD, MODE_GZIP, MODE_BROTLI };
    
    static int compress(uint8_t *p_dst, const uint8_t *p_src, int p_src_size, Mode p_mode);
    static int decompress(uint8_t *p_dst, int p_dst_max, const uint8_t *p_src, int p_src_size, Mode p_mode);
    static int decompress_dynamic(Vector<uint8_t> *p_dst, int p_max_size, const uint8_t *p_src, int p_src_size, Mode p_mode);
    
    // 可配置压缩级别
    static int zlib_level, gzip_level, zstd_level;
};
```

---

## 5. 关键宏与模式总结

### 类注册宏

| 宏 | 用途 | 所在文件 |
|-----|------|---------|
| `GDCLASS(MyClass, Parent)` | ClassDB 注册类 | `object.h:248` |
| `GDSOFTCLASS(MyClass, Parent)` | 仅 cast_to 支持，不注册 | `object.h:161` |
| `GDVIRTUAL_CALL(m_name, ...)` | 调用 GDExtension 可重写的虚方法 | `object.h:141` |
| `OBJ_SAVE_TYPE(MyClass)` | 设置保存时的类名 | `object.h:338` |

### 属性/方法/信号绑定宏

| 宏 | 用途 |
|-----|------|
| `ADD_SIGNAL(m_signal)` | 注册信号 |
| `ADD_PROPERTY(prop, setter, getter)` | 注册属性 |
| `ADD_PROPERTYI(prop, setter, getter, index)` | 带索引属性 |
| `ADD_PROPERTY_DEFAULT(prop, default)` | 默认值 |
| `ADD_GROUP(name, prefix)` | 属性分组 |
| `BIND_METHOD(obj, method)` | 绑定方法到 ClassDB |

### 常用模板

| 模板 | 用途 |
|------|------|
| `Vector<T>` | 动态数组（COW 写时复制） |
| `List<T>` | 双向链表 |
| `HashMap<K,V>` | 哈希表（开放寻址） |
| `HashSet<T>` | 哈希集合 |
| `LocalVector<T>` | 局部向量（不写时复制） |
| `RBMap<K,V>` | 红黑树有序映射 |
| `RID` | 资源标识符 |
| `Ref<T>` | 引用计数智能指针 |
| `PagedAllocator<T>` | 页面分配器（高缓存效率） |

### 关键设计模式

| 模式 | 示例 |
|------|------|
| **单例** | `RenderingServer::get_singleton()`, `Input::get_singleton()` |
| **抽象服务** | `RenderingServer` + `RendererCompositorRD` |
| **工厂注册** | `DisplayServer::register_create_function()`, `ResourceFormatLoader` |
| **RID 句柄** | GPU 资源通过 `RID` 引用，而非裸指针 |
| **方法绑定** | C++ 方法通过 `MethodBind` 暴露到脚本 |
| **信号/槽** | `Object::connect()` / `emit_signal()` |
| **策略模式** | `PhysicsServer2DManager` 可切换物理引擎 |
| **RAII** | `MutexLock`、`Ref<T>` |

---

## 6. 核心类继承关系

```
Object (core/object/object.h)
  ├── RefCounted (core/object/ref_counted.h)
  │     ├── Resource (core/io/resource.h)
  │     │     ├── PackedScene (scene/resources/packed_scene.h)
  │     │     ├── Material (scene/resources/material.h)
  │     │     ├── Shader (scene/resources/shader.h)
  │     │     ├── Texture / Texture2D (scene/resources/texture.h)
  │     │     ├── Mesh (scene/resources/mesh.h)
  │     │     ├── Theme (scene/resources/theme.h)
  │     │     ├── StyleBox (scene/resources/style_box.h)
  │     │     ├── Animation (scene/resources/animation.h)
  │     │     ├── AnimationNode (scene/animation/animation_tree.h)
  │     │     ├── InputEvent (core/input/input_event.h)
  │     │     └── Script (core/object/script_language.h)
  │     │           ├── GDScript (modules/gdscript/gdscript.h)
  │     │           └── CSharpScript (modules/mono/csharp_script.h)
  │     ├── ResourceFormatLoader (core/io/resource_loader.h)
  │     ├── SceneTreeTimer (scene/main/scene_tree.h)
  │     ├── ResourceImporter (core/io/resource_importer.h)
  │     ├── Tween (scene/animation/tween.h)
  │     └── WeakRef (core/object/ref_counted.h)
  ├── MainLoop (core/os/main_loop.h)
  │     └── SceneTree (scene/main/scene_tree.h)
  ├── Node (scene/main/node.h)
  │     ├── CanvasItem (scene/main/canvas_item.h)
  │     │     ├── Node2D (scene/2d/node_2d.h)
  │     │     │     ├── Sprite2D
  │     │     │     ├── CollisionObject2D
  │     │     │     │     ├── Area2D
  │     │     │     │     ├── PhysicsBody2D
  │     │     │     │     │     ├── RigidBody2D
  │     │     │     │     │     ├── CharacterBody2D
  │     │     │     │     │     └── StaticBody2D
  │     │     │     ├── Camera2D
  │     │     │     ├── TileMapLayer
  │     │     │     ├── Parallax2D
  │     │     │     └── AudioStreamPlayer2D
  │     │     ├── Control (scene/gui/control.h)
  │     │     │     ├── BaseButton -> Button
  │     │     │     ├── Label
  │     │     │     ├── LineEdit / TextEdit
  │     │     │     ├── Container
  │     │     │     │     ├── BoxContainer (HBox/VBox)
  │     │     │     │     ├── GridContainer
  │     │     │     │     ├── ScrollContainer
  │     │     │     │     └── SplitContainer
  │     │     │     ├── Popup -> PopupMenu
  │     │     │     ├── Tree, ItemList, TabBar
  │     │     │     └── Panel, ColorRect, TextureRect
  │     │     └── Viewport (scene/main/viewport.h)
  │     │           └── Window (scene/main/window.h)
  │     │                 └── Popup (scene/gui/popup.h)
  │     ├── Node3D (scene/3d/node_3d.h) 
  │     │     ├── VisualInstance3D
  │     │     │     └── GeometryInstance3D -> MeshInstance3D
  │     │     ├── Camera3D
  │     │     ├── Light3D (Directional/Omni/Spot/Area)
  │     │     ├── CollisionObject3D
  │     │     │     ├── Area3D
  │     │     │     └── PhysicsBody3D (Rigid/Character/Static/Animatable)
  │     │     ├── Skeleton3D
  │     │     ├── AudioStreamPlayer3D
  │     │     └── VehicleBody3D
  │     ├── AnimationMixer (scene/animation/animation_mixer.h)
  │     │     ├── AnimationPlayer
  │     │     └── AnimationTree
  │     ├── Timer
  │     ├── MultiplayerSynchronizer
  │     ├── MultiplayerSpawner
  │     └── EditorPlugin (editor/plugins/editor_plugin.h)
  │           └── (各种编辑器插件)
  ├── DisplayServer (servers/display/display_server.h)
  ├── RenderingServer (servers/rendering/rendering_server.h)
  ├── AudioServer (servers/audio/audio_server.h)
  ├── PhysicsServer2D (servers/physics_2d/physics_server_2d.h)
  ├── PhysicsServer3D (servers/physics_3d/physics_server_3d.h)
  ├── NavigationServer2D (servers/navigation_2d/navigation_server_2d.h)
  ├── NavigationServer3D (servers/navigation_3d/navigation_server_3d.h)
  ├── TextServer (servers/text/text_server.h)
  ├── XRServer (servers/xr/xr_server.h)
  ├── CameraServer (servers/camera/camera_server.h)
  ├── Input (core/input/input.h)
  ├── InputMap (core/input/input_map.h)
  ├── OS (core/os/os.h)
  ├── ThemeDB (scene/theme/theme_db.h)
  └── Engine (core/config/engine.h)
```

---

## 7. 附录：小技巧

### 快速定位

- `grep -rn "ClassName"` — 查找类定义
- `git log --oneline --all` — 查看提交历史
- `git blame 文件名` — 查看每行是谁修改的
- SCons 支持 `-j N` 并行构建，`vsproj=yes` 生成 Visual Studio 项目

### 调试技巧

- `DEV_ENABLED` 定义启用开发断言
- `DEBUG_ENABLED` 定义启用调试功能
- `PRINT_VERBOSE_ENABLED` 启用详细日志
- `Main::setup()` 中的 `MAIN_PRINT` 宏打印启动阶段

### 关键预处理器

| 定义 | 影响 |
|------|------|
| `TOOLS_ENABLED` | 编辑器模式 |
| `DEBUG_ENABLED` | 调试模式 |
| `DEV_ENABLED` | 开发者模式 |
| `_3D_DISABLED` | 禁用 3D（头显构建） |
| `PHYSICS_2D_DISABLED` | 禁用 2D 物理 |
| `PHYSICS_3D_DISABLED` | 禁用 3D 物理 |
| `THREADS_ENABLED` | 多线程支持 |

### 测试运行

```bash
# 构建测试
scons platform=windows target=template_debug tests=yes

# 运行所有测试
bin/godot.windows.template_debug.x86_64.exe -t -- --all

# 运行特定测试
bin/godot.windows.template_debug.x86_64.exe -t -- --test-case="[String]"
```

测试使用 **doctest** 框架，测试分类为 `[SceneTree]`、`[Audio]`、`[Navigation3D]` 等，它们在 `tests/test_main.cpp` 中通过 `GodotTestCaseListener` 自动初始化/销毁对应服务。

---

> 本文档基于 Godot 4.7.0-beta 源码编写。随着版本迭代，部分细节可能发生变化，
> 建议结合当前源码进行阅读。理解 Godot 源码的最佳方式是：先读第一遍概览架构，
> 再第二遍深入特定子系统，最后尝试修改和调试代码。
