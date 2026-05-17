// Static source-guide data used by app.js. Keep this file free of DOM code.
const dirs = [
  {
    name: "core",
    files: "465 tracked files / 424 C/C++/ObjC/Python files",
    role: "运行时地基：Object、Variant、ClassDB、Resource、OS、IO、数学、线程、字符串、错误处理。",
    boundary: "只要问题涉及“所有系统都要用的基础能力”，通常先看 core：对象反射、统一值类型、资源加载基础、跨平台 OS 抽象、容器、数学类型、线程和错误宏都在这里。它不应该知道具体的 Sprite2D、EditorNode 或某个渲染后端实现。",
    anchors: ["core/object/object.h:349", "core/object/class_db.h:97", "core/variant/variant.h:93", "core/io/resource_loader.h:103", "core/os/os.h:46"],
    entry: "从 <span class=\"source\">core/register_core_types.cpp:134</span> 开始看它注册了哪些全局类型，再分别进入 <span class=\"source\">core/object</span>、<span class=\"source\">core/variant</span>、<span class=\"source\">core/io</span>、<span class=\"source\">core/os</span>。",
    trail: [
      "脚本调用 C++：Object → ClassDB → MethodBind → Variant。",
      "资源进入运行时：Resource → ResourceLoader → ResourceFormatLoader → ResourceCache。",
      "平台能力抽象：OS 虚函数 → platform/*/os_* 实现。"
    ],
    questions: [
      "脚本如何调用 C++ 方法？看 Object、ClassDB、MethodBind、Variant。",
      "资源如何被加载、缓存、保存？看 core/io/resource_loader、resource_saver、resource。",
      "跨平台能力如何抽象？看 core/os/os.h 和各平台 OS 子类。"
    ],
    pitfalls: [
      "不要把 ObjectDB 当成所有权系统；它是全局对象登记和 ObjectID 查找，不负责替你释放对象。",
      "不要把 Resource 和 Node 混在一起；Resource 用 Ref 引用计数，Node 由场景树或手动生命周期管理。",
      "不要忽略 Variant 成本；它统一了调用语义，但容器、Object、PackedArray 的拷贝/引用规则会影响性能和生命周期。"
    ],
    read: "先读 Object/ClassDB/Variant，再读 ResourceLoader，最后按需要进入 OS、math、templates。"
  },
  {
    name: "scene",
    files: "836 tracked files / 678 C/C++/ObjC/Python files",
    role: "用户最常接触的对象层：Node、SceneTree、2D/3D 节点、GUI、动画、主题、资源类型。",
    boundary: "只要问题涉及“用户在编辑器或脚本里直接操作的对象”，通常先看 scene。它把用户语义封装成 Node、Resource、Control、Animation、Viewport 等高层对象，再把底层执行交给 servers。",
    anchors: ["scene/main/node.h:54", "scene/main/scene_tree.h:89", "scene/register_scene_types.cpp:390", "scene/resources/packed_scene.h:246"],
    entry: "先读 <span class=\"source\">scene/register_scene_types.cpp:390</span> 看注册了哪些节点和资源，再读 <span class=\"source\">scene/main</span> 建立 Node/SceneTree/Viewport 骨架。",
    trail: [
      "场景生命周期：PackedScene → Node 实例化 → SceneTree enter_tree/ready/process。",
      "2D/3D 可见对象：CanvasItem/Node3D → RenderingServer RID。",
      "GUI：Control → Container/Theme/Font → TextServer/RenderingServer。"
    ],
    questions: [
      "一个节点如何进入树、ready、process、退出？看 scene/main/node.* 和 scene_tree.*。",
      "2D/3D 节点如何提交渲染或物理状态？看 scene/2d、scene/3d，再跟到 servers。",
      "GUI 控件如何布局和绘制？看 scene/gui、scene/theme、servers/text。"
    ],
    pitfalls: [
      "不要以为 scene 直接执行渲染或物理；它通常只维护用户语义、变换、资源引用和 RID。",
      "不要只看 cpp 函数名；Node 生命周期、process 列表、groups、owner、internal mode 等关键状态都在头文件的数据结构里。",
      "不要跳过 Viewport/Window；输入、渲染目标、世界对象和 GUI 分发都绕不开它们。"
    ],
    read: "先读 scene/main，再按功能进入 2d、3d、gui、animation、resources。"
  },
  {
    name: "servers",
    files: "463 tracked files / 319 C/C++/ObjC/Python files",
    role: "底层能力抽象：渲染、物理、导航、音频、显示、文本、XR、调试、摄像头、电影写出。",
    boundary: "只要问题涉及“真正执行重活的系统接口”，通常进入 servers。它定义跨场景、跨平台的服务 API，隐藏后端差异，并通过 RID 或单例管理底层状态。",
    anchors: ["servers/register_server_types.cpp:146", "servers/rendering/rendering_server.h:64", "servers/physics_3d/physics_server_3d.h:236", "servers/display/display_server.h:62"],
    entry: "先读 <span class=\"source\">servers/register_server_types.cpp:146</span> 看服务类型和单例怎么注册，再进入具体 server 的抽象头文件，而不是直接跳到 backend。",
    trail: [
      "渲染：RenderingServer API → renderer_rd/storage 或 canvas/scene → RenderingDevice。",
      "物理：PhysicsServer2D/3D API → Manager 选择实现 → godot_physics 或 jolt 模块。",
      "显示：DisplayServer 抽象 → platform/* 的窗口、输入法、屏幕、剪贴板实现。"
    ],
    questions: [
      "为什么节点只拿 RID？看 RenderingServer/PhysicsServer API 和 RID_Owner。",
      "如何替换物理或文本后端？看 ServerManager、Extension、modules 实现。",
      "多线程渲染/物理如何隔离？看 wrap_mt、sync、flush、draw、step。"
    ],
    pitfalls: [
      "不要把 Server API 和具体后端混为一谈；server.h 多数是抽象契约，真实执行常在 modules 或 renderer_rd。",
      "不要把 RID 当 Resource；RID 是服务内部对象句柄，释放路径通常是对应 Server 的 free。",
      "不要忽略 sync/flush/draw/step；很多 Server 调用会延迟到这些阶段才真正生效。"
    ],
    read: "先读 register_server_types，再读对应 server.h，最后读具体 backend。"
  },
  {
    name: "modules",
    files: "3036 tracked files / 989 C/C++/ObjC/Python files",
    role: "可选功能和第三方库包装：GDScript、C#、物理实现、导入器、编解码、网络、XR、文本后端。",
    boundary: "modules 是功能接入点，不是单一层。一个模块可以在 CORE 注册资源加载器，在 SERVERS 注册后端，在 SCENE 注册节点，在 EDITOR 注册导入器或面板。判断模块位置要看它的 register_types 和初始化 level。",
    anchors: ["modules/SCsub:20", "modules/modules_builders.py:43", "modules/gdscript/register_types.cpp:137", "modules/jolt_physics", "modules/gltf"],
    entry: "先看 <span class=\"source\">modules/SCsub:20</span> 和 <span class=\"source\">modules/modules_builders.py:43</span> 理解模块启用表与初始化调度代码如何生成，再进入目标模块的 <span class=\"source\">register_types.*</span> 和 <span class=\"source\">SCsub</span>。",
    trail: [
      "脚本语言：gdscript/mono → ScriptLanguage → Script/ScriptInstance → Object/ClassDB。",
      "导入格式：gltf/fbx/svg/webp 等 → ResourceFormatLoader/Importer → editor/import。",
      "后端实现：godot_physics/jolt/text_server_adv → ServerManager 或 Server create function。"
    ],
    questions: [
      "某个格式在哪里支持？查 gltf、fbx、jpg、webp、svg、ktx、dds、mp3、ogg 等模块。",
      "脚本语言如何接入？看 gdscript、mono 与 ScriptLanguage。",
      "模块何时初始化？看 register_types.h/cpp 和 initialization level。"
    ],
    pitfalls: [
      "不要手改生成出来的 register_module_types.gen.cpp；真正入口在各模块 register_types 和构建脚本。",
      "不要看到 thirdparty 库就从 thirdparty 开读；先看模块如何封装它。",
      "不要忘记构建开关；禁用模块后相关类、宏、导入器和后端都可能不存在。"
    ],
    read: "先看 modules/SCsub 和 modules_builders.py，再进入目标模块的 register_types 和 SCsub。"
  },
  {
    name: "platform",
    files: "818 tracked files / 349 C/C++/ObjC/Python files",
    role: "平台入口与 OS/DisplayServer 实现：Windows、LinuxBSD、macOS、Android、iOS、Web、visionOS。",
    boundary: "platform 负责把 Godot 的跨平台抽象接到具体操作系统。它处理进程入口、事件循环、窗口、输入、动态库、文件路径、平台导出 glue 等差异，但不定义 Object/Node/Server 的通用语义。",
    anchors: ["platform/windows/godot_windows.cpp:68", "platform/windows/os_windows.*", "platform/*/detect.py"],
    entry: "从当前平台的 <span class=\"source\">godot_*.cpp</span> 看进程入口，再看 <span class=\"source\">os_*</span> 和 <span class=\"source\">display_server_*</span>，最后看 <span class=\"source\">detect.py</span> 如何参与构建。",
    trail: [
      "启动：godot_*.cpp → OS_* 构造 → Main::setup → OS::run。",
      "窗口/输入：DisplayServer_* → platform 原生 API → InputEvent/Viewport。",
      "构建判断：platform/*/detect.py → SConstruct platform_list/platform_opts。"
    ],
    questions: [
      "程序从哪里进入 Main？看各平台 godot_*.cpp。",
      "窗口、输入、文件系统、动态库如何落到系统 API？看 OS_* 和 DisplayServer_*。",
      "构建如何判断平台可用？看 detect.py。"
    ],
    pitfalls: [
      "不要把某个平台实现当成所有平台行为；Windows、Web、Android、macOS 的事件循环和窗口模型差异很大。",
      "不要在 scene 层直接引入平台实现；跨平台语义应通过 OS、DisplayServer、Input、FileAccess 等抽象走。",
      "不要忽略导出模板条件；TOOLS_ENABLED 和 template 构建会改变平台代码路径。"
    ],
    read: "选当前平台入口，跟到 OS 子类，再回到 Main。"
  },
  {
    name: "editor",
    files: "1814 tracked files / 645 C/C++/ObjC/Python files",
    role: "编辑器主程序、插件、Inspector、导入导出、项目管理、调试、脚本编辑、主题和图标。",
    boundary: "editor 是工具构建中的 Godot 应用。它负责编辑器 UI、Dock、Inspector、插件、导入导出界面、项目管理和调试体验，但普通导出模板通常不会包含这些工具代码。",
    anchors: ["editor/editor_node.h:120", "editor/plugins/editor_plugin.h:59", "editor/inspector/editor_inspector.h:731"],
    entry: "先读 <span class=\"source\">editor/editor_node.h:120</span> 和对应 cpp，理解主界面如何组织；再读 <span class=\"source\">EditorPlugin</span>、<span class=\"source\">EditorInspector</span>、<span class=\"source\">FileSystemDock</span>。",
    trail: [
      "属性面板：选中对象 → EditorInspector → Object property list/ClassDB → EditorProperty 控件。",
      "插件工具：EditorPlugin 注册 → EditorNode 挂接 UI/菜单/编辑器工具。",
      "导入导出：editor/import 或 editor/export → ResourceImporter/ExportPlatform → modules/platform 支持。"
    ],
    questions: [
      "Inspector 为什么能动态显示属性？看 ClassDB + EditorInspector。",
      "新节点的编辑工具如何注册？看 EditorPlugin。",
      "导入和导出流程在哪里？看 editor/import、editor/export。"
    ],
    pitfalls: [
      "不要把编辑器和引擎运行时完全分开理解；编辑器大量复用 scene/gui、ResourceLoader、ClassDB。",
      "不要忘记 TOOLS_ENABLED；很多 editor 代码在导出模板中根本不存在。",
      "不要从 UI 控件直接推断底层行为；编辑器经常只是调用运行时 API 或资源导入器。"
    ],
    read: "先读 EditorNode 和 EditorPlugin，再按 Dock/Inspector/Import/Export 分支阅读。"
  },
  {
    name: "drivers",
    files: "236 tracked files / 182 C/C++/ObjC/Python files",
    role: "相对通用的驱动 glue：PNG、渲染驱动辅助、可访问性、音频/输入等底层适配。",
    boundary: "drivers 放相对平台无关或可复用的驱动适配层。它通常负责把某个库或底层能力注册到 Godot 的统一接口里，例如 ImageLoader、ResourceSaver、AccessibilityServer。",
    anchors: ["drivers/register_driver_types.cpp", "drivers/png"],
    entry: "先看 <span class=\"source\">drivers/register_driver_types.cpp</span>，确认驱动在哪个阶段注册，再进入具体驱动目录。",
    trail: [
      "PNG：drivers/png → ImageLoaderPNG/ResourceSaverPNG → ImageLoader/ResourceSaver。",
      "可访问性：ACCESSKIT_ENABLED → AccessibilityServerAccessKit → Display/GUI 可访问性更新。",
      "渲染/音频辅助：驱动 glue → Server 或 platform 具体实现。"
    ],
    questions: [
      "通用图片格式如何接入 ImageLoader？看 drivers/png 和 register_driver_types。",
      "平台无关驱动和平台相关实现如何分工？看 drivers 与 platform 的边界。"
    ],
    pitfalls: [
      "不要把 drivers 当成功能入口；用户 API 往往在 core/scene/servers，drivers 只是低层接入。",
      "不要把 drivers 和 thirdparty 混淆；drivers 是 Godot 封装代码，thirdparty 是外部库源码。",
      "不要忽略注册/注销对称性；加载器和 saver 注册后需要在 unregister 中移除。"
    ],
    read: "按具体驱动进入，确认它在哪个 register 阶段接入。"
  },
  {
    name: "main",
    files: "13 tracked files / 9 C/C++/ObjC/Python files",
    role: "跨平台启动、命令行解析、初始化、主循环、清理、性能统计和启动资源。",
    boundary: "main 是引擎生命周期总控，不是具体子系统实现。它决定系统初始化顺序、运行模式、主循环阶段和清理顺序，是所有端到端追踪最终要回来的时间轴。",
    anchors: ["main/main.cpp:1027", "main/main.cpp:3008", "main/main.cpp:3988", "main/main.cpp:4896"],
    entry: "按 <span class=\"source\">setup</span> → <span class=\"source\">setup2</span> → <span class=\"source\">start</span> → <span class=\"source\">iteration</span> → <span class=\"source\">cleanup</span> 的顺序读，不要直接从文件顶部读到文件底。",
    trail: [
      "初始化：register_core_types → initialize_modules(CORE) → register_server_types → register_scene_types → editor/module levels。",
      "运行模式：命令行参数 → editor/project_manager/script/main scene/export 分支。",
      "每帧：timer sync → physics loop → process → navigation → RenderingServer sync/draw → extension frame。"
    ],
    questions: [
      "引擎何时注册 core/server/scene/editor？看 setup/setup2。",
      "每帧顺序是什么？看 Main::iteration。",
      "编辑器、项目管理器、游戏主场景何时创建？看 Main::start。"
    ],
    pitfalls: [
      "不要把 Main::start 当唯一启动逻辑；大量单例和类型已经在 setup/setup2 中建立。",
      "不要忽略 cleanup 的反向顺序；关闭崩溃常常来自注销顺序或残留单例。",
      "不要把 fixed physics step 和 render frame 混为一谈；Main::iteration 中两者节奏不同。"
    ],
    read: "按 setup -> setup2 -> start -> iteration -> cleanup 顺序读。"
  },
  {
    name: "tests",
    files: "223 tracked files / 179 C/C++/ObjC/Python files",
    role: "单元测试和回归测试，覆盖 core、scene、servers、modules 等关键行为。",
    boundary: "tests 是行为证据层。它不定义生产逻辑，但能说明容器、Variant、Object、Resource、SceneTree、Server 抽象的预期行为，尤其适合验证底层改动有没有破坏契约。",
    anchors: ["tests/test_main.h", "tests/core", "tests/scene"],
    entry: "按修改影响面找测试目录：core 改动看 <span class=\"source\">tests/core</span>，scene 改动看 <span class=\"source\">tests/scene</span>，模块改动看对应模块测试或 doctest 注册。",
    trail: [
      "定位行为：先找同名类或同名目录测试，再回源码看实现。",
      "新增回归：复现 bug → 写最小测试 → 修改实现 → 跑相关测试。",
      "底层契约：Variant/Object/RID/Resource 等改动要优先查测试覆盖。"
    ],
    questions: [
      "某个底层容器或算法的预期行为是什么？先找 tests。",
      "修改 Object/Variant/Resource 后如何避免回归？跑相关测试。"
    ],
    pitfalls: [
      "不要用一个局部测试通过证明全局行为安全；需要匹配改动影响面。",
      "不要只看测试名；底层测试常覆盖边界条件、线程安全或序列化细节。",
      "不要忘记编辑器/工具代码可能没有普通运行时测试覆盖，需要额外手动路径或 doctool 验证。"
    ],
    read: "按你修改的子系统找对应测试目录。"
  },
  {
    name: "doc",
    files: "836 tracked files / 2 C/C++/ObjC/Python files",
    role: "类文档、导读材料和生成文档输入。当前导读页也放在 doc/study。",
    boundary: "doc 是用户 API 文档和学习材料，不是实现源码。它能告诉你某个类承诺给用户的行为，但真正行为仍要回到 _bind_methods、类实现和资源序列化。",
    anchors: ["doc/classes", "doc/study"],
    entry: "读 API 时先看 <span class=\"source\">doc/classes/ClassName.xml</span>，再回源码找同名类和 <span class=\"source\">_bind_methods()</span>；导读材料放在 <span class=\"source\">doc/study</span>。",
    trail: [
      "API 文档：doc/classes → _bind_methods → ClassDB registration。",
      "修改用户可见 API：源码绑定 → doc XML → doctool/文档生成检查。",
      "学习材料：doc/study → 对应源码锚点 → 实际实现验证。"
    ],
    questions: [
      "某个类暴露给用户的 API 文档怎么写？看 doc/classes。",
      "修改绑定 API 后文档如何同步？看 doctool 和 XML。"
    ],
    pitfalls: [
      "不要把文档当实现；文档可能滞后或省略底层细节。",
      "不要只改源码不改文档；用户可见 API 变化需要同步 doc/classes。",
      "不要在导读里引用不存在的源码锚点；文档解释要能回到当前源码树验证。"
    ],
    read: "读 API 时结合源码 _bind_methods 与 doc/classes。"
  },
  {
    name: "thirdparty",
    files: "4805 tracked files / 4489 C/C++/ObjC/Python files",
    role: "引擎 vendored 的外部依赖。通常不承载 Godot 业务逻辑，但影响编解码、字体、压缩、渲染和平台能力。",
    boundary: "thirdparty 是外部依赖源码仓库的本地副本。它提供算法、编解码、字体、压缩、图形或网络能力，但 Godot 的用户 API、注册逻辑、资源语义通常在 modules/drivers/core 的封装层。",
    anchors: ["thirdparty"],
    entry: "除非定位到外部库 bug，否则先从调用它的 <span class=\"source\">modules/*</span>、<span class=\"source\">drivers/*</span> 或 <span class=\"source\">core/*</span> 封装层进入。",
    trail: [
      "格式支持：模块/驱动封装 → thirdparty 库调用 → Godot Resource/Image/Audio API。",
      "升级依赖：thirdparty 变更 → 构建脚本 → 封装层 API 适配 → 测试和导入样例。",
      "安全/许可证：thirdparty 源码 → COPYRIGHT/LICENSE 记录 → 模块使用范围。"
    ],
    questions: [
      "一个功能是 Godot 自己写的还是外部库？看 modules 和 thirdparty 的边界。",
      "升级依赖会影响哪些模块？从模块 register 和构建脚本反查。"
    ],
    pitfalls: [
      "不要先读 thirdparty 来理解 Godot 行为；先读 Godot 封装层更快。",
      "不要随意重排或格式化 thirdparty；它通常需要尽量接近上游，减少升级冲突。",
      "不要忽略许可证和补丁记录；第三方库变更可能影响分发合规。"
    ],
    read: "除非调外部库 bug，否则先读 Godot 封装层。"
  }

];

const modules = [
  ["astcenc", "纹理/压缩", "ASTC 纹理压缩编码支持，面向移动和现代 GPU 纹理格式。", "thirdparty astcenc"],
  ["basis_universal", "纹理/压缩", "Basis Universal 超压缩纹理支持，服务导入和运行时纹理处理。", "basis/ktx 相关"],
  ["bcdec", "纹理/压缩", "BC/DXT 纹理解码工具，常用于压缩纹理读取。", "轻量解码"],
  ["betsy", "渲染/纹理", "GPU 纹理压缩相关辅助模块，用于特定纹理处理路径。", "渲染资产管线"],
  ["bmp", "图片格式", "BMP 图片加载支持。", "Resource/ImageLoader"],
  ["camera", "设备", "摄像头输入模块，配合 CameraServer 暴露设备流。", "servers/camera"],
  ["csg", "3D/建模", "Constructive Solid Geometry 节点，提供 CSG 形状布尔建模。", "scene 3D 节点"],
  ["cvtt", "纹理/压缩", "Convection Texture Tools，用于纹理压缩/转换。", "导入管线"],
  ["dds", "图片格式", "DDS 纹理文件读取，常见于压缩纹理资产。", "ImageLoader"],
  ["enet", "网络", "ENet 可靠 UDP 网络库，服务 multiplayer 传输。", "Multiplayer"],
  ["etcpak", "纹理/压缩", "ETC/EAC 纹理压缩工具。", "移动纹理"],
  ["fbx", "导入", "FBX 场景/模型导入支持。", "editor/import"],
  ["freetype", "字体/文本", "FreeType 字体栅格化支持，是字体系统的重要底层依赖。", "Text/Font"],
  ["gdscript", "脚本", "Godot 原生脚本语言：解析器、分析器、编译器、VM、LSP、编辑器集成。", "ScriptLanguage"],
  ["glslang", "渲染/shader", "GLSL/SPIR-V 编译相关支持，服务 shader 管线。", "RenderingDevice"],
  ["gltf", "导入/导出", "glTF 2.0 场景、模型、动画、材质导入导出。", "editor/import/export"],
  ["godot_physics_2d", "物理", "Godot 内置 2D 物理实现。", "PhysicsServer2D"],
  ["godot_physics_3d", "物理", "Godot 内置 3D 物理实现。", "PhysicsServer3D"],
  ["gridmap", "3D/场景", "GridMap 节点和网格化 3D 关卡编辑支持。", "scene/editor"],
  ["hdr", "图片格式", "Radiance HDR 图像加载支持。", "ImageLoader"],
  ["interactive_music", "音频", "交互式音乐播放/编排相关能力。", "Audio"],
  ["jolt_physics", "物理", "Jolt Physics 3D 后端接入，可作为 3D 物理实现。", "PhysicsServer3D"],
  ["jpg", "图片格式", "JPEG 图片加载支持。", "ImageLoader"],
  ["jsonrpc", "网络/工具", "JSON-RPC 协议支持，常用于工具通信。", "Debugger/Editor"],
  ["ktx", "纹理/压缩", "KTX 纹理容器支持。", "纹理导入"],
  ["lightmapper_rd", "渲染/烘焙", "基于 RenderingDevice 的光照烘焙模块。", "Editor/Rendering"],
  ["mbedtls", "网络/安全", "TLS/加密相关支持，服务 HTTPS、WebSocket、安全网络。", "core/crypto/io"],
  ["meshoptimizer", "导入/3D", "Mesh 优化、压缩和 glTF meshopt 支持。", "Import"],
  ["mobile_vr", "XR", "移动 VR 支持模块。", "XRServer"],
  ["mono", "脚本", "C#/.NET 支持模块，工具构建中还包含绑定生成。", "ScriptLanguage"],
  ["mp3", "音频格式", "MP3 音频解码和导入/播放支持。", "AudioStream"],
  ["msdfgen", "字体/文本", "MSDF 字体生成支持，提升缩放文字/图标渲染质量。", "Font"],
  ["multiplayer", "网络", "高级 Multiplayer API、RPC 等网络游戏能力。", "SceneTree/Node"],
  ["navigation_2d", "导航", "2D 导航服务器实现、导航网格和路径查询。", "NavigationServer2D"],
  ["navigation_3d", "导航", "3D 导航服务器实现、导航网格和 agent。", "NavigationServer3D"],
  ["noise", "工具/资源", "噪声生成资源，服务程序化纹理、地形和效果。", "Resource"],
  ["objectdb_profiler", "调试", "ObjectDB 分析和对象生命周期调试工具。", "Debugger"],
  ["ogg", "音频容器", "Ogg 容器支持，通常配合 Vorbis/Theora。", "Audio/Video"],
  ["openxr", "XR", "OpenXR 接入，面向 VR/AR 设备。", "XRServer"],
  ["raycast", "工具/物理", "Raycast 相关辅助模块。", "Physics/Utility"],
  ["regex", "文本", "正则表达式支持。", "core/string"],
  ["svg", "图片格式", "SVG 矢量图加载/栅格化支持。", "ImageLoader"],
  ["text_server_adv", "字体/文本", "高级文本服务后端，处理复杂脚本 shaping。", "TextServer"],
  ["text_server_fb", "字体/文本", "Fallback 文本服务后端，提供基础文本能力。", "TextServer"],
  ["tga", "图片格式", "TGA 图片加载支持。", "ImageLoader"],
  ["theora", "视频/音频", "Theora 视频支持，常和 Ogg 容器结合。", "VideoStream"],
  ["tinyexr", "图片格式", "OpenEXR/HDR 图像支持。", "ImageLoader"],
  ["upnp", "网络", "UPnP 端口映射等网络辅助能力。", "Networking"],
  ["vhacd", "物理/导入", "凸分解工具，用于把复杂 mesh 转为物理碰撞形状。", "Import/Physics"],
  ["visual_shader", "渲染/shader", "可视化 Shader 编辑和资源支持。", "Editor/Shader"],
  ["vorbis", "音频格式", "Vorbis 音频解码支持。", "AudioStream"],
  ["webp", "图片格式", "WebP 图片加载支持。", "ImageLoader"],
  ["webrtc", "网络", "WebRTC 通信支持。", "Multiplayer/Web"],
  ["websocket", "网络", "WebSocket 客户端/服务器支持。", "Networking"],
  ["webxr", "XR/Web", "Web 平台 XR 支持。", "platform/web"],
  ["xatlas_unwrap", "3D/导入", "UV unwrap 支持，常用于光照贴图和导入处理。", "Import/Lightmap"],
  ["zip", "IO/压缩", "ZIP/PCK 相关压缩与归档支持。", "Resource/Export"]
];

const moduleAdvice = {
  "脚本": "先看 ScriptLanguage 接入，再看资源类型、实例桥、编译/运行时和编辑器集成。",
  "物理": "先看 register_types 如何接入 PhysicsServer，再看 body/space/shape/joint 的后端实现。",
  "导航": "先看 NavigationServer 接口，再看 map、region、agent、query 的运行时数据结构。",
  "导入": "先看 editor/import 或 ResourceImporter，再看格式解析器，最后看 thirdparty 解析库。",
  "导入/导出": "先看 ResourceFormatLoader/Importer/Exporter，再看 editor/export 和平台打包分支。",
  "导入/3D": "先看 glTF/FBX/mesh 资源转换路径，再看 Mesh、Animation、Material、Skeleton 如何生成。",
  "3D/导入": "先看导入器生成的 Mesh/UV/Lightmap 数据，再看 editor 工具如何调用。",
  "图片格式": "先看 ImageLoader 或 ResourceFormatLoader 注册，再看解码器和 Image 数据写入。",
  "纹理/压缩": "先看导入管线如何选择压缩格式，再看编码器、KTX/Basis/ASTC/BC 数据路径。",
  "渲染/纹理": "先看资源或导入阶段，再确认是否最终提交给 RenderingServer/RenderingDevice。",
  "渲染/shader": "先看 Shader/RenderingServer 抽象，再看 glslang、SPIR-V、RD shader 编译路径。",
  "渲染/烘焙": "先看编辑器触发入口，再看 RenderingDevice 计算任务和输出资源。",
  "字体/文本": "先看 TextServer 后端，再看 Font/File loading、shaping、glyph cache 和 fallback。",
  "文本": "先看 core string API 或 Regex wrapper，再确认 thirdparty pcre2 边界。",
  "网络": "先看高层 API 是 Multiplayer、PacketPeer 还是 HTTP/WebSocket，再跟到模块协议实现。",
  "网络/安全": "先看 StreamPeerTLS/Crypto 接口，再看 mbedTLS 包装层和证书加载。",
  "网络/工具": "先看工具侧调用者，例如 debugger、language server 或 editor，然后看协议封装。",
  "音频": "先看 AudioStream/AudioServer，再看模块资源如何产生 playback。",
  "音频格式": "先看 AudioStream 资源和 importer，再看解码器如何向 AudioServer 提供采样数据。",
  "音频容器": "先区分容器和编码，Ogg 通常还要跟 Vorbis/Theora。",
  "视频/音频": "先看 VideoStream 资源，再看容器、解码和播放同步。",
  "XR": "先看 XRServer 和 XRInterface，再看平台 runtime、action map、render target 和 tracking。",
  "XR/Web": "先看 WebXR 与 platform/web 的 JS bridge，再看 XRServer 接口。",
  "设备": "先看对应 Server 单例，再看平台实现如何枚举设备和推送数据。",
  "3D/建模": "先看节点/资源 API，再看编辑器 gizmo 和生成 mesh 的算法边界。",
  "3D/场景": "先看 scene 节点，再看 editor 插件和 Resource/MeshLibrary 数据。",
  "工具/资源": "先看 Resource 类和 Inspector/EditorPlugin，再看运行时是否参与。",
  "工具/物理": "先看它是 editor 辅助还是运行时查询工具，再接 PhysicsServer。",
  "物理/导入": "先看导入器如何生成碰撞资源，再看后端是否只在运行时消费。",
  "调试": "先看 EngineDebugger/EditorDebugger 接口，再看采样、消息和 UI 展示。",
  "IO/压缩": "先看 FileAccess/ResourceLoader/Export 打包路径，再看压缩库边界。"
};

const concepts = [
  {
    id: "startup-axis",
    title: "启动轴",
    aliases: ["启动轴", "启动链路轴", "启动时间轴", "Main 生命周期轴", "Main 启动链"],
    summary: "从平台入口到 Main::setup/setup2/start/iteration/cleanup 的时间线，用来解释 Godot 怎样把一个进程变成正在跑的 MainLoop。启动轴回答“谁先初始化、每帧谁调度、退出时谁先清理”。",
    article: [
      {
        type: "lead",
        text: "启动轴不是一个目录名，而是一条时间线：平台入口创建 OS 子类，`Main::setup()` 建立 core 和项目配置，`Main::setup2()` 注册 servers/scene 并创建运行时服务，`Main::start()` 选择或创建 MainLoop，平台 `OS::run()` 进入循环，每帧回到 `Main::iteration()`，退出时 `Main::cleanup()` 按相反方向释放系统。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "把 Godot 启动想成开机流程。平台代码先把程序门打开，然后 Main 负责装好对象系统、资源系统、渲染和物理服务，再决定这次是开编辑器、跑项目还是执行命令行工具。之后每一帧都从同一个主循环入口推进。"
      },
      {
        type: "paragraph",
        text: "读源码时，启动轴最重要的不是背函数名，而是知道“现在处在哪个阶段”。很多对象能不能创建、某个 Server 是否存在、脚本语言是否初始化、主场景有没有加载，都取决于它发生在 setup、setup2、start、iteration 还是 cleanup。"
      },
      {
        type: "flow",
        title: "启动轴的主干流程",
        steps: [
          { title: "平台入口", text: "`platform/windows/godot_windows.cpp:68` 和 `platform/linuxbsd/godot_linuxbsd.cpp:70` 创建 OS 子类并调用 Main。" },
          { title: "`Main::setup()`", text: "`main/main.cpp:1027` 初始化 OS、Engine、core 类型、ProjectSettings、MessageQueue 和 CORE 级模块。" },
          { title: "`Main::setup2()`", text: "`main/main.cpp:3008` 注册 Server 和 Scene 类型，初始化 DisplayServer、RenderingServer、AudioServer 等运行时基础。" },
          { title: "`Main::start()`", text: "`main/main.cpp:3988` 选择 MainLoop，默认是 SceneTree，并在项目模式加载主场景。" },
          { title: "`OS::run()`", text: "`platform/windows/os_windows.cpp:2343` 或 LinuxBSD 对应实现调用 `main_loop->initialize()` 并循环处理事件。" },
          { title: "`Main::iteration()`", text: "`main/main.cpp:4896` 每帧推进物理、process、消息队列、导航、RenderingServer sync/draw。" },
          { title: "`Main::cleanup()`", text: "`main/main.cpp:5191` 停脚本、清 Scene、卸 Server、删 MessageQueue、注销 core。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "平台入口非常薄。Windows 的 `widechar_main()` 在 `platform/windows/godot_windows.cpp:68` 构造 `OS_Windows`，把参数转成 UTF-8，调用 `Main::setup()` 和 `Main::start()`，成功后执行 `os.run()`，最后 `Main::cleanup()`。LinuxBSD 的 `main()` 在 `platform/linuxbsd/godot_linuxbsd.cpp:70` 做同样分工。也就是说，跨平台差异先被 OS 子类吸收，真正的引擎启动顺序集中到 Main。"
      },
      {
        type: "paragraph",
        text: "`Main::setup()` 的源码入口是 `main/main.cpp:1027`。它先调用 `OS::get_singleton()->initialize()`，再创建 `Engine`，执行 `register_core_types()`，建立 `InputMap`、`ProjectSettings`、`TranslationServer`、`Performance` 等 core 层单例，并解析命令行。`main/main.cpp:2255` 初始化 CORE 级模块，`main/main.cpp:2899` 创建主 `MessageQueue`。"
      },
      {
        type: "paragraph",
        text: "`Main::setup2()` 在 `main/main.cpp:3008` 进入第二阶段。它在 `main/main.cpp:3197` 调 `register_server_types()`，随后初始化 SERVERS 级模块；在后续 Display/Rendering/Audio 初始化块里创建具体服务，例如 `main/main.cpp:3540` 创建 `RenderingServerDefault` 并调用 `init()`；`main/main.cpp:3759` 调 `register_scene_types()`；`main/main.cpp:3839` 再把 Server 单例注册到 Engine。"
      },
      {
        type: "paragraph",
        text: "`Main::start()` 的入口是 `main/main.cpp:3988`。它不是最早的初始化点，而是“开始运行哪种 MainLoop”的决策点。默认情况下，`main/main.cpp:4337` 会创建 `SceneTree`；如果传入脚本或自定义 MainLoop 类型，它会通过 `ResourceLoader` 和 `ClassDB::instantiate()` 创建对应对象；`main/main.cpp:4411` 把 MainLoop 交给 OS。项目主场景加载发生在 `main/main.cpp:4737` 附近：`ResourceLoader::load()` 得到 `PackedScene`，再 `instantiate()` 成 Node 并加入 SceneTree。"
      },
      {
        type: "paragraph",
        text: "真正的帧循环由平台 `OS::run()` 驱动。Windows 的 `OS_Windows::run()` 在 `platform/windows/os_windows.cpp:2343` 调 `main_loop->initialize()`，循环 `DisplayServer::process_events()` 和 `Main::iteration()`，结束后 `main_loop->finalize()`。`Main::iteration()` 在 `main/main.cpp:4896` 做固定物理步、`physics_process()`、Server sync/step、空闲 `process()`、MessageQueue flush、导航更新以及 `RenderingServer::sync()` 和 `draw()`。"
      },
      {
        type: "table",
        title: "启动轴的证据锚点",
        headers: ["阶段", "源码入口", "说明"],
        rows: [
          ["平台入口", "`platform/windows/godot_windows.cpp:68`、`platform/linuxbsd/godot_linuxbsd.cpp:70`", "创建平台 OS，调用 Main。"],
          ["core setup", "`main/main.cpp:1027`、`main/main.cpp:1059`、`core/register_core_types.cpp:134`", "注册 Object、RefCounted、Resource、Variant 等基础类型。"],
          ["server/scene setup", "`main/main.cpp:3008`、`3197`、`3759`", "注册 Server 与 Scene 类型，初始化模块。"],
          ["MainLoop 选择", "`main/main.cpp:3988`、`4337`、`4411`", "创建或加载 MainLoop，默认 SceneTree。"],
          ["主场景加载", "`main/main.cpp:4737`、`scene/resources/packed_scene.cpp:2507`", "ResourceLoader 加载 PackedScene，再实例化 Node。"],
          ["每帧推进", "`main/main.cpp:4896`、`5004`、`5037`、`5052`", "物理、process、队列、Server sync/draw。"],
          ["清理", "`main/main.cpp:5191`、`5261`、`5289`、`5354`、`5371`", "按 editor/scene/servers/core 反向释放。"]
        ]
      },
      {
        type: "heading",
        title: "案例：项目主场景从参数到运行"
      },
      {
        type: "flow",
        title: "主场景路径进入 SceneTree",
        steps: [
          { title: "命令行或 ProjectSettings 给出场景路径", text: "`--scene` 或 `application/run/main_scene` 在启动阶段被解析。" },
          { title: "`Main::start()` 进入项目路径", text: "`main/main.cpp:4737` 附近用 `ResourceLoader::load(local_game_path)` 加载。" },
          { title: "得到 `PackedScene`", text: "加载结果不是直接运行中的树，而是可实例化的资源。" },
          { title: "`PackedScene::instantiate()`", text: "`scene/resources/packed_scene.cpp:2507` 调 `SceneState::instantiate()` 创建 Node 树。" },
          { title: "`SceneTree::add_current_scene()`", text: "主场景挂到 root 下，后续由 SceneTree 的 process/physics 调度。" }
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：以为 `Main::start()` 才是所有初始化的开始。实际 Object、Variant、ProjectSettings、MessageQueue、部分模块和 Server 类型已经在 setup/setup2 建立。",
          "误区二：把平台入口当成引擎主逻辑。平台代码主要准备 OS 和事件循环，跨平台启动顺序集中在 `main/main.cpp`。",
          "误区三：把一帧等同于一次 `_process()`。`Main::iteration()` 还包含固定物理步、Server 同步、消息队列、导航、渲染提交和性能采样。",
          "误区四：忽略 cleanup 顺序。很多关闭期崩溃来自某个对象晚于它依赖的 Server 或 ClassDB 被释放。"
        ]
      },
      {
        type: "heading",
        title: "边界和相邻概念"
      },
      {
        type: "table",
        headers: ["概念", "启动轴怎么看它", "不属于启动轴的部分"],
        rows: [
          ["对象轴", "启动轴决定 Object/ClassDB/Variant 什么时候注册可用。", "具体 set/get/call、信号和绑定规则属于对象轴。"],
          ["场景轴", "启动轴决定何时创建 SceneTree、加载主场景、进入每帧。", "Node 父子关系、ready/process 传播属于场景轴。"],
          ["服务轴", "启动轴决定 RenderingServer、PhysicsServer、AudioServer 等什么时候创建、sync、draw、清理。", "Server API 和后端实现属于服务轴。"],
          ["模块系统", "模块按 CORE/SERVERS/SCENE/EDITOR 插入启动时间线。", "单个模块的业务实现不等于启动轴本身。"]
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读当前平台入口，例如 `platform/windows/godot_windows.cpp:68`。",
          "按 `Main::setup()`、`setup2()`、`start()`、`iteration()`、`cleanup()` 读 `main/main.cpp`，不要从文件顶部漫游到底。",
          "在每个阶段记录它注册、创建或清理了哪些全局对象和 Server。",
          "遇到具体问题时回到对应阶段：类型不存在看 setup/setup2，主场景加载看 start，每帧顺序看 iteration，退出崩溃看 cleanup。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "启动轴是 Godot 的时间坐标：它解释对象系统、场景树和服务后端分别在什么时候出现、每帧如何被推进、退出时按什么顺序撤场。"
      }
    ]
  },
  {
    id: "object-axis",
    title: "对象轴",
    aliases: ["对象轴", "对象模型轴", "类型反射轴", "Object/ClassDB/Variant 轴"],
    summary: "围绕 Object、ClassDB、MethodBind、Variant、ObjectDB 和 ScriptInstance 的读法，用来解释 C++ 类怎样变成脚本、编辑器、Inspector、资源系统和扩展都能认识的运行时对象。",
    article: [
      {
        type: "lead",
        text: "对象轴回答的是“Godot 怎样认识一个东西”。一个类只写成 C++ 还不够；它要继承 Object 或相关基类，用 `GDCLASS` 接入类型信息，在 `_bind_methods()` 里通过 ClassDB 暴露方法、属性和信号，运行时实例还会获得 ObjectID，并通过 Variant 作为统一值边界和脚本、编辑器、序列化、GDExtension 交换数据。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "对象轴像 Godot 的“身份证和办事窗口”。Object 给实例身份，ClassDB 是类登记表，MethodBind 是方法转接头，Variant 是通用参数盒子，ObjectDB 是按 ObjectID 查活对象的弱索引。"
      },
      {
        type: "paragraph",
        text: "所以当你在 GDScript 写 `node.call(\"set_name\", \"Player\")`，或者 Inspector 显示某个属性时，背后不是编辑器硬编码每个类，而是对象轴把 C++ 类型、方法名、属性元数据、参数和对象实例串起来。"
      },
      {
        type: "flow",
        title: "对象轴的核心链路",
        steps: [
          { title: "C++ 类继承 Object", text: "`scene/main/node.h:54` 的 Node 就是 `public Object`。" },
          { title: "`GDCLASS` 建立类型入口", text: "`core/object/object.h:248` 定义宏，子类用它接入 Godot 类型系统。" },
          { title: "`_bind_methods()` 暴露 API", text: "`ClassDB::bind_method()` 在 `core/object/class_db.h:374` 生成 MethodBind。" },
          { title: "ClassDB 保存类级元数据", text: "`core/object/class_db.h:97` 的 ClassInfo 保存 method_map、property_list、signal 等。" },
          { title: "Object 实例处理动态访问", text: "`core/object/object.cpp:233`、`316`、`768` 分别处理 set/get/callp。" },
          { title: "Variant 统一参数和返回值", text: "`core/variant/variant.h:93` 定义统一值容器。" },
          { title: "ObjectDB 提供弱查找", text: "`core/object/object.cpp:2450` 在对象构造时分配 ObjectID。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "对象轴在启动阶段由 `register_core_types()` 建立。`core/register_core_types.cpp:134` 调 `ObjectDB::setup()`、`StringName::setup()`，随后 `GDREGISTER_CLASS(Object)`、`GDREGISTER_CLASS(RefCounted)`、`GDREGISTER_CLASS(Resource)`，并在 `core/register_core_types.cpp:155` 附近调用 `Variant::register_types()`。这说明对象轴不是 scene 层才有，而是 core 层地基。"
      },
      {
        type: "paragraph",
        text: "Object 本体的源码入口是 `core/object/object.h:349`。它保存 `_instance_id`、脚本实例、metadata、信号表、连接列表、删除标志和类型缓存。构造路径在 `core/object/object.cpp:2283`，其中 `core/object/object.cpp:2294` 通过 `ObjectDB::add_instance(this)` 分配 ObjectID；析构时会断开信号并从 ObjectDB 移除。"
      },
      {
        type: "paragraph",
        text: "ClassDB 是类级登记表，不是实例容器。`core/object/class_db.h:97` 的 `ClassInfo` 保存 `method_map`、`property_list`、`property_setget` 等元数据。`ClassDB::bind_method()` 模板在 `core/object/class_db.h:374` 创建 MethodBind，`ClassDB::bind_methodfi()` 在 `core/object/class_db.cpp:1895` 把 MethodBind 写进当前类的 method_map，并拒绝普通脚本 API 的同名重载。"
      },
      {
        type: "paragraph",
        text: "动态调用从 Object 回到 ClassDB。`Object::callp()` 在 `core/object/object.cpp:768`，先尝试脚本实例处理；如果脚本没有这个方法，再用 `ClassDB::get_method(get_class_name(), p_method)` 找 MethodBind 并调用。`Object::set()` 和 `Object::get()` 在 `core/object/object.cpp:233`、`316`，同样会合并脚本、GDExtension、ClassDB 绑定属性和 metadata 等路径。"
      },
      {
        type: "paragraph",
        text: "Variant 是对象轴的值边界，不是对象所有权的替代品。`core/variant/variant.h:93` 定义 `Variant`，它用固定枚举类型承载 int、String、Object、RID、Array、Dictionary 等值。Object 进入 Variant 时通常保存 ObjectID 和指针；普通 Object 不会因此被强拥有，RefCounted 才会走引用计数语义。"
      },
      {
        type: "table",
        title: "对象轴的证据锚点",
        headers: ["部件", "源码入口", "职责"],
        rows: [
          ["Object", "`core/object/object.h:349`、`core/object/object.cpp:2283`", "实例身份、脚本实例、信号、metadata、set/get/call。"],
          ["ObjectDB", "`core/object/object.h:874`、`core/object/object.cpp:2450`", "给活对象分配 ObjectID，支持弱查找。"],
          ["ClassDB", "`core/object/class_db.h:97`、`core/object/class_db.cpp:1895`", "保存类名、继承、方法、属性、信号和创建函数。"],
          ["MethodBind", "`core/object/method_bind.h:38`、`core/object/class_db.h:374`", "把 C++ 成员函数包装成动态可调用对象。"],
          ["Variant", "`core/variant/variant.h:93`", "脚本、反射、属性、信号和序列化的通用值容器。"],
          ["核心注册", "`core/register_core_types.cpp:134`", "启动时建立 Object、RefCounted、Resource、Variant 等基础类型。"]
        ]
      },
      {
        type: "heading",
        title: "案例：脚本调用一个 C++ 方法"
      },
      {
        type: "flow",
        title: "`node.set_name(\"Player\")` 的对象轴路径",
        steps: [
          { title: "Node 是 Object 子类", text: "`scene/main/node.h:54` 让 Node 继承 Object。" },
          { title: "Node 绑定方法", text: "Node 的 `_bind_methods()` 把 `set_name` 等方法登记到 ClassDB。" },
          { title: "脚本传入 Variant 参数", text: "字符串参数以 Variant 形式穿过动态调用层。" },
          { title: "`Object::callp()` 查方法", text: "`core/object/object.cpp:768` 先查脚本，再查 ClassDB。" },
          { title: "MethodBind 调真实 C++ 函数", text: "绑定层拆出参数并调用 Node 的 C++ 成员函数。" }
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：对象轴等于场景轴。Object 只是运行时对象根，Node 才是能进入 SceneTree 的对象。",
          "误区二：ClassDB 保存对象实例。ClassDB 保存类级元数据；实例身份和弱查找看 Object/ObjectDB。",
          "误区三：ObjectID 是所有权。ObjectID 只是弱句柄，不能阻止普通 Object 被释放。",
          "误区四：Variant 让所有值都安全长活。Variant 统一传值，但普通 Object 指针仍要验证是否还活着。",
          "误区五：Inspector 是编辑器手写属性 UI。Inspector 主要读 Object/ClassDB/ScriptInstance 暴露出的属性表。"
        ]
      },
      {
        type: "heading",
        title: "边界和相邻概念"
      },
      {
        type: "table",
        headers: ["概念", "对象轴负责", "不负责"],
        rows: [
          ["启动轴", "启动轴让 Object/ClassDB/Variant 先注册可用。", "对象轴不决定 setup/start/cleanup 的时间顺序。"],
          ["场景轴", "场景轴使用 Object 子类 Node，并依赖 ClassDB 实例化节点。", "对象轴本身没有父子树、ready、process 或场景切换。"],
          ["服务轴", "Server 单例也常是 Object，可被 ClassDB 暴露给脚本。", "对象轴不保存 GPU、物理世界或音频混音状态。"],
          ["资源系统", "Resource 是 RefCounted/Object，能参与属性和 Variant。", "Object 本身不等于可保存文件；保存加载由 ResourceLoader/ResourceSaver 处理。"]
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "读 `core/register_core_types.cpp:134`，确认对象基础类型在哪个启动阶段注册。",
          "读 `core/object/object.h:349` 和 `object.cpp:2283`，理解 Object 保存的实例状态和 ObjectID。",
          "读 `core/object/class_db.h:97`、`374` 和 `class_db.cpp:1895`，看类元数据如何登记。",
          "读 `core/object/object.cpp:233`、`316`、`768`，跟一次 set/get/call 的真实分发。",
          "最后读 `core/variant/variant.h:93` 和 MethodBind，区分动态调用的值边界和函数包装。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "对象轴是 Godot 的运行时识别系统：它把 C++ 类、对象实例、脚本调用、编辑器属性、信号和 Variant 参数统一到一套可反射的对象世界里。"
      }
    ]
  },
  {
    id: "scene-axis",
    title: "场景轴",
    aliases: ["场景轴", "场景生命周期轴", "SceneTree 轴", "Node 树轴", "PackedScene 到 SceneTree 轴"],
    summary: "从 PackedScene/SceneState 到 Node 树，再到 SceneTree 的 enter_tree、ready、process、physics_process、场景切换和删除队列。场景轴解释用户可见对象怎样被实例化并按帧调度。",
    article: [
      {
        type: "lead",
        text: "场景轴回答的是“一个场景怎样从文件变成正在跑的节点树”。`.tscn/.scn` 先作为 PackedScene/SceneState 被加载，`instantiate()` 创建 Node 树，Node 进入 SceneTree 后触发 enter_tree/ready，之后由 SceneTree 在每帧调用 physics/process、处理组、定时器、Tween、场景切换和删除队列。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "PackedScene 像蓝图，Node 像蓝图造出来的零件，SceneTree 像现场调度员。蓝图加载完成后还没有“活着”；只有实例化成 Node，并挂进 SceneTree，才会收到 `_enter_tree()`、`_ready()`、`_process()` 这些生命周期回调。"
      },
      {
        type: "paragraph",
        text: "读场景轴时要一直区分三件事：文件里的场景资源、运行中的 Node 实例、每帧调度这些实例的 SceneTree。很多 bug 正是把这三者混在一起造成的。"
      },
      {
        type: "flow",
        title: "场景轴的主干流程",
        steps: [
          { title: "资源加载", text: "`ResourceLoader::load()` 读 `.tscn/.scn`，返回 PackedScene。" },
          { title: "PackedScene 持有 SceneState", text: "`scene/resources/packed_scene.h:246` 的 PackedScene 内部有 `Ref<SceneState> state`。" },
          { title: "SceneState 记录节点表", text: "`scene/resources/packed_scene.h:38` 保存节点、属性、连接、组和路径。" },
          { title: "`instantiate()` 创建 Node", text: "`scene/resources/packed_scene.cpp:155` 按 NodeData 循环，`318` 通过 ClassDB 实例化节点类型。" },
          { title: "Node 进入树", text: "`scene/main/node.cpp:341` 传播 enter_tree，设置 tree、viewport、组并发通知。" },
          { title: "ready 和每帧处理", text: "`scene/main/node.cpp:323` 传播 ready；`scene/main/scene_tree.cpp:639` 和 `688` 推进 physics/process。" },
          { title: "删除和场景切换", text: "`scene/main/scene_tree.cpp:1637` 处理 queue_delete，`1702` 到 `1721` 处理 change_scene。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "场景轴的类型在 `register_scene_types()` 中注册。`scene/register_scene_types.cpp:390` 先创建 SceneStringNames、初始化 Node 层级缓存和资源格式加载器，然后注册 `Node`、`CanvasItem`、`Viewport`、`Window`、`Control`、GUI、2D/3D、动画和资源类型。这些类型注册后，PackedScene 实例化时才能通过 ClassDB 创建具体节点。"
      },
      {
        type: "paragraph",
        text: "PackedScene 是 Resource，不是运行节点。源码入口是 `scene/resources/packed_scene.h:246`；它保存 `Ref<SceneState> state`。SceneState 在 `scene/resources/packed_scene.h:38`，里面有 `names`、`variants`、`node_paths`、`nodes`、`connections` 等压缩表。也就是说，场景文件进内存后先变成一套可实例化的数据表。"
      },
      {
        type: "paragraph",
        text: "`SceneState::instantiate()` 在 `scene/resources/packed_scene.cpp:155`。它遍历保存的 `NodeData`，处理继承场景、子场景实例、placeholder、缺失节点，普通节点则在 `scene/resources/packed_scene.cpp:318` 调 `ClassDB::instantiate(snames[n.type])` 创建 Object，再 cast 成 Node。后续代码恢复属性、父子关系、组、信号连接和本地资源。"
      },
      {
        type: "paragraph",
        text: "Node 的源码入口是 `scene/main/node.h:54`。它是 Object 子类，但多了父子关系、owner、组、路径、process 模式、线程组、viewport、tree 指针和生命周期通知。`Node::_propagate_enter_tree()` 在 `scene/main/node.cpp:341`，会继承 tree/viewport、加入组、发送 `NOTIFICATION_ENTER_TREE` 和 `tree_entered`，然后递归子节点；`Node::_propagate_ready()` 在 `scene/main/node.cpp:323`，先递归子节点，再发 ready。"
      },
      {
        type: "paragraph",
        text: "SceneTree 是默认 MainLoop，源码入口是 `scene/main/scene_tree.h:89`。`SceneTree::initialize()` 在 `scene/main/scene_tree.cpp:586` 把 root 节点接到树上。`SceneTree::physics_process()` 在 `scene/main/scene_tree.cpp:639`，会发 `physics_frame`、处理 picking、调用 `_process(true)`、刷新消息队列、timer/tween 和删除队列。`SceneTree::process()` 在 `scene/main/scene_tree.cpp:688`，会 poll multiplayer、发 `process_frame`、调用 `_process(false)`、flush 场景切换和删除。"
      },
      {
        type: "table",
        title: "场景轴的证据锚点",
        headers: ["部件", "源码入口", "职责"],
        rows: [
          ["Scene 类型注册", "`scene/register_scene_types.cpp:390`", "注册 Node、Viewport、Control、PackedScene 相关类型和资源格式。"],
          ["PackedScene", "`scene/resources/packed_scene.h:246`、`packed_scene.cpp:2507`", "场景资源壳，调用 SceneState 实例化。"],
          ["SceneState", "`scene/resources/packed_scene.h:38`、`packed_scene.cpp:155`", "保存节点、属性、连接和路径表，并负责实例化。"],
          ["Node", "`scene/main/node.h:54`、`node.cpp:341`、`323`", "父子树、owner、组、enter_tree、ready、通知。"],
          ["SceneTree", "`scene/main/scene_tree.h:89`、`scene_tree.cpp:586`、`639`、`688`", "默认 MainLoop，负责节点调度和每帧生命周期。"],
          ["删除和切场景", "`scene/main/node.cpp:3461`、`scene_tree.cpp:1637`、`1702`", "queue_free、delete_queue 和 change_scene。"]
        ]
      },
      {
        type: "heading",
        title: "案例：`change_scene_to_file()`"
      },
      {
        type: "flow",
        title: "运行时切换场景",
        steps: [
          { title: "脚本调用切场景", text: "用户调用 `SceneTree.change_scene_to_file(path)`。" },
          { title: "ResourceLoader 加载 PackedScene", text: "`scene/main/scene_tree.cpp:1702` 调 `ResourceLoader::load(p_path)`。" },
          { title: "PackedScene 实例化 Node", text: "`scene/main/scene_tree.cpp:1712` 调 `change_scene_to_packed()`，再 `p_scene->instantiate()`。" },
          { title: "挂接到 root", text: "`_flush_scene_change()` 把 pending_new_scene 加到 root 下。" },
          { title: "进入生命周期", text: "add_child 触发 enter_tree/ready，后续由 process/physics 推进。" }
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：PackedScene 就是运行中的场景。PackedScene 是 Resource，只有 `instantiate()` 后才产生 Node 实例。",
          "误区二：Node 等于 Resource。Node 是树上的运行对象；Resource 是可加载、可共享、可保存的数据对象。",
          "误区三：SceneTree 负责渲染或物理求解。SceneTree 调度节点和帧阶段，真正渲染/物理执行在 Server 轴。",
          "误区四：`queue_free()` 立刻删除。它把 ObjectID 放进 SceneTree 删除队列，通常在安全点 flush。",
          "误区五：父节点和 owner 是一回事。parent 决定运行时树结构，owner 决定场景保存边界。"
        ]
      },
      {
        type: "heading",
        title: "边界和相邻概念"
      },
      {
        type: "table",
        headers: ["概念", "场景轴负责", "不负责"],
        rows: [
          ["启动轴", "启动轴创建 SceneTree 并把主场景加入运行。", "场景轴不决定 core/server 的全局初始化顺序。"],
          ["对象轴", "Node 是 Object 子类，PackedScene 实例化依赖 ClassDB。", "场景轴不解释 MethodBind、Variant 调用和类元数据细节。"],
          ["服务轴", "场景节点把渲染、物理、音频、导航状态转给 Server。", "场景轴不保存 GPU 资源、物理求解器内部状态或平台窗口实现。"],
          ["编辑器", "编辑器本身也是一棵 SceneTree。", "编辑器 Dock、Inspector、导入导出 UI 属于 editor 工具层。"]
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/resources/packed_scene.h:38` 和 `246`，区分 SceneState 与 PackedScene。",
          "读 `scene/resources/packed_scene.cpp:155`，跟实例化如何创建节点、恢复属性和连接。",
          "读 `scene/main/node.h:54` 和 `node.cpp:341`、`323`，理解 enter_tree 与 ready 的传播顺序。",
          "读 `scene/main/scene_tree.h:89` 和 `scene_tree.cpp:639`、`688`，理解每帧调度。",
          "最后读 `scene_tree.cpp:1637`、`1702` 和 `node.cpp:3461`，掌握删除和切场景的安全点。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "场景轴是 Godot 的用户对象运行线：它把场景资源实例化成 Node 树，再用 SceneTree 管理生命周期、每帧回调、切场景和安全删除。"
      }
    ]
  },
  {
    id: "service-axis",
    title: "服务轴",
    aliases: ["服务轴", "Server 轴", "底层服务轴", "RID/Server 轴", "Server 委托轴"],
    summary: "围绕 RenderingServer、PhysicsServer、AudioServer、DisplayServer、TextServer、NavigationServer 和 RID 的读法。服务轴解释场景层怎样把渲染、物理、音频、显示、文本等重活交给可替换后端。",
    article: [
      {
        type: "lead",
        text: "服务轴回答的是“真正干重活的系统在哪里”。Node、Resource 和脚本表达用户语义，但可见物体、碰撞体、音频混音、窗口输入、文本 shaping、导航等底层状态通常进入 Server 单例。Server API 用 RID 或对象接口隐藏后端，后端可以是平台实现、渲染设备、物理模块、音频驱动或文本模块。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "把场景层想成前台表单，服务轴像后台专业部门。Sprite2D、MeshInstance3D、RigidBody3D、Control 这些节点保存用户看得懂的状态；真正把纹理送到 GPU、让刚体参与求解、把声音混出去、从系统窗口收输入，都是 Server 或平台后端做的。"
      },
      {
        type: "paragraph",
        text: "RID 是服务轴的关键线索。看到 RID，通常说明场景层没有直接持有后端对象，而是拿着一个句柄，请对应 Server 去创建、更新、同步和释放真实资源。"
      },
      {
        type: "flow",
        title: "服务轴的典型委托流程",
        steps: [
          { title: "场景节点表达用户状态", text: "例如可见节点、物理节点、GUI 控件或音频播放器。" },
          { title: "调用 Server API", text: "节点把 mesh、transform、shape、text、viewport 等状态提交给对应 Server。" },
          { title: "Server 返回或接收 RID", text: "`core/templates/rid.h:38` 的 RID 是 64 位句柄，不暴露后端对象。" },
          { title: "Server 内部 storage 保存真实对象", text: "`RID_Owner` 在 `core/templates/rid_owner.h:517` 管句柄到对象槽的映射。" },
          { title: "Main::iteration 推进同步点", text: "物理 sync/step、RenderingServer sync/draw、导航 process 都在帧循环中推进。" },
          { title: "后端执行", text: "渲染后端、物理模块、音频驱动、DisplayServer 平台实现等完成真实工作。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "服务轴的类型注册入口是 `servers/register_server_types.cpp:146`。这里注册 `TextServerManager`、`TextServer`、`DisplayServer`、`RenderingServer`、`AudioServer`、`CameraServer`、`RenderingDevice`、音频资源等。`servers/register_server_types.cpp:393` 的 `register_server_singletons()` 把 `AudioServer`、`DisplayServer`、`RenderingServer`、`NavigationServer2D/3D`、`PhysicsServer2D/3D`、`XRServer` 等加入 Engine 单例，脚本才能按全局单例访问。"
      },
      {
        type: "paragraph",
        text: "服务轴在启动轴的 setup2 阶段建立。`main/main.cpp:3197` 调 `register_server_types()`；DisplayServer 根据平台 create function 创建；`main/main.cpp:3540` 创建 `RenderingServerDefault(OS::get_singleton()->is_separate_thread_rendering_enabled())` 并 `init()`；随后初始化 AudioServer、XRServer、CameraServer、PhysicsServer 等，`main/main.cpp:3839` 再注册单例。"
      },
      {
        type: "paragraph",
        text: "RenderingServer 的抽象接口在 `servers/rendering/rendering_server.h:64`，它继承 Object 并声明大量虚函数，例如纹理、mesh、canvas、viewport、instance、global shader 参数和帧控制。`servers/rendering/rendering_server.h:966`、`967` 声明 `draw()` 与 `sync()`；`servers/rendering/rendering_server_default.cpp:435` 的实现会根据是否有渲染线程选择 `command_queue.sync()` 或 `flush_all()`，`draw()` 则可能把 `_draw` 推给渲染线程。"
      },
      {
        type: "paragraph",
        text: "PhysicsServer3D 的入口在 `servers/physics_3d/physics_server_3d.h:236`。它同样继承 Object，但暴露的是 shape、space、area、body、joint 等物理 API，返回和接收大量 RID。`servers/physics_3d/physics_server_3d.h:816` 附近声明 `sync()`、`flush_queries()`、`end_sync()`、`step()` 等帧阶段方法，Main 的物理循环会显式调用这些同步点。"
      },
      {
        type: "paragraph",
        text: "RID 不是 Resource，也不是裸指针。`core/templates/rid.h:38` 定义的 RID 只保存一个 64 位 id。真实对象由 `RID_Owner` 或专用 storage 管理；`core/templates/rid_owner.h:175` 的 `make_rid()` 分配并初始化句柄，`core/templates/rid_owner.h:191` 的 `get_or_null()` 验证 id 并取对象槽，`core/templates/rid_owner.h:330` 的 `free()` 释放句柄。"
      },
      {
        type: "table",
        title: "服务轴的证据锚点",
        headers: ["部件", "源码入口", "职责"],
        rows: [
          ["Server 类型注册", "`servers/register_server_types.cpp:146`", "注册 Server 抽象、资源和相关类型。"],
          ["Server 单例", "`servers/register_server_types.cpp:393`、`main/main.cpp:3839`", "把 Server 加入 Engine 单例，让脚本和系统可访问。"],
          ["RenderingServer", "`servers/rendering/rendering_server.h:64`、`966`、`967`", "渲染资源、实例、viewport、sync/draw 抽象 API。"],
          ["RenderingServerDefault", "`main/main.cpp:3540`、`servers/rendering/rendering_server_default.cpp:435`", "默认渲染 Server，实现线程队列、sync、draw。"],
          ["PhysicsServer3D", "`servers/physics_3d/physics_server_3d.h:236`、`816`", "物理 shape/body/space/joint API 和物理步同步点。"],
          ["RID/RID_Owner", "`core/templates/rid.h:38`、`core/templates/rid_owner.h:175`、`191`、`330`", "句柄、后端对象槽、验证与释放。"],
          ["每帧推进", "`main/main.cpp:4896`、`5004`、`5019`、`5037`、`5052`", "物理 Server、导航 Server、RenderingServer 在主循环中被同步和执行。"]
        ]
      },
      {
        type: "heading",
        title: "案例：一个可见 3D 节点为什么要进 RenderingServer"
      },
      {
        type: "flow",
        title: "从场景语义到渲染后端",
        steps: [
          { title: "Node 保存用户语义", text: "例如 MeshInstance3D 保存 mesh、material、transform、可见性等高层状态。" },
          { title: "场景层创建或更新 RID", text: "可见对象最终通过 RenderingServer 的 instance、mesh、material、viewport 等 RID API 表达。" },
          { title: "RenderingServer 保存渲染状态", text: "Server 内部 storage 根据 RID 找真实渲染对象，并记录本帧变更。" },
          { title: "`Main::iteration()` 调 sync/draw", text: "`main/main.cpp:5037` 先 `RenderingServer::sync()`，`5052` 附近按需 `draw()`。" },
          { title: "Renderer/RenderingDevice 提交命令", text: "默认 RD 后端把状态变成 GPU 资源和 draw/compute 提交。" }
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：scene 目录直接实现渲染或物理。scene 多数保存用户语义和句柄，底层执行在 Server 和后端。",
          "误区二：Server API 等于具体后端。`RenderingServer`、`PhysicsServer` 是抽象契约，具体执行可能在 renderer_rd、modules/jolt_physics、platform/display 等位置。",
          "误区三：RID 是 Resource。RID 是服务内部对象句柄，释放要回到对应 Server 或 RID_Owner，不走 ResourceLoader。",
          "误区四：Server 调用一定立即执行。RenderingServerDefault 可能通过命令队列和渲染线程延后执行，真正同步点在 sync/draw。",
          "误区五：所有服务都在 servers 目录完整实现。DisplayServer 的真实窗口和输入实现落在 platform；部分物理、文本、编码能力落在 modules。"
        ]
      },
      {
        type: "heading",
        title: "边界和相邻概念"
      },
      {
        type: "table",
        headers: ["概念", "服务轴负责", "不负责"],
        rows: [
          ["启动轴", "启动轴创建、初始化、每帧同步和清理 Server。", "服务轴不决定整个进程的 setup/start/cleanup 顺序。"],
          ["对象轴", "Server 单例常作为 Object 暴露给脚本和 ClassDB。", "服务轴不解释 Object 的 set/get/call 和信号元数据。"],
          ["场景轴", "场景节点通过 RID 和 Server API 委托底层执行。", "服务轴不保存 Node 父子树、owner、ready/process 状态。"],
          ["平台层", "DisplayServer、AudioDriver、窗口和输入最终落到平台实现。", "服务轴不是某一个平台的原生 API 封装细节。"],
          ["模块层", "Jolt、Godot Physics、TextServer 后端等可作为服务实现接入。", "服务轴不是所有第三方库源码本身。"]
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `servers/register_server_types.cpp:146` 和 `393`，确认 Server 类型与单例边界。",
          "回到 `main/main.cpp:3197`、`3540`、`3839`，看服务什么时候创建和初始化。",
          "读目标 Server 的抽象头文件，例如 `servers/rendering/rendering_server.h:64` 或 `servers/physics_3d/physics_server_3d.h:236`。",
          "看到 RID 时读 `core/templates/rid.h:38` 和 `rid_owner.h:175`、`191`、`330`，确认句柄归属。",
          "最后跟到具体后端和帧同步点：渲染看 `RenderingServerDefault::sync/draw`，物理看 `PhysicsServer*::sync/step`，显示看 `platform/*/display_server_*`。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "服务轴是 Godot 的执行后端边界：场景层表达意图，Server 用 RID 和单例接口保存底层状态，并在主循环同步点把工作交给渲染、物理、音频、显示、文本和平台后端。"
      }
    ]
  },
  {
    id: "object",
    title: "Object",
    aliases: ["Object", "Godot Object", "对象基类", "对象系统"],
    summary: "Godot 运行时对象模型的根：它让 C++ 对象拥有类型信息、属性、方法、信号、脚本实例、元数据和全局 ObjectID。",
    article: [
      {
        type: "lead",
        text: "Object 是 Godot 大多数运行时对象的共同起点。Node、Resource、RefCounted、Script、Server 单例等都站在它提供的对象模型上：能被脚本调用，能出现在 Inspector，能发信号，能通过 `ObjectID` 被重新查找，也能挂脚本实例。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "把 Godot 想成一个大型游戏工厂，Object 就像每个“正式员工”的工牌和服务台规则。只要一个 C++ 类继承 Object 并完成注册，Godot 就知道它叫什么、有哪些公开方法、有哪些属性、能不能发信号、有没有脚本附在身上。"
      },
      {
        type: "paragraph",
        text: "但 Object 不是“场景里的物体”。场景里的物体通常是 Node。Object 更底层：Node 是 Object，Resource 也是 Object，很多编辑器对象、脚本对象、服务对象也都是 Object。Object 解决的是“这个东西如何被引擎统一识别和操作”，不是“这个东西如何显示在场景里”。"
      },
      {
        type: "flow",
        title: "从新手视角看 Object 的作用",
        steps: [
          { title: "C++ 类继承 Object", text: "例如 `Node : public Object`，类进入 Godot 对象体系。" },
          { title: "用 `GDCLASS` 接入类型系统", text: "生成类名、父类、快速类型判断和 ClassDB 初始化入口。" },
          { title: "在 `_bind_methods()` 暴露 API", text: "脚本、Inspector、文档和编辑器才知道方法、属性、信号。" },
          { title: "实例创建后进入 ObjectDB", text: "获得一个 `ObjectID`，后续可以做弱查找和调试统计。" },
          { title: "脚本和编辑器统一访问", text: "`set/get/call/connect` 最终都走 Object 提供的公共路径。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "源码入口在 `core/object/object.h:349`。Object 本体保存了几类关键状态：`_instance_id` 在 `core/object/object.h:428`，脚本实例指针在 `core/object/object.h:449`，metadata 在 `core/object/object.h:450`，信号表和连接列表在 Object 私有区，类型缓存 `_gdtype_ptr` 则把运行期对象和 Godot 的类型描述连起来。"
      },
      {
        type: "paragraph",
        text: "Object 子类通常使用 `GDCLASS(T, Parent)`。这个宏定义在 `core/object/object.h:248`，它建立静态 `GDType`、`get_class_static()`、`initialize_class()`，并把类加入 ClassDB。对不注册进 ClassDB、但仍希望支持 `Object::cast_to()` 的内部类，Godot 还有 `GDSOFTCLASS`，定义在 `core/object/object.h:161`。"
      },
      {
        type: "paragraph",
        text: "Object 的第一个核心职责是身份。构造时 `Object::_construct_object()` 会初始化信号开关、删除标记、祖先类型位，并调用 `ObjectDB::add_instance(this)` 生成 `_instance_id`，源码在 `core/object/object.cpp:2283` 和 `core/object/object.cpp:2450`。析构时 `Object::~Object()` 会断开信号连接并从 ObjectDB 移除，源码在 `core/object/object.cpp:2320` 和 `core/object/object.cpp:2494`。"
      },
      {
        type: "paragraph",
        text: "第二个核心职责是反射调用。`Object::set()` 和 `Object::get()` 分别在 `core/object/object.cpp:233`、`core/object/object.cpp:316`。它们不是简单访问 C++ 成员，而是按顺序询问脚本实例、GDExtension、ClassDB 绑定属性、script 特殊属性、metadata，最后才落到 `_setv/_getv` 这样的虚分发。`Object::callp()` 在 `core/object/object.cpp:768`，会先让脚本实例处理，再去 ClassDB 查 `MethodBind`。"
      },
      {
        type: "paragraph",
        text: "第三个核心职责是信号。Object 通过 `signal_map` 保存本对象发出的信号连接，通过 `connections` 记录别人连到自己的连接。`Object::connect()` 在 `core/object/object.cpp:1517`，会验证 Callable、确认信号是否存在于 `GDType` 或脚本信号里，再登记连接。真正发信号的 `emit_signalp()` 在 `core/object/object.cpp:1178`，会把信号名和参数分发给每个连接的 Callable。"
      },
      {
        type: "paragraph",
        text: "第四个核心职责是生命周期钩子。Godot 推荐用 `memnew` 创建 Object 子类，因为 `postinitialize_handler()` 会调用 `_initialize()` 和 `_postinitialize()`，源码在 `core/object/object.cpp:2377`。`_initialize()` 会缓存类型并确保类初始化，`_postinitialize()` 会创建信号互斥锁并发送 `NOTIFICATION_POSTINITIALIZE`，源码在 `core/object/object.cpp:220` 和 `core/object/object.cpp:226`。删除前还有 `predelete_handler()`，用于发送预删除通知并阻止不安全释放。"
      },
      {
        type: "table",
        title: "Object 负责什么",
        headers: ["职责", "你看到的 API 或现象", "源码入口"],
        rows: [
          ["运行时身份", "`get_instance_id()`、ObjectDB 查找、泄漏统计", "`core/object/object.h:428`、`core/object/object.cpp:2450`"],
          ["类型信息", "`get_class()`、`is_class()`、`Object::cast_to<T>()`", "`core/object/object.h:248`、`core/object/object.h:855`"],
          ["属性访问", "`set()`、`get()`、Inspector 属性列表", "`core/object/object.cpp:233`、`core/object/object.cpp:475`"],
          ["方法调用", "`call()`、`callv()`、脚本调用 C++", "`core/object/object.cpp:768`、`core/object/object.cpp:1835`"],
          ["信号系统", "`connect()`、`emit_signal()`、连接列表清理", "`core/object/object.cpp:1178`、`core/object/object.cpp:1517`"],
          ["脚本桥接", "`set_script()`、脚本属性和脚本方法合并到对象上", "`core/object/object.cpp:967`、`core/object/object.h:449`"],
          ["元数据", "`set_meta()`、`get_meta()`、编辑器附加信息", "`core/object/object.h:450`、`core/object/object.cpp:1013`"]
        ]
      },
      {
        type: "flow",
        title: "Object 创建与释放路径",
        steps: [
          { title: "`memnew(MyObject)`", text: "分配内存并构造 C++ 对象。" },
          { title: "`Object::_construct_object()`", text: "初始化标记、祖先位和 `_instance_id`。" },
          { title: "`ObjectDB::add_instance()`", text: "分配 slot、validator，生成 64 位 `ObjectID`。" },
          { title: "`postinitialize_handler()`", text: "缓存类型、初始化 ClassDB、发送 post initialize 通知。" },
          { title: "对象可被脚本和编辑器访问", text: "属性、方法、信号、metadata 进入统一路径。" },
          { title: "`memdelete()` 或 `free()`", text: "预删除通知、断开连接、ObjectDB 移除、释放实例绑定和信号锁。" }
        ]
      },
      {
        type: "flow",
        title: "一次 `obj.set(\"x\", value)` 的查找顺序",
        steps: [
          { title: "脚本实例", text: "如果对象挂了脚本，先让 `script_instance->set()` 处理。" },
          { title: "GDExtension", text: "扩展类可提供自己的 get/set 回调。" },
          { title: "ClassDB 属性", text: "C++ 绑定属性通过 ClassDB 查 setter。" },
          { title: "`script` 与 metadata 特殊路径", text: "`script` 会改脚本，`metadata/...` 会写元数据。" },
          { title: "对象虚分发", text: "最后调用 `_setv()`，让子类处理动态属性。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：GDScript 里每个 Node 都继承 Object"
      },
      {
        type: "code",
        code: [
          "var node := Node.new()",
          "print(node.get_class())       # Node",
          "print(node.is_class(\"Object\")) # true",
          "node.set_meta(\"from\", \"demo\")",
          "print(node.get_meta(\"from\")) # demo",
          "node.free()"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个例子里 `get_class()`、`is_class()`、`set_meta()`、`get_meta()`、`free()` 都不是 Node 自己发明的能力，而是从 Object 继承来的基础对象能力。"
      },
      {
        type: "subheading",
        title: "案例二：C++ 子类怎样被脚本调用"
      },
      {
        type: "code",
        code: [
          "class TomatoCounter : public Object {",
          "    GDCLASS(TomatoCounter, Object);",
          "",
          "    int total = 0;",
          "",
          "protected:",
          "    static void _bind_methods() {",
          "        ClassDB::bind_method(D_METHOD(\"add\", \"value\"), &TomatoCounter::add);",
          "        ClassDB::bind_method(D_METHOD(\"get_total\"), &TomatoCounter::get_total);",
          "    }",
          "",
          "public:",
          "    void add(int p_value) { total += p_value; }",
          "    int get_total() const { return total; }",
          "};",
          "",
          "// 注册阶段还需要 GDREGISTER_CLASS(TomatoCounter);"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`GDCLASS` 负责接入类型系统，`_bind_methods()` 负责把方法交给 ClassDB。之后脚本调用 `counter.add(3)` 时，本质上会走 Object/ClassDB/MethodBind/Variant 这一条反射调用链。"
      },
      {
        type: "subheading",
        title: "案例三：ObjectID 是弱查找，不是强引用"
      },
      {
        type: "code",
        code: [
          "ObjectID id = node->get_instance_id();",
          "",
          "// 之后某一帧再查：",
          "Object *maybe = ObjectDB::get_instance(id);",
          "if (maybe) {",
          "    // 对象还活着，可以继续做类型检查。",
          "    Node *node_again = Object::cast_to<Node>(maybe);",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "ObjectID 能避免直接保存裸指针后悬空，但它不会让对象继续活着。对象释放后，ObjectDB 的 validator 会失效，`get_instance()` 返回 null。需要真正延长生命周期时，应使用正确的所有权机制，例如 `Ref<T>`、场景树持有，或明确的释放策略。"
      },
      {
        type: "heading",
        title: "Object 和相邻概念的边界"
      },
      {
        type: "table",
        headers: ["概念", "和 Object 的关系", "边界"],
        rows: [
          ["Node", "继承 Object，并加入父子树、通知和 process 生命周期。", "Object 本身没有场景树关系。"],
          ["Resource", "继承 RefCounted/Object，并可被保存、加载、缓存和共享。", "Object 本身不代表可序列化资源。"],
          ["ClassDB", "保存 Object 子类公开出来的类型、方法、属性、信号。", "ClassDB 是登记表，不是对象实例。"],
          ["ObjectDB", "给活着的 Object 分配 ObjectID 并支持弱查找。", "ObjectDB 不拥有对象生命周期。"],
          ["Variant", "可以装 Object 指针或 RefCounted 引用，并参与反射调用。", "Variant 不是 Object 的所有权系统。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Object 等于场景节点。实际是 Node 继承 Object，Object 本身没有树结构、父子关系、enter_tree 或 process。",
          "误区二：ObjectDB 管对象生命周期。ObjectDB 只登记和查找对象，不替你决定谁拥有对象，也不会自动释放普通 Object。",
          "误区三：ObjectID 是引用。它只是带 validator 的弱句柄，不能阻止对象销毁。",
          "误区四：所有 Object 都能手动 `free()`。RefCounted 走引用计数，调 `free()` 在调试路径会报错；Node 常用 `queue_free()` 避免在树遍历或信号发射中立刻释放。",
          "误区五：脚本属性直接写进 C++ 类。脚本实例挂在 Object 上，Object 的 set/get/call 路径把脚本成员和 C++ 绑定成员合并成统一视图。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/object/object.h:349` 的成员区，建立 Object 保存了哪些状态的地图。",
          "再读 `GDCLASS` 和 `GDSOFTCLASS`，理解 Godot 为什么不用普通 C++ RTTI 作为主对象模型。",
          "跟 `Object::_construct_object()` 到 `ObjectDB::add_instance()`，确认 ObjectID 的生成方式。",
          "跟 `Object::set()`、`Object::get()`、`Object::callp()`，理解脚本、扩展和 ClassDB 如何合并。",
          "最后读 `_bind_methods()` 和 `Object::connect()`，把公开 API 与信号系统补齐。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Object 是 Godot 让 C++ 世界、脚本世界、编辑器世界和扩展世界互相认识的最底层对象协议；它不是所有权系统，也不是场景树，但没有它，上层的 Node、Resource、Inspector、信号和脚本调用都接不起来。"
      }
    ]
  },
  {
    id: "classdb",
    title: "ClassDB",
    aliases: ["ClassDB", "类数据库", "类型数据库", "反射系统", "_bind_methods", "bind_method", "GDREGISTER_CLASS"],
    summary: "Godot 的运行时类型登记表：保存 Object 子类的类名、继承、创建函数、方法、属性、信号和常量，让脚本、Inspector、文档、序列化和 GDExtension 能按统一元数据工作。",
    article: [
      {
        type: "lead",
        text: "ClassDB 是 Godot 的“类型登记中心”。Object 让每个对象有统一的对象协议，ClassDB 则回答“这个类叫什么、继承谁、能不能创建、有哪些方法、属性、信号和常量”。脚本调用 C++、Inspector 自动列属性、文档工具生成 API、GDExtension 注册扩展类，都要经过这张表。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 Object 是每个对象的工牌，ClassDB 就是整家公司的人事系统和通讯录。它不保存某一个具体员工今天在哪里，而是保存“有哪些岗位、每个岗位能做什么、这个岗位能不能招人、岗位之间谁继承谁”。"
      },
      {
        type: "paragraph",
        text: "比如 GDScript 里写 `Node.new()`、Inspector 里看到 `process_mode` 属性、脚本调用 `node.queue_free()`，这些都不是编辑器硬编码出来的。Node 在启动注册阶段把自己放进 ClassDB，又在 `_bind_methods()` 里告诉 ClassDB：我有这些方法、属性和信号。之后脚本和编辑器就能按名字查到它们。"
      },
      {
        type: "flow",
        title: "从使用者视角看 ClassDB",
        steps: [
          { title: "源码里有一个 C++ 类", text: "例如 `Node : public Object`，并使用 `GDCLASS(Node, Object)`。" },
          { title: "启动时注册类", text: "`scene/register_scene_types.cpp:438` 调 `GDREGISTER_CLASS(Node)`。" },
          { title: "类进入 ClassDB", text: "`ClassDB::register_class<T>()` 设置创建函数、暴露状态、API 类型和类指针。" },
          { title: "`_bind_methods()` 写入成员信息", text: "`ClassDB::bind_method`、`ADD_PROPERTY`、`ADD_SIGNAL` 把公开 API 变成元数据。" },
          { title: "脚本、Inspector、文档统一查询", text: "运行时按类名和成员名查 MethodBind、PropertyInfo、信号和默认值。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "源码入口是 `core/object/class_db.h:97` 的 `class ClassDB`。真正的全局表是 `ClassDB::classes`，声明在 `core/object/class_db.h:194`，定义在 `core/object/class_db.cpp:67`。它的 key 是 `StringName` 类名，value 是 `ClassInfo`。`ClassInfo` 定义在 `core/object/class_db.h:120`，保存 `inherits_ptr`、`gdtype`、`gdextension`、`method_map`、`property_list`、`property_map`、`property_setget`、`creation_func`、`exposed`、`is_virtual`、`is_runtime` 等关键状态。"
      },
      {
        type: "paragraph",
        text: "类注册分两步。第一步由 `GDCLASS` 生成的 `initialize_class()` 完成父类初始化、创建 `GDType`、调用 `Object::_add_class_to_classdb()`，最终进入 `ClassDB::_add_class()`，源码在 `core/object/class_db.cpp:935`。第二步由 `GDREGISTER_CLASS` 触发 `ClassDB::register_class<T>()`，模板入口在 `core/object/class_db.h:245`，它会设置 `creation_func`、`exposed`、`class_ptr`、`api` 等“这个类能不能被外部创建和看见”的信息。"
      },
      {
        type: "paragraph",
        text: "方法绑定发生在 `_bind_methods()` 中。`ClassDB::bind_method()` 模板在 `core/object/class_db.h:374`，它根据 C++ 成员函数指针创建 `MethodBind`，再进入 `ClassDB::bind_methodfi()`。`bind_methodfi()` 的实现入口在 `core/object/class_db.cpp:1895`，它设置方法名、默认参数、提示 flags，并把 `MethodBind*` 写入当前类的 `method_map`。Godot 不支持同一个类里用同名方法重载；同名绑定会报错。"
      },
      {
        type: "paragraph",
        text: "属性绑定不是直接把字段地址暴露出去，而是把属性名关联到 setter/getter 方法。`ADD_PROPERTY` 宏最终走 `ClassDB::add_property()`，声明在 `core/object/class_db.h:469`，实现是 `core/object/class_db.cpp:1423`。它会检查 setter/getter 是否存在，把 `PropertyInfo` 放进 `property_list` 和 `property_map`，并在 `property_setget` 中缓存 setter/getter 的 `MethodBind`。所以 Inspector 修改属性时，本质上仍然是通过 setter 调用 C++。"
      },
      {
        type: "paragraph",
        text: "运行时查询路径很直接：`Object::callp()` 会用对象类名和方法名调用 `ClassDB::get_method()`，源码在 `core/object/object.cpp:820` 和 `core/object/class_db.cpp:1132`；`Object::set()` 和 `Object::get()` 会分别进入 `ClassDB::set_property()`、`ClassDB::get_property()`，源码在 `core/object/object.cpp:259`、`core/object/object.cpp:338`、`core/object/class_db.cpp:1569`、`core/object/class_db.cpp:1618`。ClassDB 查到 MethodBind 后，再用 Variant 参数完成调用。"
      },
      {
        type: "paragraph",
        text: "ClassDB 还负责按类名创建对象。`ClassDB::instantiate()` 在 `core/object/class_db.cpp:698`，内部走 `_instantiate_internal()`：先查 `ClassInfo`，检查类是否存在、是否 disabled、是否 exposed、是否有 `creation_func`，再调用内置 C++ 创建函数或 GDExtension 创建回调。扩展类注册入口是 `ClassDB::register_extension_class()`，源码在 `core/object/class_db.cpp:2246`，它把外部动态库注册的类也放进同一张 `classes` 表。"
      },
      {
        type: "table",
        title: "ClassDB 里主要存什么",
        headers: ["数据", "作用", "源码入口"],
        rows: [
          ["`classes`", "全局类名到 `ClassInfo` 的映射，ClassDB 的核心表。", "`core/object/class_db.h:194`、`core/object/class_db.cpp:67`"],
          ["`ClassInfo::inherits_ptr`", "指向父类 ClassInfo，用于继承查询和向父类查方法/属性。", "`core/object/class_db.h:122`"],
          ["`ClassInfo::gdtype`", "保存类名、父类名、信号、常量、枚举等类型元数据。", "`core/object/class_db.h:124`"],
          ["`ClassInfo::method_map`", "方法名到 `MethodBind*` 的映射，脚本调用 C++ 时会查这里。", "`core/object/class_db.h:128`"],
          ["`ClassInfo::property_list/property_map`", "Inspector、文档和属性列表使用的 `PropertyInfo` 元数据。", "`core/object/class_db.h:133`"],
          ["`ClassInfo::property_setget`", "属性名到 setter/getter MethodBind 的快速路径。", "`core/object/class_db.h:147`"],
          ["`ClassInfo::creation_func`", "按类名实例化对象时调用的创建函数。", "`core/object/class_db.h:158`"],
          ["`compat_classes`", "旧类名到新类名的兼容映射，用于迁移和反序列化兼容。", "`core/object/class_db.h:196`"]
        ]
      },
      {
        type: "flow",
        title: "一个类进入 ClassDB 的注册链",
        steps: [
          { title: "`GDCLASS(Node, Object)`", text: "在类声明里生成静态类型、类名、父类和初始化入口。" },
          { title: "`GDREGISTER_CLASS(Node)`", text: "注册阶段调用，宏定义在 `core/object/class_db.h:568`。" },
          { title: "`ClassDB::register_class<Node>()`", text: "模板入口在 `core/object/class_db.h:245`，设置 creation/exposed/api。" },
          { title: "`Node::initialize_class()`", text: "先初始化父类，再把 `GDType` 加入 ClassDB。" },
          { title: "`ClassDB::_add_class()`", text: "创建 `ClassInfo`，设置 `gdtype` 和 `inherits_ptr`。" },
          { title: "`Node::_bind_methods()`", text: "把方法、属性、信号和常量写入类型元数据。" }
        ]
      },
      {
        type: "flow",
        title: "脚本调用 C++ 方法的查询链",
        steps: [
          { title: "`node.queue_free()`", text: "脚本侧传入对象、方法名和 Variant 参数。" },
          { title: "`Object::callp()`", text: "通用 Object 调用入口，先尝试脚本实例，再查 C++ 绑定。" },
          { title: "`ClassDB::get_method(\"Node\", \"queue_free\")`", text: "沿 `ClassInfo::inherits_ptr` 向父类查 `method_map`。" },
          { title: "`MethodBind::call()`", text: "检查参数并调用真正的 C++ 成员函数。" },
          { title: "返回 `Variant`", text: "调用结果回到脚本世界。" }
        ]
      },
      {
        type: "flow",
        title: "Inspector 修改属性的查询链",
        steps: [
          { title: "Inspector 选中对象", text: "例如选中一个 Node。" },
          { title: "`Object::get_property_list()`", text: "对象合并脚本属性、扩展属性和 ClassDB 属性列表。" },
          { title: "`ClassDB::get_property_list()`", text: "返回 `PropertyInfo`，Inspector 按类型创建控件。" },
          { title: "用户修改控件", text: "属性名和值进入 `Object::set()`。" },
          { title: "`ClassDB::set_property()`", text: "找到 setter 的 MethodBind 并调用 C++ setter。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：Node 怎样把 `queue_free` 暴露给脚本"
      },
      {
        type: "code",
        code: [
          "// scene/main/node.h:54",
          "class Node : public Object {",
          "    GDCLASS(Node, Object);",
          "    ...",
          "};",
          "",
          "// scene/register_scene_types.cpp:438",
          "GDREGISTER_CLASS(Node);",
          "",
          "// scene/main/node.cpp:3747",
          "void Node::_bind_methods() {",
          "    ClassDB::bind_method(D_METHOD(\"queue_free\"), &Node::queue_free);",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这三段连起来才完整：`GDCLASS` 让 Node 有 Godot 类型信息，`GDREGISTER_CLASS(Node)` 让它进入 ClassDB 并能被按类名创建，`ClassDB::bind_method` 让脚本和编辑器能按名字找到 `queue_free` 的 MethodBind。"
      },
      {
        type: "subheading",
        title: "案例二：属性不是字段，而是 setter/getter 元数据"
      },
      {
        type: "code",
        code: [
          "// scene/main/node.cpp:4038",
          "ADD_PROPERTY(",
          "    PropertyInfo(Variant::INT, \"process_mode\", PROPERTY_HINT_ENUM,",
          "        \"Inherit,Pausable,When Paused,Always,Disabled\"),",
          "    \"set_process_mode\",",
          "    \"get_process_mode\"",
          ");"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Inspector 看到 `process_mode` 后，会知道它是一个整数枚举属性；用户改值时，ClassDB 不会直接写某个字段，而是调用 `set_process_mode`。这也是为什么绑定属性前通常要先绑定对应 setter/getter 方法。"
      },
      {
        type: "subheading",
        title: "案例三：按类名创建对象"
      },
      {
        type: "code",
        code: [
          "Object *obj = ClassDB::instantiate(\"Node\");",
          "Node *node = Object::cast_to<Node>(obj);",
          "if (node) {",
          "    node->set_name(\"RuntimeNode\");",
          "    memdelete(node);",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`instantiate(\"Node\")` 会查 `classes[\"Node\"]`，确认 Node 可实例化，然后调用 `ClassInfo::creation_func`。如果类是 abstract、未 exposed、disabled，或者没有创建函数，实例化会失败。"
      },
      {
        type: "subheading",
        title: "案例四：最小自定义 C++ 类绑定"
      },
      {
        type: "code",
        code: [
          "class HealthBox : public Object {",
          "    GDCLASS(HealthBox, Object);",
          "",
          "    int health = 100;",
          "",
          "protected:",
          "    static void _bind_methods() {",
          "        ClassDB::bind_method(D_METHOD(\"set_health\", \"value\"), &HealthBox::set_health);",
          "        ClassDB::bind_method(D_METHOD(\"get_health\"), &HealthBox::get_health);",
          "        ADD_PROPERTY(PropertyInfo(Variant::INT, \"health\"), \"set_health\", \"get_health\");",
          "        ADD_SIGNAL(MethodInfo(\"health_changed\", PropertyInfo(Variant::INT, \"value\")));",
          "    }",
          "",
          "public:",
          "    void set_health(int p_value) { health = p_value; }",
          "    int get_health() const { return health; }",
          "};",
          "",
          "// 注册阶段：GDREGISTER_CLASS(HealthBox);"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这段展示了 ClassDB 的三个常见动作：方法进 `method_map`，属性进 `property_list/property_setget`，信号进类型元数据。之后脚本能调用 `set_health()`，Inspector 能显示 `health`，连接面板能看到 `health_changed`。"
      },
      {
        type: "heading",
        title: "ClassDB 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "每个实例的对象协议：调用、属性、信号连接、脚本实例、ObjectID。", "不保存所有类的公共 API 总表。"],
          ["ClassDB", "类级元数据：类名、继承、创建函数、MethodBind、PropertyInfo、信号、常量。", "不保存每个对象实例的运行状态。"],
          ["MethodBind", "把一个 C++ 函数包装成可按 Variant/ptrcall 调用的对象。", "不决定一个方法属于哪个类；归属由 ClassDB 的 method_map 管。"],
          ["Variant", "跨脚本、编辑器、序列化和 C++ 传递值。", "不保存类有哪些方法和属性。"],
          ["ObjectDB", "实例 ID 到 Object 指针的弱查找。", "不保存类元数据，也不负责方法绑定。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：有 `GDCLASS` 就能被脚本创建。还需要注册阶段调用 `GDREGISTER_CLASS` 或对应注册函数，并且类要有 creation function 和 exposed 状态。",
          "误区二：`_bind_methods()` 是普通初始化逻辑。它应该只绑定方法、属性、信号、常量等类型元数据，不应该依赖具体实例状态。",
          "误区三：属性绑定等于公开字段。ClassDB 属性通常通过 setter/getter 调用，`PropertyInfo` 只是告诉系统如何展示和校验这个属性。",
          "误区四：ClassDB 只服务脚本。Inspector、文档生成、序列化、编辑器补全、GDExtension、默认值计算都会用到它。",
          "误区五：ClassDB 管对象生命周期。它只知道如何创建某些类；对象创建后的持有、释放、引用计数仍由 Object/RefCounted/Node/Resource 等规则决定。",
          "误区六：方法可以像 C++ 一样重载。ClassDB 的方法名映射不支持同名重载，绑定重复方法名会报错。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/object/class_db.h:97` 到 `ClassInfo`，弄清 ClassDB 保存了哪些元数据。",
          "跟 `GDREGISTER_CLASS` 宏到 `ClassDB::register_class<T>()`，理解 exposed、creation_func、api 的意义。",
          "再读 `ClassDB::_add_class()`，看 `GDType` 和 `inherits_ptr` 如何进入全局表。",
          "读 `bind_method()` 和 `bind_methodfi()`，确认 MethodBind 如何写入 `method_map`。",
          "读 `add_property()`、`set_property()`、`get_property()`，理解 Inspector 和 Object set/get 的真实调用路径。",
          "最后读 `register_extension_class()`，把 GDExtension 如何进入同一张类型表补上。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "ClassDB 是 Godot 的运行时类型目录；Object 负责“某个对象怎么被操作”，ClassDB 负责“某个类公开了哪些能力”。看懂 ClassDB，脚本调用、Inspector 属性、文档 API 和扩展注册就会连成一条线。"
      }
    ]
  },
  {
    id: "methodbind",
    title: "MethodBind",
    aliases: ["MethodBind", "方法绑定", "bind_method", "D_METHOD", "DEFVAL", "ptrcall", "validated_call", "call"],
    summary: "Godot 把 C++ 成员函数变成运行时可调用 API 的包装对象：ClassDB 保存 MethodBind，Object::callp 按方法名取出它，再通过 Variant call、validated_call 或 ptrcall 调到真正的 C++ 函数。",
    article: [
      {
        type: "lead",
        text: "MethodBind 是“脚本按名字调用 C++ 方法”的中间适配器。C++ 里原本只有成员函数指针，脚本和编辑器却需要按字符串方法名、Variant 参数、默认参数、const/static 标记和返回值信息来调用。MethodBind 把这些东西包装成统一对象，再交给 ClassDB 保存。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 ClassDB 是电话簿，MethodBind 就是电话簿里某个联系人旁边的“拨号按钮”。你按 `queue_free` 这个名字，ClassDB 找到按钮；按钮知道真正要拨给哪个 C++ 成员函数、需要几个参数、少传参数时用哪些默认值、返回值要怎么装回 Variant。"
      },
      {
        type: "paragraph",
        text: "比如 C++ 类里有 `void Node::queue_free()`，脚本写 `node.queue_free()` 时不可能直接拿到 C++ 函数指针。Godot 在启动注册阶段通过 `ClassDB::bind_method(D_METHOD(\"queue_free\"), &Node::queue_free)` 创建 MethodBind。运行时脚本只传方法名和参数，MethodBind 负责把这次动态调用接到真正 C++ 方法上。"
      },
      {
        type: "flow",
        title: "从使用者视角看 MethodBind",
        steps: [
          { title: "C++ 类写 `_bind_methods()`", text: "声明哪些 C++ 方法要公开给脚本、Inspector、文档和扩展。" },
          { title: "`ClassDB::bind_method()`", text: "接收方法名、C++ 成员函数指针和默认参数。" },
          { title: "创建 `MethodBind`", text: "模板根据函数签名生成对应包装类，记录参数和返回值类型。" },
          { title: "存入 ClassDB", text: "以方法名为 key，放进当前类的 `method_map`。" },
          { title: "运行时按名字调用", text: "`Object::callp()` 找到 MethodBind，然后执行 `call()` 或快路径调用。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "抽象基类入口是 `core/object/method_bind.h:38` 的 `class MethodBind`。它保存 `method_id`、`hint_flags`、`name`、`instance_class`、`default_arguments`、`argument_count`，以及 `_static`、`_const`、`_returns`、`_returns_raw_obj_ptr` 等标记。它还缓存 `argument_types`，让调用方能查询返回值和参数的 Variant 类型。"
      },
      {
        type: "paragraph",
        text: "最关键的三个虚函数在 `core/object/method_bind.h:115` 附近：`call(Object *, const Variant **, int, Callable::CallError &)` 是通用 Variant 调用；`validated_call(Object *, const Variant **, Variant *)` 是已经验证参数后的调用；`ptrcall(Object *, const void **, void *)` 是裸指针快路径。它们对应 Godot 动态调用的三个性能层级。"
      },
      {
        type: "paragraph",
        text: "绑定入口通常是 `ClassDB::bind_method()`，模板定义在 `core/object/class_db.h:374`。这个模板把 `DEFVAL(...)` 默认参数先存成 Variant 数组，调用 `create_method_bind(p_method)` 生成具体 MethodBind；如果返回类型是 `Object *`，还会设置 `_returns_raw_obj_ptr`。最后它进入 `ClassDB::bind_methodfi()`，把绑定真正写进 ClassDB。"
      },
      {
        type: "paragraph",
        text: "`D_METHOD` 和 `DEFVAL` 是绑定语法里的两个常见宏。`DEFVAL` 在 `core/object/class_db.h:61`，只是把默认值传入绑定模板；`D_METHOD` 在 `core/object/class_db.h:79`。调试构建中它会保留参数名，非调试构建里退化成普通方法名字符串。也就是说，参数名主要服务调试、文档和可读性，不是 C++ 调用必需品。"
      },
      {
        type: "paragraph",
        text: "具体 MethodBind 不是手写一个类对应一个函数，而是由模板生成。`core/object/method_bind_common.h:53` 的 `MethodBindT` 包装“无返回、非 const”的成员函数；后面还有 `MethodBindTC`、`MethodBindTR`、`MethodBindTRC`，分别处理 const、返回值、返回值加 const。构造时它们保存成员函数指针，调用 `_generate_argument_types()`，再设置参数数量。"
      },
      {
        type: "paragraph",
        text: "`ClassDB::bind_methodfi()` 在 `core/object/class_db.cpp:1895`。它会加写锁、设置方法名、找到当前类的 `ClassInfo`、检查重复绑定、在调试构建中保存参数名和顺序，然后把 MethodBind 写进 `type->method_map[mdname]`。Godot 不支持同一个类里按 C++ 风格重载同名方法；重复绑定会报错。"
      },
      {
        type: "table",
        title: "MethodBind 里保存的关键信息",
        headers: ["字段或接口", "作用", "源码入口"],
        rows: [
          ["`name`", "运行时方法名，脚本和 Object::call 用它查找。", "`core/object/method_bind.h:43`"],
          ["`instance_class`", "这个绑定属于哪个 Object 类。", "`core/object/method_bind.h:44`"],
          ["`default_arguments`", "缺少尾部参数时使用的默认值。", "`core/object/method_bind.h:45`、`method_bind.cpp:94`"],
          ["`argument_count`", "真实 C++ 方法需要的参数数量。", "`core/object/method_bind.h:48`"],
          ["`argument_types`", "返回值和参数的 Variant 类型缓存，`argument_types[0]` 是返回值。", "`core/object/method_bind.h:54`、`method_bind.cpp:99`"],
          ["`_const/_static/_returns`", "标记 const 方法、静态方法、是否有返回值。", "`core/object/method_bind.h:50`"],
          ["`call()`", "Variant 参数通用调用，最适合脚本动态调用。", "`core/object/method_bind.h:115`"],
          ["`ptrcall()`", "裸指针快路径，减少 Variant 拆装。", "`core/object/method_bind.h:118`"]
        ]
      },
      {
        type: "table",
        title: "MethodBind 的三种调用模式",
        headers: ["模式", "参数形态", "适合场景", "代价和限制"],
        rows: [
          ["`call()`", "`const Variant **` 参数，返回 `Variant`。", "脚本、Object::call、编辑器动态调用。", "要检查数量、类型、默认参数和转换，最通用但成本最高。"],
          ["`validated_call()`", "仍用 Variant 指针，但假设类型和数量已经验证。", "内置类型方法、缓存后的调用路径。", "省掉部分检查，调用方必须保证参数合法。"],
          ["`ptrcall()`", "`const void **` 参数和 `void *` 返回值。", "GDExtension、生成绑定、已知签名的高性能路径。", "指针布局必须严格匹配；传错就是底层错误。"]
        ]
      },
      {
        type: "flow",
        title: "注册阶段：从 C++ 成员函数到 ClassDB",
        steps: [
          { title: "`_bind_methods()`", text: "类在静态绑定阶段声明公开 API。" },
          { title: "`D_METHOD(\"foo\", \"arg\")`", text: "提供方法名和可选参数名。" },
          { title: "`ClassDB::bind_method()`", text: "收集默认参数，调用 `create_method_bind()`。" },
          { title: "`MethodBindT/TC/TR/TRC`", text: "按函数签名生成具体包装器。" },
          { title: "`ClassDB::bind_methodfi()`", text: "检查重复、保存参数名、设置 flags 和默认值。" },
          { title: "`ClassInfo::method_map`", text: "最终按方法名保存 `MethodBind*`。" }
        ]
      },
      {
        type: "heading",
        title: "运行时调用链"
      },
      {
        type: "paragraph",
        text: "`Object::callp()` 是理解 MethodBind 的最好入口，源码在 `core/object/object.cpp:768`。它先特殊处理 `free`，再给脚本实例一次机会；如果脚本实例没有处理这个方法，就用 `ClassDB::get_method(get_class_name(), p_method)` 查 MethodBind。找到后调用 `method->call(this, p_args, p_argcount, r_error)`；找不到就返回 `CALL_ERROR_INVALID_METHOD`。"
      },
      {
        type: "paragraph",
        text: "`Object::call_const()` 在 `core/object/object.cpp:832` 附近走类似路径，但会检查 `method->is_const()`。这解释了为什么 const 上下文不能随便调用会修改对象状态的方法：MethodBind 不只是函数指针，它还带着方法契约。"
      },
      {
        type: "paragraph",
        text: "通用 `call()` 最终进入 `binder_common.h` 里的 `call_with_variant_args_*_dv` 系列辅助函数，例如 `core/variant/binder_common.h:259`。这些函数处理参数过多、参数过少、默认参数补齐，再通过 `VariantCasterAndValidate` 做严格类型转换和 Object 子类校验。`VariantCasterAndValidate` 的检查入口在 `core/variant/variant_caster.h:162`。"
      },
      {
        type: "paragraph",
        text: "`ptrcall()` 的参数转换则依赖 `PtrToArg`。`core/variant/method_ptrcall.h:42` 开始定义 direct、convert、by-reference 等策略；`PtrToArg<T *>` 在 `core/variant/method_ptrcall.h:252`，`PtrToArg<Ref<T>>` 在 `276`。这套机制让已知签名的调用可以直接传 C++ 类型地址，而不是每次都把值装进 Variant 再拆出来。"
      },
      {
        type: "flow",
        title: "运行时：`node.call(\"rotate\", 1.0)`",
        steps: [
          { title: "参数进入 Variant 数组", text: "`1.0` 以 `Variant::FLOAT` 形态传入。" },
          { title: "`Object::callp()`", text: "先尝试脚本实例，再走 C++ 绑定方法。" },
          { title: "`ClassDB::get_method()`", text: "沿类和父类查 `method_map`。" },
          { title: "`MethodBind::call()`", text: "检查参数数量、默认值、类型转换和对象类型。" },
          { title: "调用成员函数指针", text: "模板包装器把参数拆成 C++ 类型并执行真实方法。" },
          { title: "返回 Variant", text: "返回值重新装回 Variant，错误信息写入 `CallError`。" }
        ]
      },
      {
        type: "flow",
        title: "`call()`、`validated_call()`、`ptrcall()` 的关系",
        steps: [
          { title: "`call()`", text: "最外层动态入口，能处理默认参数和错误码。" },
          { title: "参数被验证", text: "确认数量、Variant 类型、Object 子类关系。" },
          { title: "`validated_call()`", text: "已知参数可靠时可跳过部分检查，仍以 Variant 承载返回值。" },
          { title: "`ptrcall()`", text: "已知签名时直接用 C++ 指针传参和写返回值。" },
          { title: "真实 C++ 方法", text: "三条路最终都落到同一个成员函数或扩展回调。" }
        ]
      },
      {
        type: "heading",
        title: "GDExtension 里的 MethodBind"
      },
      {
        type: "paragraph",
        text: "GDExtension 也接入同一套概念。`core/extension/gdextension.cpp:49` 定义了 `GDExtensionMethodBind : public MethodBind`，内部保存扩展提供的 `call_func`、`validated_call_func`、`ptrcall_func`、方法 userdata、参数信息和返回信息。这样外部动态库注册的方法，也能像内置 C++ 方法一样被 ClassDB、Object::callp 和脚本系统调用。"
      },
      {
        type: "paragraph",
        text: "C ABI 层的入口在 `core/extension/gdextension_interface.cpp:1335` 和 `1350`：`gdextension_object_method_bind_call()` 把扩展传入的 MethodBind 指针转回 `MethodBind *` 并调用 `mb->call()`；`gdextension_object_method_bind_ptrcall()` 调 `mb->ptrcall()`。扩展侧按类名和方法名拿 MethodBind 的入口是 `gdextension_classdb_get_method_bind()`，在 `core/extension/gdextension_interface.cpp:1631`。"
      },
      {
        type: "table",
        title: "内置绑定和扩展绑定的对比",
        headers: ["维度", "内置 C++ 方法", "GDExtension 方法"],
        rows: [
          ["MethodBind 实现", "`method_bind_common.h` 模板类保存 C++ 成员函数指针。", "`GDExtensionMethodBind` 保存扩展提供的 call/ptrcall 回调。"],
          ["注册入口", "`ClassDB::bind_method()`、`bind_static_method()`、`bind_vararg_method()`。", "扩展注册类和方法时构造 `GDExtensionClassMethodInfo`。"],
          ["查找入口", "脚本、Object::callp、Inspector 都从 ClassDB 查。", "扩展也通过 ClassDB 按类名/方法名/hash 查。"],
          ["快路径", "模板生成 `ptrcall()`。", "扩展直接提供 `ptrcall_func`。"],
          ["热重载风险", "内置方法生命周期跟引擎一致。", "扩展 reload 后缓存的 MethodBind 可能失效，源码有 `valid` 检查。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：最常见的绑定"
      },
      {
        type: "code",
        code: [
          "class HealthBox : public Object {",
          "    GDCLASS(HealthBox, Object);",
          "",
          "protected:",
          "    static void _bind_methods() {",
          "        ClassDB::bind_method(D_METHOD(\"set_health\", \"value\"), &HealthBox::set_health);",
          "        ClassDB::bind_method(D_METHOD(\"get_health\"), &HealthBox::get_health);",
          "    }",
          "",
          "public:",
          "    void set_health(int p_value) { health = p_value; }",
          "    int get_health() const { return health; }",
          "",
          "private:",
          "    int health = 100;",
          "};"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`set_health` 会生成无返回 MethodBind，`get_health` 会生成带返回值的 MethodBind。因为 `get_health()` 是 const，模板包装器会设置 `_const`，所以 const 调用路径允许它执行。"
      },
      {
        type: "subheading",
        title: "案例二：默认参数从右侧开始补"
      },
      {
        type: "code",
        code: [
          "ClassDB::bind_method(",
          "    D_METHOD(\"move_by\", \"x\", \"y\", \"speed\"),",
          "    &Mover::move_by,",
          "    DEFVAL(1.0)",
          ");",
          "",
          "// 脚本可以写：",
          "// mover.move_by(10, 20)       # speed 使用默认值 1.0",
          "// mover.move_by(10, 20, 4.0)  # speed 使用传入值"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "默认参数保存在 MethodBind 的 `default_arguments`。`binder_common.h` 的调用辅助函数会计算缺少几个尾部参数，并从默认值数组中补齐。默认参数不是随便补中间某个洞，而是补尾部缺失参数。"
      },
      {
        type: "subheading",
        title: "案例三：运行时手动查 MethodBind"
      },
      {
        type: "code",
        code: [
          "MethodBind *mb = ClassDB::get_method(\"Node\", \"set_name\");",
          "Variant name = StringName(\"Player\");",
          "const Variant *args[] = { &name };",
          "Callable::CallError err;",
          "",
          "if (mb) {",
          "    mb->call(node, args, 1, err);",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这就是 Object::callp 做的核心工作，只是平时你不会手写。它说明 MethodBind 本质上是“类名 + 方法名”查出来的一段可调用元数据。"
      },
      {
        type: "subheading",
        title: "案例四：测试里验证各种函数形态"
      },
      {
        type: "code",
        code: [
          "// tests/core/object/test_method_bind.cpp:113",
          "ClassDB::bind_method(D_METHOD(\"test_method\"), &MethodBindTester::test_method);",
          "ClassDB::bind_method(D_METHOD(\"test_method_args\"), &MethodBindTester::test_method_args);",
          "ClassDB::bind_method(D_METHOD(\"test_methodr\"), &MethodBindTester::test_methodr);",
          "ClassDB::bind_method(D_METHOD(\"test_methodrc_args\"), &MethodBindTester::test_methodrc_args);",
          "ClassDB::bind_method(",
          "    D_METHOD(\"test_method_default_args\"),",
          "    &MethodBindTester::test_method_default_args,",
          "    DEFVAL(9), DEFVAL(4), DEFVAL(5)",
          ");"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个测试覆盖普通方法、带参数方法、const 方法、带返回值方法、默认参数和 Object 子类转换。它是阅读 MethodBind 行为的好证据，因为它从 `Object::call()` 走真实动态调用链。"
      },
      {
        type: "heading",
        title: "MethodBind 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "实例级入口：`callp()`、`call_const()`、脚本实例优先级、对象状态。", "不保存所有方法绑定表。"],
          ["ClassDB", "类级表：方法名到 `MethodBind*` 的映射、继承查询、绑定注册。", "不执行具体 C++ 成员函数。"],
          ["MethodBind", "包装某个 C++ 方法或扩展方法，并提供 call/validated_call/ptrcall。", "不决定这个方法属于哪个类；归属由 ClassDB 的 `method_map` 管。"],
          ["Variant", "通用参数和返回值容器，参与 `call()` 路径。", "不保存方法名、默认参数和函数指针。"],
          ["PropertyInfo", "描述属性或参数类型、hint、usage、名称。", "不负责执行 setter/getter；它只描述元数据。"],
          ["Callable", "保存一个可稍后调用的目标。", "不一定等于 MethodBind；它可能指向脚本函数、自定义 callable 或对象方法。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：MethodBind 就是 C++ 方法本体。它只是包装器，真正逻辑还在原来的成员函数或扩展回调里。",
          "误区二：ClassDB::bind_method 支持 C++ 风格重载。同一个类的同名方法会被拒绝，脚本 API 不能靠参数列表重载。",
          "误区三：D_METHOD 决定参数类型。参数类型来自 C++ 函数签名和 `GetTypeInfo`，D_METHOD 主要提供方法名和参数名。",
          "误区四：默认参数可以补任意位置。MethodBind 的默认参数按尾部缺失参数补齐。",
          "误区五：ptrcall 只是 call 的别名。ptrcall 是裸指针快路径，类型和内存布局必须严格匹配。",
          "误区六：MethodBind 只服务 GDScript。Inspector、文档、C#、GDExtension、UndoRedo 和编辑器工具也会通过它或它的元数据工作。",
          "误区七：扩展可以永久缓存 MethodBind 而不用管版本。GDExtension 查 MethodBind 带 hash，方法签名变化或热重载会让旧缓存失效。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/object/method_bind.h:38`，确认 MethodBind 保存哪些元数据和暴露哪些虚函数。",
          "再读 `core/object/class_db.h:374` 的 `bind_method()` 模板，理解默认参数和 `create_method_bind()` 的入口。",
          "进入 `core/object/method_bind_common.h:53`，看 `MethodBindT/TC/TR/TRC` 怎样按函数签名生成包装器。",
          "读 `core/object/class_db.cpp:1895` 的 `bind_methodfi()`，确认 MethodBind 如何写入 `ClassInfo::method_map`。",
          "读 `core/object/object.cpp:768` 的 `Object::callp()`，跟运行时如何从方法名走到 `method->call()`。",
          "最后读 `binder_common.h`、`variant_caster.h` 和 `method_ptrcall.h`，区分 Variant 调用、验证调用和 ptrcall 的成本边界。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "MethodBind 是 Godot 动态 API 的函数适配器：ClassDB 用它把方法名映射到 C++ 或扩展回调，Object::callp 用它执行动态调用，Variant 和 ptrcall 则分别提供通用性与性能。"
      }
    ]
  },
  {
    id: "variant",
    title: "Variant",
    aliases: ["Variant", "变体", "动态值", "值容器", "Variant::Type", "callp", "ptrcall", "Array", "Dictionary", "PackedArray"],
    summary: "Godot 的通用值容器：用一个固定类型标签和一块紧凑存储，把 int、String、Vector、Object、Array、Dictionary 等值统一带过脚本、编辑器、序列化、反射调用和 GDExtension 边界。",
    article: [
      {
        type: "lead",
        text: "Variant 是 Godot 的“通用值盒子”。当引擎只知道“这里会传一个值”，但不知道它在运行时到底是整数、字符串、节点、数组、字典还是数学类型时，就用 Variant 装起来。脚本调用 C++、Inspector 改属性、信号传参、资源序列化、GDExtension ABI 都依赖它。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Variant 想成快递盒：盒子外面贴着一张标签，写着里面是 `INT`、`STRING`、`OBJECT`、`ARRAY` 还是别的类型；盒子里面放真正的数据。收件人先看标签，再用正确方式拆盒。"
      },
      {
        type: "paragraph",
        text: "为什么 Godot 需要它？因为脚本语言很灵活：你可以把 `3`、`\"hello\"`、`Node`、`Array` 都当参数传给同一个调用系统。C++ 本来更严格，函数参数类型在编译期就定好了。Variant 就是脚本世界和 C++ 世界之间的统一包装层。"
      },
      {
        type: "flow",
        title: "从使用者视角看 Variant",
        steps: [
          { title: "脚本写下一个值", text: "例如 `node.set(\"visible\", true)` 或 `sprite.position = Vector2(10, 20)`。" },
          { title: "值被装进 Variant", text: "Variant 记录类型标签，例如 `BOOL`、`VECTOR2`、`OBJECT`。" },
          { title: "反射系统按统一接口传递", text: "`MethodBind::call()` 使用 `const Variant **` 接收参数。" },
          { title: "C++ 取出真实类型", text: "绑定层检查并转换类型，再调用具体 C++ 函数。" },
          { title: "结果再装回 Variant", text: "返回值回到脚本、编辑器或序列化系统。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "源码入口是 `core/variant/variant.h:93` 的 `class Variant`。它的核心结构不是继承树，而是一个标签加一个 union：`Variant::Type type` 记录当前值类型，`_data` 这个 union 存储 bool、int、float、对象信息、packed array 指针或一段内联内存。类型枚举从 `core/variant/variant.h:96` 开始，包含 `NIL`、原子类型、数学类型、`StringName`、`NodePath`、`RID`、`OBJECT`、`CALLABLE`、`SIGNAL`、`DICTIONARY`、`ARRAY` 和一组 packed array。"
      },
      {
        type: "paragraph",
        text: "Variant 的设计目标是“统一”但不是“随便”。它的可存类型是固定枚举，不是任意 C++ 类型。要让一个 C++ 对象进入 Variant，通常它得是 `Object` 派生类；要让一个方法能被脚本调用，参数和返回值也要能映射到 Variant 支持的类型。"
      },
      {
        type: "paragraph",
        text: "存储模型很关键。`variant.h:160` 的注释写明：`real_t` 为 float 时 Variant 占 24 字节，为 double 时占 40 字节；只有 `AABB/Transform2D`、`Basis/Transform3D`、`Projection` 以及 `PackedArray/Array/Dictionary` 等情况需要额外内存。多数小值能直接放在 union 或 `_mem` 内联区里。"
      },
      {
        type: "paragraph",
        text: "生命周期由 `needs_deinit` 和 `_clear_internal()` 分工处理。`needs_deinit` 在 `core/variant/variant.h:268` 标出哪些类型需要析构、释放池内存或减少引用。`clear()` 在 `core/variant/variant.h:314` 先快速判断当前类型是否需要清理；真正的清理逻辑在 `core/variant/variant.cpp:1367`，例如销毁 `String`、释放 `Transform3D` 池对象、对 `OBJECT` 调 `_get_obj().unref()`、对 packed arrays 调 `PackedArrayRefBase::destroy()`。"
      },
      {
        type: "paragraph",
        text: "复制也不是简单 memcpy。`Variant::reference()` 在 `core/variant/variant.cpp:1129` 按类型复制：小型标量直接拷贝，大型数学对象按需要分配或复制，`Array`/`Dictionary` 拷贝的是带引用计数的内部数据，packed array 走 `PackedArrayRefBase`，`OBJECT` 则交给 `ObjData::ref()`。"
      },
      {
        type: "table",
        title: "Variant::Type 的主要类别",
        headers: ["类别", "典型类型", "怎么理解"],
        rows: [
          ["空值", "`NIL`", "没有值，类似脚本里的 null。"],
          ["原子值", "`BOOL`、`INT`、`FLOAT`、`STRING`", "最常见的小值和字符串。"],
          ["数学值", "`VECTOR2`、`RECT2`、`TRANSFORM3D`、`BASIS`、`PROJECTION`", "引擎大量使用的坐标、矩阵、包围盒等值类型。"],
          ["运行时句柄", "`RID`、`OBJECT`、`CALLABLE`、`SIGNAL`", "连接到对象系统、Server 资源句柄、回调和信号。"],
          ["容器", "`ARRAY`、`DICTIONARY`", "脚本最常用的动态容器，内部有共享和引用计数规则。"],
          ["PackedArray", "`PACKED_BYTE_ARRAY`、`PACKED_VECTOR3_ARRAY` 等", "紧凑数组，用于大量同类型数据，减少普通 Array 的 Variant 包装成本。"]
        ]
      },
      {
        type: "table",
        title: "存储与生命周期规则",
        headers: ["值类型", "Variant 内部怎么放", "清理时发生什么"],
        rows: [
          ["`BOOL`、`INT`、`FLOAT`", "直接放进 union 的 `_bool/_int/_float`。", "不需要析构，`needs_deinit` 是 false。"],
          ["小型数学值、`Color`", "通常放进 `_mem` 内联存储。", "很多是平凡值，清理很快。"],
          ["`String`、`StringName`、`NodePath`", "在 `_mem` 中 placement-new 构造对象。", "`_clear_internal()` 显式调用析构。"],
          ["`AABB`、`Basis`、`Transform3D`、`Projection`", "过大的对象放到 `VariantPools` 分配的内存。", "析构后归还池内存。"],
          ["`OBJECT`", "存 `ObjData { ObjectID id, Object *obj }`。", "RefCounted 会增减引用；普通 Object 不会被 Variant 自动拥有。"],
          ["`Array`、`Dictionary`", "内联保存容器对象，容器内部指向共享私有数据。", "析构容器对象，私有数据按引用计数释放。"],
          ["Packed arrays", "存 `PackedArrayRefBase *`，内部持有同类型 `Vector<T>`。", "引用计数归零后删除 packed array ref。"]
        ]
      },
      {
        type: "heading",
        title: "Object 在 Variant 里的特殊性"
      },
      {
        type: "paragraph",
        text: "`OBJECT` 类型不是把整个对象复制进 Variant，而是保存 `ObjData`。`ObjData` 定义在 `core/variant/variant.h:169`，里面有 `ObjectID id` 和 `Object *obj`。构造 `Variant(const Object *p_object)` 的入口在 `core/variant/variant.cpp:2507`，它调用 `ref_pointer()` 保存对象信息。"
      },
      {
        type: "paragraph",
        text: "如果对象是 `RefCounted`，`ObjData::ref_pointer()` 会调用 `init_ref()`，`ObjData::ref()` 会调用 `reference()`，`ObjData::unref()` 会在引用归零时 `memdelete()`，这些逻辑在 `core/variant/variant.cpp:1074` 到 `1117`。如果对象不是 `RefCounted`，Variant 只是保存 ID 和指针，并不拥有这个对象。"
      },
      {
        type: "paragraph",
        text: "因此 `Variant::operator Object *()` 在 `core/variant/variant.cpp:2024` 只是返回缓存的裸指针；如果你担心普通 Object 已经释放，应该看 `get_validated_object()`，它在 `core/variant/variant.cpp:2043` 通过 `ObjectDB::get_instance(_get_obj().id)` 重新验证。"
      },
      {
        type: "flow",
        title: "Object Variant 的安全读取路径",
        steps: [
          { title: "`Variant v = node`", text: "Variant 存入 `ObjectID` 和 `Object *`。" },
          { title: "对象可能被释放", text: "非 RefCounted 对象不由 Variant 拥有。" },
          { title: "危险路径：直接转 `Object *`", text: "`operator Object*` 返回缓存指针，不负责重新查 ObjectDB。" },
          { title: "安全路径：`get_validated_object()`", text: "用 `ObjectID` 到 `ObjectDB` 查询当前是否仍有效。" },
          { title: "再 `Object::cast_to<T>()`", text: "确认对象活着后再做类型转换和使用。" }
        ]
      },
      {
        type: "heading",
        title: "调用、运算和构造是怎么接上的"
      },
      {
        type: "paragraph",
        text: "脚本调用 C++ 时，`MethodBind` 的通用入口使用 Variant 参数。`core/object/method_bind.h:115` 声明 `virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const = 0;`，旁边还有 `validated_call()` 和 `ptrcall()`。`call()` 是通用、安全但更动态的路径；`ptrcall()` 是已知类型后的快路径，用裸指针参数减少 Variant 检查和拆装成本。"
      },
      {
        type: "paragraph",
        text: "Variant 本身也能按方法名调用。`Variant::callp()` 在 `core/variant/variant_call.cpp:1416`：如果当前 Variant 是 `OBJECT`，就转给 `Object::callp()`；否则在内置类型方法表里查，例如 `String.length()`、`Array.push_back()` 这类 built-in 方法。内置方法表由 `_register_variant_methods()` 注册，入口在 `core/variant/variant_call.cpp:3202`。"
      },
      {
        type: "paragraph",
        text: "运算符也走表驱动。`Variant::evaluate()` 在 `core/variant/variant_op.cpp:1041`，它读取左右操作数类型，再从 `operator_evaluator_table[p_op][type_a][type_b]` 里查求值函数。构造函数表在 `core/variant/variant_construct.cpp:264` 的 `Variant::construct()` 使用，会按目标类型和参数数目查匹配项，并用 `can_convert_strict()` 检查参数能否严格转换。"
      },
      {
        type: "flow",
        title: "脚本方法调用中的 Variant",
        steps: [
          { title: "GDScript 发起调用", text: "例如 `body.apply_impulse(Vector2(0, -300))`。" },
          { title: "参数装成 `Variant`", text: "`Vector2` 作为 `Variant::VECTOR2` 进入调用层。" },
          { title: "ClassDB 找到 `MethodBind`", text: "按对象类名和方法名查绑定函数。" },
          { title: "`MethodBind::call()` 检查并拆参", text: "从 `const Variant **` 读取参数，校验类型和数量。" },
          { title: "执行 C++ 方法", text: "调用真正的成员函数。" },
          { title: "返回值装回 `Variant`", text: "脚本拿到统一返回值。"}
        ]
      },
      {
        type: "flow",
        title: "Variant::callp() 的分发路径",
        steps: [
          { title: "调用 `v.callp(method, args, count, ret, err)`", text: "调用方只知道 Variant 和方法名。" },
          { title: "如果 `v` 是 `OBJECT`", text: "检查对象是否为空，调 `Object::callp()`。" },
          { title: "如果是内置类型", text: "查 `builtin_method_info[type][method]`。" },
          { title: "找不到方法", text: "设置 `CALL_ERROR_INVALID_METHOD`。" },
          { title: "找到方法", text: "执行内置方法 wrapper，结果写入 `r_ret`。" }
        ]
      },
      {
        type: "flow",
        title: "Variant 生命周期",
        steps: [
          { title: "构造", text: "根据传入值设置 `type`，把数据放进 union、`_mem`、对象信息或引用计数容器。" },
          { title: "复制", text: "`reference()` 按类型决定直接拷贝、增加引用、复制池对象或复制容器句柄。" },
          { title: "使用", text: "通过类型判断、转换、`callp()`、`evaluate()`、set/get 等路径访问。" },
          { title: "清理", text: "`clear()` 查 `needs_deinit`，必要时进入 `_clear_internal()`。" },
          { title: "回到 NIL", text: "清理后 `type = NIL`，表示盒子空了。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：C++ 通用函数按 Variant 类型处理值"
      },
      {
        type: "code",
        code: [
          "String describe_value(const Variant &p_value) {",
          "    switch (p_value.get_type()) {",
          "        case Variant::INT:",
          "            return \"int: \" + itos(int64_t(p_value));",
          "        case Variant::STRING:",
          "            return \"string: \" + String(p_value);",
          "        case Variant::OBJECT: {",
          "            Object *obj = p_value.get_validated_object();",
          "            return obj ? obj->get_class() : \"freed or null object\";",
          "        }",
          "        default:",
          "            return Variant::get_type_name(p_value.get_type());",
          "    }",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这段代码展示了正确姿势：先读 `get_type()`，再按类型取值。处理 `OBJECT` 时用 `get_validated_object()`，避免拿到已经无效的裸指针。"
      },
      {
        type: "subheading",
        title: "案例二：MethodBind 的通用调用形态"
      },
      {
        type: "code",
        code: [
          "Variant arg0 = Vector2(10, 20);",
          "const Variant *args[] = { &arg0 };",
          "Callable::CallError error;",
          "",
          "// 真实代码里 MethodBind 通常来自 ClassDB::get_method().",
          "Variant ret = method_bind->call(object, args, 1, error);",
          "if (error.error != Callable::CallError::CALL_OK) {",
          "    // 参数数量、类型或方法状态不匹配。",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`call()` 的好处是统一：脚本语言、编辑器和扩展层可以用同一种参数形态调用 C++。代价是要做参数检查、转换和包装；生成绑定或性能敏感路径会尽量走 `ptrcall()`。"
      },
      {
        type: "subheading",
        title: "案例三：Array 和 Dictionary 复制不是深拷贝"
      },
      {
        type: "code",
        code: [
          "# GDScript 直觉示例",
          "var a = [1, 2]",
          "var b = a",
          "b[0] = 99",
          "print(a) # 通常会看到 [99, 2]",
          "",
          "var c = a.duplicate(true)",
          "c[0] = 7",
          "print(a) # 深拷贝后不再被 c 的修改影响"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`Array` 的私有数据在 `core/variant/array.cpp:38`，`Array::_ref()` 在 `core/variant/array.cpp:56` 增加引用计数。`Dictionary::_ref()` 在 `core/variant/dictionary.cpp:287`。所以 Variant 复制容器时要理解共享语义，需要独立副本时用 `duplicate()`。"
      },
      {
        type: "subheading",
        title: "案例四：Object Variant 的安全使用"
      },
      {
        type: "code",
        code: [
          "Variant maybe_node = some_object;",
          "",
          "Object *obj = maybe_node.get_validated_object();",
          "Node *node = Object::cast_to<Node>(obj);",
          "if (node) {",
          "    node->set_name(\"CheckedNode\");",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这比直接写 `Object *obj = maybe_node;` 更稳。直接转换只取缓存指针；验证路径会用 `ObjectID` 问 `ObjectDB` 这个对象现在是否还活着。"
      },
      {
        type: "heading",
        title: "Variant 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "具体对象实例的统一协议：属性、方法、信号、脚本实例、ObjectID。", "不负责装载任意值，也不表示 int/String/Array 这些值。"],
          ["ClassDB", "类级元数据：类名、继承、MethodBind、PropertyInfo、信号。", "不保存每次调用传入的具体参数值。"],
          ["MethodBind", "把 C++ 方法包装成可由 Variant 或 ptrcall 调用的入口。", "不定义 Variant 的类型集合和存储方式。"],
          ["Variant", "统一承载运行时值，支持调用、转换、运算、构造、set/get。", "不是任意类型系统，也不自动拥有所有 Object。"],
          ["Array/Dictionary", "脚本容器，内部保存 Variant 元素或键值。", "不是深拷贝默认语义；复制常常共享内部数据。"],
          ["RID", "Server 资源句柄，常作为 Variant 值传递。", "不暴露具体 Server 对象的 C++ 指针和生命周期。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Variant 是任意类型容器。它只能保存 `Variant::Type` 枚举支持的类型，普通 C++ 类型不能直接塞进去。",
          "误区二：Variant 等于 Object。Object 是对象协议；Variant 可以装 Object 指针，也可以装 int、String、Vector、Array 等值。",
          "误区三：`Variant(Object*)` 一定拥有对象。只有 RefCounted 会走引用计数；普通 Object 仍由原来的所有权规则管理。",
          "误区四：直接转 `Object *` 总是安全。`operator Object*` 返回缓存指针；普通 Object 可能已释放，必要时用 `get_validated_object()`。",
          "误区五：Array/Dictionary 赋值就是深拷贝。它们有引用计数私有数据，独立副本要显式 `duplicate()`。",
          "误区六：Variant 方便，所以核心热路径都该用它。类型已知、调用频繁时，typed API 或 `ptrcall()` 往往更合适。",
          "误区七：所有转换都理所当然。Godot 区分 `can_convert()` 和 `can_convert_strict()`，构造和绑定调用会按规则检查。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/variant/variant.h:93` 到 `Variant::Type`，确认 Variant 到底支持哪些值。",
          "继续读 `type`、`ObjData`、`PackedArrayRefBase` 和 `_data` union，建立存储模型。",
          "读 `needs_deinit`、`clear()` 和 `Variant::_clear_internal()`，理解哪些类型需要析构或引用计数清理。",
          "读 `Variant::reference()` 和各类构造函数，尤其是 `Variant(const Object *)`，弄清复制和对象引用规则。",
          "读 `Variant::callp()`、`Variant::evaluate()`、`Variant::construct()`，把方法调用、运算符和构造表串起来。",
          "最后回到 `MethodBind::call()`/`ptrcall()`，理解脚本调用 C++ 时 Variant 为什么既是统一层也是性能边界。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Variant 是 Godot 把动态脚本、C++ 引擎、编辑器、序列化和扩展 ABI 连接起来的通用值协议；它让系统统一，但它的类型集合、对象引用、容器共享和调用成本都必须按源码规则理解。"
      }
    ]
  },
  {
    id: "callable-signal",
    title: "Callable / Signal",
    aliases: ["Callable", "Signal", "Callable / Signal", "信号", "可调用对象", "connect", "disconnect", "emit_signal", "call_deferred", "callable_mp", "bind", "unbind", "CONNECT_DEFERRED"],
    summary: "Callable 统一表示“之后可以调用的目标”，Signal 统一表示“某个 Object 上的某个信号”。它们把信号连接、延迟调用、脚本函数、C++ 方法指针和编辑器保存的连接串进同一套 Object/Variant 调用系统。",
    article: [
      {
        type: "lead",
        text: "Callable 和 Signal 是 Godot 运行时里“把动作保存起来、稍后再触发”的核心值类型。Callable 代表一个可调用目标，可能是对象方法、脚本函数、C++ 方法指针、静态函数或带绑定参数的包装器；Signal 代表某个对象上的某个信号，并把 `connect()`、`emit()`、`disconnect()` 转交给对应 Object。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "Callable 可以理解成一张“回拨卡片”：上面写着要找哪个对象、调用哪个方法，以及可能提前贴上的几个参数。你不一定马上拨出去，可以把卡片交给信号系统、延迟调用队列、UndoRedo 或数组排序函数，等合适的时候再执行。"
      },
      {
        type: "paragraph",
        text: "Signal 可以理解成对象身上的“门铃”。`button.pressed` 是 Button 对象上的一个门铃；你把一个 Callable 接到门铃上，门铃响时 Godot 就按连接表依次调用这些 Callable。Signal 不是全局广播站，它始终属于某个 Object。"
      },
      {
        type: "flow",
        title: "从使用者视角看 Callable / Signal",
        steps: [
          { title: "创建 Callable", text: "例如 `Callable(self, \"_on_pressed\")` 或 C++ 的 `callable_mp(this, &T::method)`。" },
          { title: "连接到 Signal", text: "`button.pressed.connect(callable)` 最终进入 `Object::connect()`。" },
          { title: "Object 保存连接", text: "源对象按信号名保存目标 Callable，目标对象也记录反向连接。" },
          { title: "信号发射", text: "`emit_signal()` 或 `Signal::emit()` 找出连接列表。" },
          { title: "逐个调用 Callable", text: "立即调用、延迟调用、一次性断开、追加源对象等 flags 在这里生效。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "源码入口是 `core/variant/callable.h:48` 的 `class Callable`。它内部不是直接保存 `Object *`，而是保存 `StringName method` 和一个 union：标准 Callable 用 `uint64_t object` 保存 ObjectID；自定义 Callable 用 `CallableCustom *custom`。`is_standard()` 表示对象方法，`is_custom()` 表示自定义实现，`is_null()` 表示空 Callable。"
      },
      {
        type: "paragraph",
        text: "标准 Callable 的构造函数在 `core/variant/callable.cpp:392` 和 `407` 附近：`Callable(const Object *p_object, const StringName &p_method)` 会保存对象的 `get_instance_id()` 和方法名。调用时 `Callable::callp()` 在 `core/variant/callable.cpp:44` 先判断 null/custom；标准 Callable 会用 `ObjectDB::get_instance(ObjectID(object))` 找回对象，再调用 `obj->callp(method, p_arguments, p_argcount, r_call_error)`。"
      },
      {
        type: "paragraph",
        text: "自定义 Callable 由 `CallableCustom` 支撑，抽象类在 `core/variant/callable.h:143`。它定义 `hash()`、`get_as_text()`、比较函数、`is_valid()`、`get_object()`、`call()`、`rpc()`、绑定参数查询等接口。`Callable` 复制和析构时会对 `CallableCustom` 做引用计数，相关逻辑在 `core/variant/callable.cpp:416` 到 `444`。"
      },
      {
        type: "paragraph",
        text: "Signal 的源码入口是 `core/variant/callable.h:178` 的 `class Signal`。它只保存 `StringName name` 和 `ObjectID object`，本身不保存连接列表。`Signal::emit()`、`Signal::connect()`、`Signal::disconnect()` 在 `core/variant/callable.cpp:534`、`543`、`550`，都是先通过 ObjectDB 找回对象，再转给 `Object::emit_signalp()`、`Object::connect()`、`Object::disconnect()`。"
      },
      {
        type: "paragraph",
        text: "真正的信号连接数据在 Object 内部。`Object::Connection` 定义在 `core/object/object.h:387`，包含 `Signal signal`、`Callable callable` 和 flags。`Object::SignalData` 定义在 `core/object/object.h:411`，其中 `slot_map` 用 Callable 作为 key 保存每个连接槽，`connections` 则是目标对象上的反向连接列表。这种双向记录让源对象释放或目标对象释放时都能清理连接。"
      },
      {
        type: "table",
        title: "Callable 的三种形态",
        headers: ["形态", "内部表示", "典型来源", "调用方式"],
        rows: [
          ["空 Callable", "`method == StringName()` 且 `object == 0`。", "默认构造、无效返回值。", "`callp()` 返回 `CALL_ERROR_INSTANCE_IS_NULL`。"],
          ["标准 Callable", "`method` 非空，`object` 保存 ObjectID。", "`Callable(obj, \"method\")`、脚本里的 `Callable(self, \"foo\")`。", "通过 ObjectDB 找对象，再走 `Object::callp()`。"],
          ["自定义 Callable", "`method` 为空，`custom` 指向 `CallableCustom`。", "`bind()`、`unbind()`、`callable_mp()`、Variant built-in 方法、脚本 lambda。", "调用 `custom->call()`，具体行为由派生类决定。"]
        ]
      },
      {
        type: "table",
        title: "Signal 连接真正存在哪里",
        headers: ["结构", "存放位置", "作用"],
        rows: [
          ["`Signal`", "Variant 值类型，保存 ObjectID 和信号名。", "作为脚本可见的信号引用和代理入口，不保存 slot 列表。"],
          ["`Object::signal_map`", "源 Object 内部，key 是信号名。", "找到某个信号的所有连接槽。"],
          ["`SignalData::slot_map`", "源 Object 的每个 SignalData 内。", "以 Callable 的 base comparator 为 key，保存目标 callable、flags、引用计数。"],
          ["`Object::connections`", "目标 Object 内部。", "记录“有哪些源信号连到了我”，目标释放时反向断开。"],
          ["`MethodInfo user`", "SignalData 内。", "保存每实例动态添加的 user signal 元数据。"]
        ]
      },
      {
        type: "heading",
        title: "Callable 调用链"
      },
      {
        type: "paragraph",
        text: "`Callable::callp()` 是核心运行入口。空 Callable 返回 `CALL_ERROR_INSTANCE_IS_NULL`；custom Callable 先检查 `is_valid()` 再调用 `custom->call()`；标准 Callable 则通过 ObjectDB 查对象，再调用对象的 `callp()`。所以 Callable 不绕开 Object/ClassDB/MethodBind，它只是把“稍后调用谁”保存成一个值。"
      },
      {
        type: "paragraph",
        text: "`Callable::call_deferredp()` 在 `core/variant/callable.cpp:40`，直接调用 `MessageQueue::get_singleton()->push_callablep(*this, p_arguments, p_argcount, true)`。这解释了 `call_deferred()` 为什么不是下一行立即执行，而是把 Callable 和参数装进队列，等 MessageQueue flush。"
      },
      {
        type: "paragraph",
        text: "`bind()` 和 `unbind()` 并不修改原 Callable，而是创建新的 CallableCustom 包装器。`CallableCustomBind::call()` 在 `core/variant/callable_bind.cpp:141` 会把绑定参数追加到调用参数后面，再转给原 Callable；`CallableCustomUnbind::call()` 在 `core/variant/callable_bind.cpp:251` 会从传入参数尾部扣掉指定数量，再转发。测试 `tests/core/variant/test_callable.cpp:172` 专门覆盖了多层 bind/unbind 的有效参数顺序。"
      },
      {
        type: "flow",
        title: "`Callable(self, \"foo\").call(1)` 的执行路径",
        steps: [
          { title: "Callable 保存 ObjectID 和方法名", text: "构造时不保存强引用，也不复制对象。" },
          { title: "`callp()` 被调用", text: "参数以 `const Variant **` 进入。" },
          { title: "ObjectDB 查对象", text: "根据 ObjectID 找当前仍然存在的对象。" },
          { title: "`Object::callp(\"foo\")`", text: "对象先问脚本实例，再问 ClassDB/MethodBind。" },
          { title: "返回 Variant 或 CallError", text: "结果和错误码回到 Callable 调用方。" }
        ]
      },
      {
        type: "flow",
        title: "`bind()` 和 `unbind()` 的参数变化",
        steps: [
          { title: "原始 Callable", text: "`target(a, b, c)` 需要三个运行时参数。" },
          { title: "`bind(x)`", text: "创建包装器，调用时把 `x` 追加到末尾。" },
          { title: "`unbind(1)`", text: "创建包装器，调用时忽略传入参数尾部的 1 个。" },
          { title: "可以叠加", text: "每次 bind/unbind 都包住上一个 Callable。" },
          { title: "连接比较看 base comparator", text: "信号连接/断开时可忽略 bind 层，避免同一目标重复连接混乱。" }
        ]
      },
      {
        type: "heading",
        title: "Signal 发射链"
      },
      {
        type: "paragraph",
        text: "`Object::connect()` 在 `core/object/object.cpp:1517`。它先拒绝空 Callable，再按信号名查 `signal_map`；如果还没有 SignalData，会确认这个信号是否来自 ClassDB/GDType、脚本 signal 或工具模式容错，合法后创建 SignalData。随后它用 `p_callable.get_base_comparator()` 作为 key，避免带不同 bind 包装的同一基础 Callable 被重复连接。"
      },
      {
        type: "paragraph",
        text: "`Object::emit_signalp()` 在 `core/object/object.cpp:1178`。它先检查 `_block_signals`，再从 `signal_map` 复制一份 Callable 和 flags 到临时数组。复制这一步很重要：信号回调过程中可能 disconnect，甚至源对象或目标对象可能被释放，直接遍历原 map 会不安全。"
      },
      {
        type: "paragraph",
        text: "发射时 flags 会改变行为：`CONNECT_ONE_SHOT` 在调用前先断开；`CONNECT_APPEND_SOURCE_OBJECT` 会把源对象作为额外参数插入；`CONNECT_DEFERRED` 不立即调用，而是把 Callable 推进 MessageQueue；普通连接则直接 `callable.callp()`。相关代码在 `core/object/object.cpp:1224`、`1262`、`1287`。"
      },
      {
        type: "paragraph",
        text: "对象析构时也会清理信号。`Object::~Object()` 在 `core/object/object.cpp:2320` 附近先清理本对象发出的所有连接，再遍历 `connections` 断开所有连到本对象的外部信号，最后从 ObjectDB 移除实例。这样信号系统不会长期保存已释放对象的连接记录。"
      },
      {
        type: "table",
        title: "ConnectFlags 的效果",
        headers: ["flag", "源码值", "作用"],
        rows: [
          ["`CONNECT_DEFERRED`", "`1`", "信号发射时不立即调用，而是推入 MessageQueue。"],
          ["`CONNECT_PERSIST`", "`2`", "提示场景保存系统持久化连接，例如 PackedScene 保存。"],
          ["`CONNECT_ONE_SHOT`", "`4`", "触发一次后自动断开。"],
          ["`CONNECT_REFERENCE_COUNTED`", "`8`", "允许重复 connect 同一 Callable，通过 reference_count 计数。"],
          ["`CONNECT_APPEND_SOURCE_OBJECT`", "`16`", "发射时把源对象插入到参数列表中。"],
          ["`CONNECT_INHERITED`", "`32`", "编辑器构建使用，标记继承场景里的连接来源。"]
        ]
      },
      {
        type: "flow",
        title: "`button.pressed.emit()` 的运行路径",
        steps: [
          { title: "Signal 保存 Button 的 ObjectID 和 `pressed`", text: "Signal 只是代理，不保存连接列表。" },
          { title: "`Signal::emit()`", text: "通过 ObjectDB 找回 Button。" },
          { title: "`Object::emit_signalp(\"pressed\")`", text: "进入源对象的 signal_map。" },
          { title: "复制 slot 列表", text: "防止回调中 disconnect 或释放对象破坏遍历。" },
          { title: "按 flags 处理", text: "one-shot、append source、deferred 等逻辑在这里生效。" },
          { title: "`Callable::callp()`", text: "最终调用目标对象方法、脚本函数或自定义 Callable。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：GDScript 里连接信号"
      },
      {
        type: "code",
        code: [
          "func _ready():",
          "    $Button.pressed.connect(Callable(self, \"_on_button_pressed\"))",
          "",
          "func _on_button_pressed():",
          "    print(\"pressed\")"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这段脚本会创建一个标准 Callable：对象是 `self` 的 ObjectID，方法名是 `_on_button_pressed`。`pressed.connect()` 通过 Signal 代理进入 Button 的 `Object::connect()`，连接表保存在 Button 这个源对象上。"
      },
      {
        type: "subheading",
        title: "案例二：C++ 里用 `callable_mp` 连接成员函数"
      },
      {
        type: "code",
        code: [
          "button->connect(",
          "    SNAME(\"pressed\"),",
          "    callable_mp(this, &MyPanel::_on_pressed),",
          "    Object::CONNECT_DEFERRED",
          ");"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`callable_mp` 定义在 `core/object/callable_mp.h`，会创建 `CallableCustomMethodPointer`。它保存实例指针、ObjectID 和成员函数指针；调用时先用 ObjectDB 确认对象还活着，再用 `call_with_variant_args` 执行 C++ 方法。"
      },
      {
        type: "subheading",
        title: "案例三：绑定额外参数"
      },
      {
        type: "code",
        code: [
          "func _ready():",
          "    for slot in inventory_slots:",
          "        slot.pressed.connect(_on_slot_pressed.bind(slot.index))",
          "",
          "func _on_slot_pressed(index):",
          "    select_slot(index)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`bind(slot.index)` 创建的是 CallableCustomBind。信号本身没有参数时，绑定值会追加到调用参数后面，所以 `_on_slot_pressed(index)` 仍能收到 index。多层 bind/unbind 的细节可以看 `tests/core/variant/test_callable.cpp:172`。"
      },
      {
        type: "subheading",
        title: "案例四：一次性连接和延迟连接"
      },
      {
        type: "code",
        code: [
          "node.connect(",
          "    \"ready\",",
          "    Callable(self, \"_after_ready\"),",
          "    Object.CONNECT_ONE_SHOT | Object.CONNECT_DEFERRED",
          ")"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`CONNECT_ONE_SHOT` 会在发射时先断开连接，避免递归或重复触发；`CONNECT_DEFERRED` 会把 Callable 和参数推入 MessageQueue。两者一起用时，就是“触发一次，并且稍后安全点调用”。"
      },
      {
        type: "subheading",
        title: "案例五：PackedScene 为什么能保存部分信号连接"
      },
      {
        type: "code",
        code: [
          "// tests/scene/test_packed_scene.cpp:82",
          "sub_node->connect(",
          "    \"ready\",",
          "    callable_mp(main_scene_root, &Node::is_ready),",
          "    Object::CONNECT_PERSIST",
          ");"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`CONNECT_PERSIST` 是场景保存系统识别连接的提示。测试 `tests/scene/test_packed_scene.cpp:59` 验证了带持久化 flag 的连接会进入 PackedScene 的 SceneState。"
      },
      {
        type: "heading",
        title: "Callable / Signal 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Callable", "保存可调用目标：对象方法、自定义调用体、绑定参数、RPC 入口。", "不保存某个信号的所有连接槽。"],
          ["Signal", "保存某个 Object 的某个信号名，并代理 connect/emit/disconnect。", "不是全局事件总线，也不拥有连接列表。"],
          ["Object", "保存 signal_map、connections、block 状态，并执行 connect/emit/disconnect。", "不是每个 Callable 的强引用所有者。"],
          ["ObjectDB", "用 ObjectID 找回对象，帮助 Callable/Signal 检查目标是否还存在。", "不阻止对象被释放。"],
          ["Variant", "承载信号参数、Callable、Signal 值。", "不决定何时调用。"],
          ["MessageQueue", "延迟执行 Callable 或 Object call/set/notification。", "不验证信号是否存在，也不保存持久连接元数据。"],
          ["MethodBind", "执行对象方法的 C++ 绑定入口。", "不表示“稍后调用这个目标”；这是 Callable 的职责。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Signal 是全局事件中心。Signal 总是绑定到某个 ObjectID 和信号名。",
          "误区二：Callable 会让目标对象不被释放。标准 Callable 保存 ObjectID，不是强引用；调用前要能从 ObjectDB 找回对象。",
          "误区三：Signal 对象保存连接列表。真正的连接表在源 Object 的 `signal_map`，目标 Object 只保存反向连接列表。",
          "误区四：`bind()` 修改原 Callable。它返回新的 CallableCustom 包装器，原 Callable 不变。",
          "误区五：连接时带不同 bind 参数就一定是不同连接。Object::connect 用 base comparator 比较，bind 层可能被忽略以避免重复连接。",
          "误区六：`CONNECT_DEFERRED` 只是稍微晚一点同步调用。它会进入 MessageQueue，具体执行时机取决于队列 flush。",
          "误区七：对象析构时信号连接会自然无害地留着。Object 析构会主动断开双向连接；如果在发射中释放对象，源码还会给出错误提示。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/variant/callable.h:48` 的 Callable 字段和 `CallError`，确认标准 Callable 与 custom Callable 的区别。",
          "读 `Callable::callp()`、`call_deferredp()`、`is_valid()`，理解 ObjectDB、Object::callp 和 MessageQueue 如何接入。",
          "读 `CallableCustom`，再读 `callable_bind.cpp` 和 `callable_mp.h`，理解 bind/unbind 与 C++ 方法指针 Callable。",
          "读 `core/variant/callable.h:178` 的 Signal，确认它只是 ObjectID + 信号名代理。",
          "读 `Object::Connection`、`SignalData`、`signal_map`、`connections`，理解连接表的真实存储。",
          "最后跟 `Object::connect()`、`emit_signalp()`、`_disconnect()` 和 Object 析构，确认连接、发射、断开、清理的完整生命周期。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Callable 是 Godot 对“可调用目标”的统一值表示，Signal 是对象信号的轻量代理；真正让它们可靠工作的，是 ObjectDB 校验、Object 的双向连接表、Variant 参数和 MessageQueue 延迟执行一起组成的运行时链路。"
      }
    ]
  },
  {
    id: "objectdb",
    title: "ObjectDB",
    aliases: ["ObjectDB", "ObjectID", "对象数据库", "对象索引", "实例 ID", "get_instance", "add_instance", "remove_instance", "queue_delete"],
    summary: "Godot 的全局对象弱索引：每个 Object 构造时登记为 ObjectID，释放时注销；Callable、Signal、Variant、MessageQueue 和 SceneTree 删除队列用它把 ID 安全地查回当前仍存在的 Object。",
    article: [
      {
        type: "lead",
        text: "ObjectDB 是 Godot 给所有 Object 实例建立的一张全局弱索引表。它让系统可以把 `ObjectID` 重新查回 `Object *`，并能发现“这个旧 ID 对应的对象已经释放或槽位已复用”。它不是所有权系统，不负责延长对象生命，只负责登记、查找和失效判断。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 ObjectDB 想成前台登记簿。每个 Object 出生时拿一个号码牌，也就是 `ObjectID`；别人不直接拿它的地址，而是拿号码牌来前台问：“这个对象现在还在吗？”如果还在，前台给出当前指针；如果已经走了，前台返回空。"
      },
      {
        type: "paragraph",
        text: "关键点是：号码牌不是护身符。拿着 ObjectID 不能阻止对象被释放，也不能说明对象一定还活着。它只是比裸指针安全，因为每次使用前都能重新查一下。"
      },
      {
        type: "flow",
        title: "从使用者视角看 ObjectDB",
        steps: [
          { title: "Object 构造", text: "`Object::_construct_object()` 调 `ObjectDB::add_instance(this)`。" },
          { title: "得到 ObjectID", text: "Object 把返回值存进 `_instance_id`。" },
          { title: "系统保存 ID", text: "Callable、Signal、Variant、MessageQueue、SceneTree 删除队列都可能保存 ObjectID。" },
          { title: "使用前重新查", text: "`ObjectDB::get_instance(id)` 解码 slot 和 validator。" },
          { title: "对象释放时注销", text: "`Object::~Object()` 调 `ObjectDB::remove_instance(this)`，旧 ID 失效。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`ObjectID` 定义在 `core/object/object_id.h:41`。它本质上包装一个 `uint64_t id`，支持 `is_ref_counted()`、`is_valid()`、`is_null()`，并且能显式转成 64 位整数。注释说明它要兼容 Variant 使用的 int64，同时避免意外损失精度。"
      },
      {
        type: "paragraph",
        text: "`ObjectDB` 定义在 `core/object/object.h:874`。ID 的位布局由几个宏决定：低 24 位是 slot index，接着 39 位是 validator，最高 1 位是 ref-counted 标记。`ObjectSlot` 在 `core/object/object.h:882`，每个槽有 `validator`、`next_free`、`is_ref_counted` 和 `Object *object`。"
      },
      {
        type: "paragraph",
        text: "Object 构造时会登记。`Object::_construct_object()` 在 `core/object/object.cpp:2283`，它设置 `_block_signals`、`_is_queued_for_deletion`、`_ancestry` 等初始状态，然后在 `core/object/object.cpp:2294` 调 `ObjectDB::add_instance(this)`，把返回的 ObjectID 存进 `_instance_id`。"
      },
      {
        type: "paragraph",
        text: "`ObjectDB::add_instance()` 在 `core/object/object.cpp:2450`。它加自旋锁，必要时扩容 `object_slots`，从空闲槽拿一个 slot，写入对象指针和 `is_ref_counted` 标记，递增 `validator_counter`，把 validator 和 slot 编进 64 位 ID；如果对象是 RefCounted，还会设置最高位的 reference bit。"
      },
      {
        type: "paragraph",
        text: "`ObjectDB::get_instance()` 是内联函数，入口在 `core/object/object.h:908`。它从 ObjectID 取低 24 位得到 slot，再取 validator；如果槽中的 validator 和 ID 中的不一致，就返回 nullptr。这个 validator 是防旧 ID 误指向新对象的关键：槽位可能被复用，但 validator 不同，旧 ID 就不会查到新对象。"
      },
      {
        type: "paragraph",
        text: "`ObjectDB::remove_instance()` 在 `core/object/object.cpp:2494`。它根据对象当前 `_instance_id` 找到 slot，在调试构建中检查槽里的 object 和 validator 是否匹配，然后把 slot 放回空闲栈，清空 validator、is_ref_counted 和 object。这样旧 ObjectID 后续查找会失败。"
      },
      {
        type: "table",
        title: "ObjectID 的位布局",
        headers: ["部分", "位数", "作用"],
        rows: [
          ["slot index", "24 位", "定位 `object_slots` 数组里的槽位，最大槽数由 `OBJECTDB_SLOT_MAX_COUNT_BITS` 控制。"],
          ["validator", "39 位", "检测旧 ID 是否还匹配当前槽位，防止槽位复用后误查到新对象。"],
          ["reference bit", "1 位", "标记这个 ObjectID 对应的对象是不是 RefCounted，`ObjectID::is_ref_counted()` 会读取它。"],
          ["总计", "64 位", "能放进 Variant 的 int64/uint64 表示里，用于跨系统传递弱对象身份。"]
        ]
      },
      {
        type: "table",
        title: "ObjectDB 的核心状态",
        headers: ["状态", "源码入口", "作用"],
        rows: [
          ["`object_slots`", "`core/object/object.cpp:2443`", "全局槽数组，每个槽保存当前对象指针和 validator。"],
          ["`slot_count`", "`core/object/object.cpp:2441`", "当前活跃对象数量，也是空闲栈的边界。"],
          ["`slot_max`", "`core/object/object.cpp:2442`", "已分配槽容量，满时扩容。"],
          ["`validator_counter`", "`core/object/object.cpp:2444`", "生成新的 validator，避免旧 ID 命中新对象。"],
          ["`spin_lock`", "`core/object/object.cpp:2440`", "保护 add/remove/get_instance 对全局槽表的访问。"],
          ["`debug_objects()`", "`core/object/object.cpp:2382`", "调试和退出时遍历仍存活对象。"]
        ]
      },
      {
        type: "heading",
        title: "ObjectDB 不是所有权系统"
      },
      {
        type: "paragraph",
        text: "ObjectDB 最容易被误解成“对象管理器”。它确实知道对象在哪里，但它不决定对象什么时候释放。普通 Object 仍然由创建者 `memdelete` 或具体 API 释放；Node 常走场景树和 `queue_free()`；Resource/RefCounted 走引用计数。ObjectDB 只在对象构造和析构时更新索引。"
      },
      {
        type: "paragraph",
        text: "这个区别解释了很多 bug：保存 ObjectID 比保存裸指针安全，但它不是强引用。跨帧保存对象时，每次使用前都要 `ObjectDB::get_instance(id)`；如果返回 nullptr，就说明对象已释放或 ID 已失效。`tests/core/object/test_object.cpp:200` 验证了构造后能用 instance id 查回同一个对象，`tests/core/object/test_object.cpp:652` 验证了对象自毁后 ObjectDB 查找返回 nullptr。"
      },
      {
        type: "flow",
        title: "为什么旧 ObjectID 不会误指新对象",
        steps: [
          { title: "旧对象 A 使用 slot 7", text: "ObjectID 里记录 slot=7、validator=100。" },
          { title: "A 释放", text: "`remove_instance()` 清空 slot 7 的 validator 和 object。" },
          { title: "新对象 B 复用 slot 7", text: "`add_instance()` 写入新的 validator，例如 101。" },
          { title: "拿旧 ID 查询", text: "`get_instance()` 取出 slot=7，但比较 validator=100。" },
          { title: "validator 不匹配", text: "返回 nullptr，而不是把 B 当成 A。" }
        ]
      },
      {
        type: "heading",
        title: "哪些系统依赖 ObjectDB"
      },
      {
        type: "paragraph",
        text: "Callable 依赖 ObjectDB。`Callable::callp()` 在 `core/variant/callable.cpp:44`，标准 Callable 会从保存的 ObjectID 查回 Object，再调用 `Object::callp()`；`Callable::is_valid()` 也会用 `get_object()` 和 `has_method()` 判断目标是否还可调用。Signal 也一样，`Signal::emit()` 在 `core/variant/callable.cpp:534`，会先用 ObjectDB 找回源对象。"
      },
      {
        type: "paragraph",
        text: "Variant 的 OBJECT 类型也依赖 ObjectDB。`Variant::get_validated_object()` 在 `core/variant/variant.cpp:2043`，它用 Variant 保存的 ObjectID 调 `ObjectDB::get_instance()`。相比 `operator Object*()` 直接返回缓存指针，validated 路径更适合检查普通 Object 是否已经释放。"
      },
      {
        type: "paragraph",
        text: "SceneTree 删除队列保存的是 ObjectID。`SceneTree::queue_delete()` 在 `scene/main/scene_tree.cpp:1637`，把对象标记 `_is_queued_for_deletion = true`，再把 `get_instance_id()` 推入 `delete_queue`。真正删除时 `_flush_delete_queue()` 在 `scene/main/scene_tree.cpp:1624`，先用 ObjectDB 查对象；如果已经不存在，就跳过，避免重复删除。"
      },
      {
        type: "paragraph",
        text: "MessageQueue 也通过 ObjectID 避免悬空调用。`CallQueue::push_callp(ObjectID, ...)` 在 `core/object/message_queue.cpp:68` 把 ID 包成 Callable；flush 时 `core/object/message_queue.cpp:258` 先取 `message->callable.get_object()`，只有目标对象仍存在才执行 call/set/notification。"
      },
      {
        type: "table",
        title: "ObjectDB 的典型使用者",
        headers: ["使用者", "保存什么", "为什么需要 ObjectDB"],
        rows: [
          ["Callable", "ObjectID + 方法名，或 custom callable 里的 ObjectID。", "调用前重新找对象，目标释放后 Callable 可失效。"],
          ["Signal", "源对象 ObjectID + 信号名。", "connect/emit/disconnect 前找回源对象。"],
          ["Variant::OBJECT", "ObjData 里有 ObjectID 和缓存指针。", "需要 validated object 时按 ID 查当前对象。"],
          ["MessageQueue", "延迟调用、set、notification 的目标 ObjectID。", "flush 时对象可能已经释放，必须重新检查。"],
          ["SceneTree delete_queue", "准备稍后删除的 ObjectID。", "真正删除前查对象，避免对象已被其他路径释放。"],
          ["调试/退出清理", "遍历 ObjectDB 剩余槽。", "报告泄漏对象和对象数量。"]
        ]
      },
      {
        type: "flow",
        title: "`queue_free()` 背后的 ObjectID 路径",
        steps: [
          { title: "Node 请求稍后释放", text: "`queue_free()` 最终进入 SceneTree 删除队列。" },
          { title: "`queue_delete(object)`", text: "设置 `_is_queued_for_deletion`，保存 `object->get_instance_id()`。" },
          { title: "当前阶段继续运行", text: "对象不会立刻 memdelete，避免树遍历或信号回调中破坏结构。" },
          { title: "`_flush_delete_queue()`", text: "安全点取出 ObjectID，调用 `ObjectDB::get_instance()`。" },
          { title: "对象仍存在才删除", text: "查到对象则 `memdelete(obj)`；查不到说明已释放，跳过。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：保存 ID，使用前重新查"
      },
      {
        type: "code",
        code: [
          "ObjectID id = object->get_instance_id();",
          "",
          "// 过了一帧，object 可能已经释放。",
          "if (Object *obj = ObjectDB::get_instance(id)) {",
          "    obj->call(\"do_work\");",
          "} else {",
          "    // 对象已经不存在，不要使用旧指针。",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这就是 ObjectID 的正确用法：把它当弱句柄，每次使用前重新查。不要把 `id.is_valid()` 当成对象仍然存在的证明；它只能说明 ID 非零。"
      },
      {
        type: "subheading",
        title: "案例二：Variant 里的 Object 安全读取"
      },
      {
        type: "code",
        code: [
          "Variant v = some_object;",
          "",
          "Object *raw = v; // 只是取缓存指针。",
          "Object *checked = v.get_validated_object(); // 通过 ObjectDB 重新查。",
          "",
          "if (checked) {",
          "    checked->call(\"safe_method\");",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`Variant::operator Object*()` 和 `get_validated_object()` 的差异，本质上就是“直接拿缓存指针”与“用 ObjectID 问 ObjectDB”。普通 Object 跨帧使用时优先考虑验证路径。"
      },
      {
        type: "subheading",
        title: "案例三：延迟调用不会解引用旧对象"
      },
      {
        type: "code",
        code: [
          "ObjectID id = target->get_instance_id();",
          "MessageQueue::get_singleton()->push_callp(id, \"refresh\", nullptr, 0, true);",
          "",
          "// 如果 target 在 flush 前已经释放，MessageQueue 会查不到对象并跳过调用。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "MessageQueue 不把裸指针长期塞进队列，而是用 ObjectID/Callable。flush 时查不到对象就不调用，避免延迟队列成为悬空指针来源。"
      },
      {
        type: "subheading",
        title: "案例四：测试里的基本保证"
      },
      {
        type: "code",
        code: [
          "// tests/core/object/test_object.cpp:200",
          "Object object;",
          "Object *p_db = ObjectDB::get_instance(object.get_instance_id());",
          "CHECK(p_db == &object);",
          "",
          "// tests/core/object/test_object.cpp:652",
          "CHECK(ObjectDB::get_instance(obj_id) == nullptr);"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "第一段证明构造后的对象能通过 ObjectID 查回；第二段证明对象在调用链中自毁后，ObjectDB 不再返回旧对象。"
      },
      {
        type: "heading",
        title: "ObjectDB 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "具体对象实例，保存 `_instance_id`，构造/析构时登记注销。", "不保存全局对象槽数组。"],
          ["ObjectID", "可跨系统传递的 64 位弱句柄。", "不阻止对象释放，也不保证对象仍存在。"],
          ["ObjectDB", "ObjectID 到 Object 指针的当前查找和失效校验。", "不拥有对象，不决定释放策略。"],
          ["RefCounted/Ref", "强引用计数所有权。", "不替所有 Object 都提供生命周期管理。"],
          ["Callable/Signal", "保存目标 ObjectID，用于稍后调用或发射。", "不自己验证槽位和 validator；验证靠 ObjectDB。"],
          ["SceneTree", "管理 Node 树和删除队列。", "不是全局 Object 索引；删除队列也要回查 ObjectDB。"],
          ["MessageQueue", "延迟执行 call/set/notification。", "不持有目标对象强引用。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：ObjectDB 管对象生命周期。它只登记和查找，不负责拥有或释放对象。",
          "误区二：ObjectID 是强引用。ObjectID 只是弱句柄，不能防止对象被 memdelete、queue_free 或引用计数释放。",
          "误区三：`ObjectID::is_valid()` 等于对象活着。它只表示 ID 非零，真正可用要 `ObjectDB::get_instance(id)`。",
          "误区四：旧 ID 可能查到复用槽位的新对象。validator 的目的就是防止这种误命中。",
          "误区五：保存 Object* 和保存 ObjectID 一样安全。裸指针释放后无法自证失效，ObjectID 至少可以重新查。",
          "误区六：queue_free 立即删除对象。它通常进入 SceneTree 删除队列，稍后用 ObjectID 回查后才删除。",
          "误区七：RefCounted 不需要 ObjectDB。RefCounted 也继承 Object，也有 ObjectID；只是它的生命周期由引用计数决定。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/object/object_id.h:41`，确认 ObjectID 只是 64 位 ID 包装。",
          "读 `core/object/object.h:874` 到 `ObjectSlot`，理解 slot、validator、reference bit 的布局。",
          "读 `Object::_construct_object()`，看 Object 如何在构造时进入 ObjectDB。",
          "读 `ObjectDB::add_instance()` 和 `remove_instance()`，理解 ID 编码和失效过程。",
          "读 `ObjectDB::get_instance()`，确认查找为什么要比较 validator。",
          "最后跟 Callable、Signal、Variant、MessageQueue、SceneTree delete_queue 的使用点，理解 ObjectDB 是弱索引而不是所有权系统。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "ObjectDB 是 Godot 的全局弱对象索引：它让 ObjectID 能安全地重新查当前对象，并用 validator 防止旧 ID 命中新对象；但对象是否活着、何时释放，仍由 Object 子类自己的生命周期规则决定。"
      }
    ]
  },
  {
    id: "refcounted",
    title: "RefCounted",
    aliases: ["RefCounted", "Ref", "Ref<T>", "引用计数", "WeakRef", "init_ref", "reference", "unreference", "get_reference_count"],
    summary: "Godot 的引用计数对象基类和智能引用包装：Resource 等对象继承 RefCounted，通常由 Ref<T> 持有；最后一个强引用释放时，unreference() 返回 true，调用方删除对象。",
    article: [
      {
        type: "lead",
        text: "RefCounted 是 Godot 给 Resource、Image、Animation、InputEvent 等对象使用的引用计数生命周期模型。它仍然是 Object，仍然有 ObjectID，也能进 ClassDB；不同点是它通常不靠手动 free 或场景树删除，而是靠 `Ref<T>` 强引用计数决定什么时候释放。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 RefCounted 想成一份共享资料。每个人拿一张借阅卡，也就是 `Ref<T>`；借阅卡数量就是引用计数。还有人拿着时资料不能销毁；最后一张借阅卡还掉时，资料就自动清理。"
      },
      {
        type: "paragraph",
        text: "这和 Node 很不一样。Node 通常挂在 SceneTree 上，用 `queue_free()` 或树结构管理生命周期；Resource 通常继承 RefCounted，用 `Ref<Resource>` 管生命周期。把 Resource 当 Node 加到树里，或者把 Node 当 Ref 保存，都会把两套生命周期模型混在一起。"
      },
      {
        type: "flow",
        title: "从使用者视角看 RefCounted",
        steps: [
          { title: "创建 RefCounted 对象", text: "例如 `Ref<Image> img; img.instantiate();` 或 `Ref<Animation> anim = memnew(Animation);`。" },
          { title: "Ref<T> 持有对象", text: "第一次持有走 `init_ref()`，复制 Ref 走 `reference()`。" },
          { title: "多个 Ref 共享同一个对象", text: "复制 Ref 不复制 Resource 内容，只增加引用计数。" },
          { title: "Ref 离开作用域或重置", text: "`Ref<T>::unref()` 调 `unreference()`。" },
          { title: "引用计数归零", text: "`unreference()` 返回 true，调用方 `memdelete` 对象。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "源码入口是 `core/object/ref_counted.h:36` 的 `class RefCounted : public Object`。它有三个计数相关成员：`SafeRefCount refcount`、`SafeRefCount refcount_init` 和 `SafeNumeric<uint32_t> dereference_count`。`static_ancestral_class` 被设置为 `AncestralClass::REF_COUNTED`，构造函数在 `core/object/ref_counted.cpp:129` 调 `Object(true)`，让 ObjectDB 登记时知道这是 RefCounted 对象。"
      },
      {
        type: "paragraph",
        text: "RefCounted 暴露的核心接口在 `core/object/ref_counted.h:49` 到 `53`：`init_ref()`、`deinit_ref()`、`reference()`、`unreference()`、`get_reference_count()`。这些方法也在 `_bind_methods()` 里暴露给脚本，绑定入口是 `core/object/ref_counted.cpp:63`。"
      },
      {
        type: "paragraph",
        text: "`reference()` 在 `core/object/ref_counted.cpp:74`。它尝试增加 `refcount`，如果计数已经是 0，增加失败并返回 false。低引用计数阶段还会通知脚本实例、GDExtension 和 instance binding，让外部语言绑定也能同步引用计数变化。"
      },
      {
        type: "paragraph",
        text: "`unreference()` 在 `core/object/ref_counted.cpp:92`。它减少引用计数，判断是否归零；如果归零，还会询问脚本实例、GDExtension 和 instance binding 是否允许销毁。最后它返回 `die`，但不直接删除对象。真正 `memdelete` 的地方通常在 `Ref<T>::unref()` 或 Variant/Object 持有者里。"
      },
      {
        type: "paragraph",
        text: "`Ref<T>` 是常用的强引用包装器，定义在 `core/object/ref_counted.h:58`。它内部只有一个 `T *reference`。复制 Ref 会调用 `reference()`，从裸指针接管会调用 `init_ref()`，析构时 `~Ref()` 会进入 `unref()`；如果 `unreference()` 返回 true，就 `memdelete` 对象。"
      },
      {
        type: "paragraph",
        text: "Resource 是 RefCounted 最重要的子类之一。`Resource : public RefCounted` 定义在 `core/io/resource.h:52`，Animation、Image、Material、Script、PackedScene 等大量资源都沿用这套模型。所以资源加载、缓存、复制和编辑器属性里的资源引用，本质上都要理解 `Ref<T>` 的强引用语义。"
      },
      {
        type: "table",
        title: "RefCounted 和 Ref<T> 的职责",
        headers: ["组件", "负责什么", "源码入口"],
        rows: [
          ["`RefCounted`", "对象基类，持有引用计数并提供 reference/unreference 接口。", "`core/object/ref_counted.h:36`"],
          ["`Ref<T>`", "强引用包装器，复制时加引用，析构时减引用并可能删除对象。", "`core/object/ref_counted.h:58`"],
          ["`refcount`", "当前强引用数量，归零时对象可以销毁。", "`core/object/ref_counted.h:38`"],
          ["`refcount_init`", "辅助处理第一次引用接管，避免重复初始化引用。", "`core/object/ref_counted.h:39`、`ref_counted.cpp:36`"],
          ["`dereference_count`", "在 unreference 临界区里避免并发销毁时仍有人执行引用回调。", "`core/object/ref_counted.h:40`、`ref_counted.cpp:92`"],
          ["`WeakRef`", "弱引用对象，只保存 ObjectID，不增加引用计数。", "`core/object/ref_counted.h:241`"]
        ]
      },
      {
        type: "table",
        title: "三种常见生命周期模型对比",
        headers: ["模型", "典型对象", "释放方式", "常见用途"],
        rows: [
          ["普通 Object", "`Object` 或内部临时对象。", "创建者手动 `memdelete` 或特定 API 释放。", "底层工具对象、临时反射对象。"],
          ["Node/SceneTree", "`Node`、`Control`、`Node2D`、`Node3D`。", "常用 `queue_free()`，由树和删除队列处理。", "场景树里的运行时实体。"],
          ["RefCounted/Ref", "`Resource`、`Image`、`Animation`、`InputEvent`。", "最后一个 `Ref<T>` 释放时删除。", "可共享、可缓存、可序列化的数据资源。"],
          ["WeakRef", "`WeakRef` 对象。", "不拥有目标，只能尝试取回。", "跨帧弱引用、避免循环引用。"]
        ]
      },
      {
        type: "heading",
        title: "Ref<T> 是怎样持有对象的"
      },
      {
        type: "paragraph",
        text: "`Ref<T>::operator=(T *p_from)` 在 `core/object/ref_counted.h:161`，会调用 `ref_pointer<true>(p_from)`。模板参数 `Init=true` 表示这是从裸指针接管，内部走 `reference->init_ref()`。而从另一个 Ref 复制时走 `ref_pointer<false>`，内部调用 `reference()` 增加已有强引用。"
      },
      {
        type: "paragraph",
        text: "`Ref<T>::unref()` 在 `core/object/ref_counted.h:198`。它调用 `reinterpret_cast<RefCounted *>(reference)->unreference()`；如果返回 true，就 `memdelete`。随后把自己的 `reference` 置空。`Ref<T>` 的析构函数在 `core/object/ref_counted.h:228`，只是调用 `unref()`。"
      },
      {
        type: "paragraph",
        text: "`Ref<T>::instantiate()` 在 `core/object/ref_counted.h:216`，会 `memnew(T(...))` 创建对象并用临时 Ref 接管，再和当前 Ref 交换。Godot 还用 `memnew_result` 特化让 `memnew(RefCounted 子类)` 更自然地返回 `Ref<T>` 形态，入口在 `core/object/ref_counted.h:232`。"
      },
      {
        type: "flow",
        title: "`Ref<Image> img; img.instantiate()` 的路径",
        steps: [
          { title: "创建 Image", text: "`memnew(Image)` 构造 RefCounted 对象并进入 ObjectDB。" },
          { title: "临时 Ref 接管", text: "`Ref<T>` 从裸指针赋值，调用 `init_ref()`。" },
          { title: "当前 Ref 获得指针", text: "`instantiate()` 用 SWAP 把新引用放进 `img`。" },
          { title: "复制 Ref", text: "其他 Ref 复制 `img` 时调用 `reference()` 增加计数。" },
          { title: "最后一个 Ref 释放", text: "`unreference()` 返回 true，`Ref<T>::unref()` 删除对象。" }
        ]
      },
      {
        type: "heading",
        title: "Variant 和 ObjectDB 里的 RefCounted"
      },
      {
        type: "paragraph",
        text: "RefCounted 仍然是 Object，所以它也有 ObjectID。`ObjectDB::add_instance()` 会把 `is_ref_counted` 写进 ObjectID 的最高位；`ObjectID::is_ref_counted()` 定义在 `core/object/object_id.h:46`。这让 Variant 和 ObjectDB 可以快速知道一个对象 ID 是否属于引用计数对象。"
      },
      {
        type: "paragraph",
        text: "Variant 保存 Object 时也要处理 RefCounted。`Variant::ObjData::ref_pointer()` 在 `core/variant/variant.cpp:1094`，如果对象 `is_ref_counted()`，会调用 `init_ref()`；复制 OBJECT Variant 时 `ObjData::ref()` 在 `core/variant/variant.cpp:1074` 会调用 `reference()`；清理时 `ObjData::unref()` 在 `core/variant/variant.cpp:1117` 会调用 `unreference()`，必要时 `memdelete`。"
      },
      {
        type: "paragraph",
        text: "`ObjectDB::get_ref<T>()` 定义在 `core/object/ref_counted.h:260`，它先用 ObjectDB 查对象，再构造成 `Ref<T>`。这和 `get_instance<T>()` 的区别是：`get_ref<T>()` 返回强引用，成功时会增加引用计数。"
      },
      {
        type: "flow",
        title: "Variant 保存 RefCounted 对象的路径",
        steps: [
          { title: "`Variant v = resource`", text: "Variant 的 OBJECT 类型保存 ObjectID 和 Object 指针。" },
          { title: "发现是 RefCounted", text: "`ObjData::ref_pointer()` 调 `init_ref()`。" },
          { title: "复制 Variant", text: "`ObjData::ref()` 调 `reference()` 增加强引用。" },
          { title: "清理 Variant", text: "`ObjData::unref()` 调 `unreference()`。" },
          { title: "计数归零才删除", text: "如果没有其他 Ref/Variant 持有，最终 `memdelete`。" }
        ]
      },
      {
        type: "heading",
        title: "WeakRef 和循环引用"
      },
      {
        type: "paragraph",
        text: "`WeakRef` 定义在 `core/object/ref_counted.h:241`，它自己是 RefCounted，但保存目标时只保存 `ObjectID ref`。`WeakRef::get_ref()` 在 `core/object/ref_counted.cpp:137`，用 ObjectDB 重新查目标；如果目标是 RefCounted，就返回 `Ref<RefCounted>`，否则返回 Object；查不到就返回空 Variant。"
      },
      {
        type: "paragraph",
        text: "引用计数解决不了循环引用。如果 A 用 Ref 持有 B，B 又用 Ref 持有 A，即使外部都释放了，两个对象的引用计数也不会归零。需要弱引用、显式断开或重新设计所有权。WeakRef 的意义就在于“记住目标，但不增加引用计数”。"
      },
      {
        type: "table",
        title: "强引用和弱引用",
        headers: ["引用方式", "是否增加引用计数", "目标释放后怎么办", "适合场景"],
        rows: [
          ["`Ref<T>`", "是。复制、赋值会增加强引用。", "最后一个 Ref 释放时目标被删除。", "资源、图片、动画、脚本等共享数据。"],
          ["`Variant` 保存 RefCounted Object", "是。Object Variant 会按 RefCounted 规则增减引用。", "Variant 清理时减少引用。", "脚本参数、属性值、容器元素。"],
          ["`WeakRef`", "否。只保存 ObjectID。", "`get_ref()` 查不到就返回空。", "避免循环引用或跨帧弱观察。"],
          ["裸 `Object *`", "否。", "对象释放后指针悬空。", "短生命周期、调用栈内确定安全的对象。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：创建和共享 Resource"
      },
      {
        type: "code",
        code: [
          "Ref<Image> image;",
          "image.instantiate();",
          "",
          "Ref<Image> same_image = image; // 增加引用计数，不复制像素数据。",
          "image.unref();               // same_image 仍然持有对象。",
          "",
          "same_image.unref();          // 最后一个 Ref 释放后对象可被删除。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "复制 `Ref<Image>` 只是共享同一个 Image 对象。要复制资源内容，需要资源自己的 duplicate/copy API，而不是简单复制 Ref。"
      },
      {
        type: "subheading",
        title: "案例二：RefCounted 不能像 Node 一样 queue_free"
      },
      {
        type: "code",
        code: [
          "Ref<Resource> texture = ResourceLoader::load(\"res://icon.png\");",
          "",
          "// 正确：让 Ref 离开作用域或调用 unref()。",
          "texture.unref();",
          "",
          "// 错误心智：把 Resource 当 Node 挂树或 queue_free。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Resource 是数据对象，不是场景树节点。它的生命周期通常由 Ref 和 ResourceCache 共同影响，而不是由 SceneTree 删除队列管理。"
      },
      {
        type: "subheading",
        title: "案例三：从 ObjectID 取回强引用"
      },
      {
        type: "code",
        code: [
          "ObjectID id = resource->get_instance_id();",
          "Ref<Resource> strong = ObjectDB::get_ref<Resource>(id);",
          "",
          "if (strong.is_valid()) {",
          "    // strong 会增加引用计数，使用期间对象不会消失。",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`get_ref<T>()` 适合你需要从 ObjectID 转成强引用的场景。普通 `get_instance<T>()` 只返回裸指针，不会增加引用计数。"
      },
      {
        type: "subheading",
        title: "案例四：WeakRef 不拥有目标"
      },
      {
        type: "code",
        code: [
          "Ref<WeakRef> weak;",
          "weak.instantiate();",
          "weak->set_ref(resource);",
          "",
          "Variant maybe_alive = weak->get_ref();",
          "if (maybe_alive.get_type() != Variant::NIL) {",
          "    Ref<RefCounted> alive = maybe_alive;",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "WeakRef 只保存 ObjectID。目标还活着时可以重新取回，目标释放后返回空。它不会延长目标生命周期。"
      },
      {
        type: "heading",
        title: "RefCounted 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "对象协议、ObjectID、属性/方法/信号。", "不为所有对象提供引用计数生命周期。"],
          ["ObjectDB", "弱索引，按 ObjectID 查当前对象。", "不增加引用计数，也不拥有对象。"],
          ["RefCounted", "在对象内部维护引用计数和外部绑定回调。", "不自动防止循环引用。"],
          ["Ref<T>", "强引用包装器，负责调用 reference/unreference 并可能 memdelete。", "不复制 Resource 内容。"],
          ["Resource", "可保存、可加载、常被缓存的 RefCounted 数据对象。", "不是场景树节点。"],
          ["WeakRef", "弱引用 ObjectID。", "不保证目标仍存在，也不增加引用计数。"],
          ["Variant", "可装 RefCounted Object，并按引用计数规则持有。", "不是资源内容复制工具。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：RefCounted 不是 Object。它继承 Object，仍然有 ObjectID、ClassDB、信号和脚本实例。",
          "误区二：Ref<T> 复制会复制资源内容。它只复制强引用，多个 Ref 指向同一个对象。",
          "误区三：RefCounted 会自己 delete 自己。`unreference()` 返回是否该死，通常由 Ref<T> 或 Variant 持有者调用 `memdelete`。",
          "误区四：ObjectID 可以替代 Ref。ObjectID 是弱句柄，Ref 是强引用，生命周期语义完全不同。",
          "误区五：Resource 可以用 Node 的 queue_free 管。Resource 通常靠 Ref/ResourceCache 管，不在 SceneTree 删除队列中。",
          "误区六：引用计数能解决所有泄漏。循环 Ref 会互相持有，需要 WeakRef 或显式断开。",
          "误区七：把 RefCounted 裸指针长期保存很安全。长期保存应使用 Ref<T>；裸指针只适合明确短生命周期场景。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/object/ref_counted.h:36`，确认 RefCounted 的成员和公开接口。",
          "读 `RefCounted::reference()`、`unreference()`、`init_ref()`，理解计数增加、减少和第一次接管。",
          "读 `Ref<T>` 的 `ref_pointer()`、`operator=(T *)`、`unref()`、析构函数，确认强引用包装器如何工作。",
          "读 `Resource : public RefCounted`，理解为什么资源系统和 RefCounted 绑定很深。",
          "读 `Variant::ObjData` 对 RefCounted 的处理，理解脚本参数和 Variant 容器为什么也会影响引用计数。",
          "最后读 WeakRef 和 `ObjectDB::get_ref<T>()`，区分强引用、弱引用和 ObjectID。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "RefCounted 是 Godot 的共享对象生命周期模型；真正日常使用的是 `Ref<T>` 强引用，最后一个强引用释放时对象才会删除，而 ObjectID、WeakRef 和裸指针都不能替代这种所有权语义。"
      }
    ]
  },
  {
    id: "resource",
    title: "Resource",
    aliases: ["Resource", "资源", "资源对象", "resource_path", "resource_name", "resource_local_to_scene", "resource_scene_unique_id", "take_over_path", "set_path_cache", "is_built_in", "duplicate_deep", "emit_changed"],
    summary: "Godot 可加载、可保存、可共享的数据对象基类：继承 RefCounted，用 resource_path 接入 ResourceCache，并通过 ResourceLoader/ResourceSaver 进出文件系统。",
    article: [
      {
        type: "lead",
        text: "Resource 是 Godot 资源系统的核心数据基类。它不是场景树里的节点，而是可被引用计数持有、可序列化、可缓存、可嵌套到其他资源里的数据对象。Texture2D、Mesh、Material、Animation、Script、PackedScene 等大量类型都建立在 Resource 语义上。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Resource 想成游戏项目里的“资料卡”。图片、材质、场景蓝图、动画轨道、脚本这些资料可以放在磁盘上，也可以被多个节点共用。节点负责“在场景里活着和运行”，Resource 负责“保存可复用的数据”。"
      },
      {
        type: "paragraph",
        text: "复制一个 `Ref<Resource>` 不会复制资料卡内容，只是多一份强引用指向同一个 Resource。要得到内容副本，需要调用 `duplicate()`、`duplicate_deep()` 或具体资源自己的复制 API。"
      },
      {
        type: "flow",
        title: "Resource 的日常路径",
        steps: [
          { title: "磁盘或脚本创建", text: "`.tres`、`.res`、`.tscn`、图片、脚本或 `Resource.new()` 得到 Resource 对象。" },
          { title: "Ref 持有生命周期", text: "Resource 继承 RefCounted，通常由 `Ref<T>`、Variant 或脚本变量持有。" },
          { title: "路径登记到缓存", text: "有 `resource_path` 的资源会进入 ResourceCache，后续同路径加载可能复用它。" },
          { title: "节点或其他资源引用", text: "Sprite2D.texture、MeshInstance3D.mesh、PackedScene 内部子资源都可以引用 Resource。" },
          { title: "保存或实例化", text: "ResourceSaver 写回文件；PackedScene 这种 Resource 还能 instantiate 成 Node 树。" }
        ]
      },
      {
        type: "heading",
        title: "源码入口"
      },
      {
        type: "paragraph",
        text: "`Resource` 声明在 `core/io/resource.h:52`，继承 `RefCounted`。这说明它首先是 Object，能进 ClassDB、能有属性/方法/信号；其次是 RefCounted，生命周期通常由 `Ref<T>` 管；最后才是资源系统对象，额外拥有路径、缓存、复制、保存、local-to-scene 和 RID 桥接语义。"
      },
      {
        type: "paragraph",
        text: "基础注册在 `core/register_core_types.cpp:148` 的 `GDREGISTER_CLASS(Resource)`。`Resource::register_custom_data_to_otdb()` 会把默认扩展 `res` 注册到 ClassDB；子类可以用 `RES_BASE_EXTENSION(\"theme\")` 这类宏声明自己的基础扩展。"
      },
      {
        type: "paragraph",
        text: "脚本可见 API 绑定集中在 `Resource::_bind_methods()`，位置是 `core/io/resource.cpp:733`。这里绑定了 `take_over_path()`、`get_path()`、`set_path_cache()`、`get_rid()`、`duplicate()`、`duplicate_deep()`、`copy_from_resource()`、`emit_changed()`，并添加 `changed`、`resource_path`、`resource_name`、`resource_local_to_scene`、`resource_scene_unique_id` 等信号和属性。"
      },
      {
        type: "table",
        title: "Resource 关键字段和职责",
        headers: ["字段/接口", "源码入口", "负责什么"],
        rows: [
          ["`name` / `resource_name`", "`resource.h:72`、`resource.cpp:189`", "编辑器展示名和脚本可读写名称；设置时会 `emit_changed()`。"],
          ["`path_cache` / `resource_path`", "`resource.h:73`、`resource.cpp:79`", "资源路径和 ResourceCache 的键；同一路径默认只允许一个缓存资源。"],
          ["`scene_unique_id`", "`resource.h:74`、`resource.cpp:129`", "场景内子资源的稳定 ID，帮助 .tscn 保存时保持可读、可合并。"],
          ["`local_to_scene`", "`resource.h:88`、`resource.cpp:652`", "标记这个资源在 PackedScene 实例化时是否要为每个场景实例复制一份。"],
          ["`emit_changed_state`", "`resource.h:87`、`resource.cpp:49`", "控制 `changed` 信号，支持阻塞期间合并成一次延后发射。"],
          ["`get_rid()`", "`resource.cpp:618`", "给 Texture、Mesh 等高层资源返回底层 Server RID 的统一入口。"]
        ]
      },
      {
        type: "heading",
        title: "路径和 ResourceCache"
      },
      {
        type: "paragraph",
        text: "`Resource::set_path()` 是 Resource 和全局缓存相连的关键函数。它先从旧 `path_cache` 移除自己，再检查新路径是否已有资源；如果已有且不是 `take_over_path()` 场景，就报错并拒绝覆盖；如果允许 takeover，就清掉旧资源路径并把当前对象登记到 `ResourceCache::resources[p_path]`。"
      },
      {
        type: "paragraph",
        text: "`set_path_cache()` 不触碰 ResourceCache，只直接改 `path_cache` 并调用虚拟扩展点 `_set_path_cache`。这主要给自定义 ResourceFormatLoader/ResourceFormatSaver 和 cache mode 处理用，不是普通用户覆盖缓存的工具。"
      },
      {
        type: "paragraph",
        text: "`ResourceCache` 定义在 `core/io/resource.h:197`，核心就是一个 `HashMap<String, Resource *> resources` 加锁保护。`ResourceCache::has()` 和 `get_ref()` 会检查资源是否正在被删除；如果引用计数已经归零，会把缓存条目清掉，避免返回即将销毁的对象。"
      },
      {
        type: "flow",
        title: "同一路径资源为什么会共享",
        steps: [
          { title: "第一次 load", text: "ResourceLoader 选中格式加载器，创建 Resource。" },
          { title: "设置路径", text: "加载完成后调用 `resource->set_path(local_path)`。" },
          { title: "进入 ResourceCache", text: "`ResourceCache::resources[local_path] = this`。" },
          { title: "再次 load", text: "默认 `CACHE_MODE_REUSE` 先查 `ResourceCache::get_ref(local_path)`。" },
          { title: "返回同一对象", text: "拿到的是同一个 Resource 的新强引用，不是重新解析文件。" }
        ]
      },
      {
        type: "table",
        title: "path 相关 API 的边界",
        headers: ["API", "做什么", "不要误解成"],
        rows: [
          ["`resource_path` / `set_path()`", "设置路径并参与 ResourceCache 唯一登记。", "普通字符串字段；它会影响后续按路径加载结果。"],
          ["`take_over_path()`", "强行让当前资源接管某个路径的缓存入口。", "复制文件或保存文件；它只是改内存缓存映射。"],
          ["`set_path_cache()`", "只改对象内部路径缓存，不登记 ResourceCache。", "安全替代 `take_over_path()`；它会绕开缓存唯一性。"],
          ["`is_built_in()`", "路径为空、含 `::` 或以 `local://` 开头时视为内置/本地资源。", "判断资源类型；它判断的是保存位置语义。"],
          ["`ResourceCache::get_ref()`", "按路径返回缓存资源的强引用。", "磁盘缓存；它只是当前进程内对象表。"]
        ]
      },
      {
        type: "heading",
        title: "加载和保存中的 Resource"
      },
      {
        type: "paragraph",
        text: "ResourceLoader 的职责是从路径得到 `Ref<Resource>`。`ResourceLoader::_load_start()` 在默认 `CACHE_MODE_REUSE` 下会先查 ResourceCache；真正解析文件的是注册进来的 `ResourceFormatLoader`。加载完成后，如果不是 ignore cache，就把资源路径设为本地路径，从而进入缓存。"
      },
      {
        type: "paragraph",
        text: "ResourceSaver 的职责是把 `Ref<Resource>` 写回路径。`ResourceSaver::save()` 会遍历注册的 `ResourceFormatSaver`，先问 `recognize(resource)`，再问 `recognize_path(resource, path)`，匹配后才实际保存。`FLAG_CHANGE_PATH` 只在保存期间临时设置路径，保存后会恢复旧路径。"
      },
      {
        type: "table",
        title: "Resource、Loader、Saver 的分工",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Resource", "保存资源对象的路径、名称、复制规则、changed 信号和可序列化属性。", "不自己决定哪个文件格式能加载/保存。"],
          ["ResourceLoader", "标准化路径、处理缓存和加载任务，选择 ResourceFormatLoader。", "不直接解析所有格式细节。"],
          ["ResourceFormatLoader", "把某类文件解析成具体 Resource。", "不管理全局路径缓存策略。"],
          ["ResourceSaver", "选择 ResourceFormatSaver，把 Resource 写回文件。", "不拥有资源生命周期。"],
          ["ResourceFormatSaver", "把某类 Resource 序列化成具体文件格式。", "不决定其他资源是否共享同一实例。"]
        ]
      },
      {
        type: "heading",
        title: "复制、深拷贝和本地场景资源"
      },
      {
        type: "paragraph",
        text: "`Resource::duplicate()` 和 `duplicate_deep()` 不会复制所有成员变量，而是读取属性列表，只复制带 `PROPERTY_USAGE_STORAGE` 的属性。`script` 会特殊处理，`resource_path` 在 `copy_from_resource()` 里会被跳过，避免把副本或替换资源直接改成原资源路径。"
      },
      {
        type: "paragraph",
        text: "深拷贝由 `_duplicate_recursive()` 处理。数组、字典和 packed arrays 会复制；遇到嵌套 Resource 时，要看 `PROPERTY_USAGE_ALWAYS_DUPLICATE`、`PROPERTY_USAGE_NEVER_DUPLICATE`、deep mode 和是否 built-in。默认 `DEEP_DUPLICATE_INTERNAL` 只复制内置/本地子资源，不会把外部 `.tres`、贴图、脚本等都复制一遍。脚本资源即使命中也会被排除。"
      },
      {
        type: "paragraph",
        text: "`resource_local_to_scene` 是场景实例隔离资源状态的开关。PackedScene 实例化时，如果某个 Resource 标记为 local-to-scene，就通过 `duplicate_for_local_scene()` 复制，并在新副本上设置 `local_scene`。随后 `setup_local_to_scene()` 会发出旧的 `setup_local_to_scene_requested` 信号并调用可重写的 `_setup_local_to_scene()`。"
      },
      {
        type: "flow",
        title: "local-to-scene 的路径",
        steps: [
          { title: "资源标记", text: "`resource_local_to_scene = true`。" },
          { title: "PackedScene instantiate", text: "场景实例化恢复属性时发现本地资源。" },
          { title: "复制资源", text: "`duplicate_for_local_scene()` 使用同一 remap cache，避免同一子资源重复复制。" },
          { title: "绑定场景根", text: "新资源的 `local_scene` 指向当前实例根节点。" },
          { title: "初始化副本", text: "调用 `_setup_local_to_scene()`，可以给每个实例设置不同状态。" }
        ]
      },
      {
        type: "table",
        title: "复制模式怎么读",
        headers: ["模式/标记", "效果", "典型用途"],
        rows: [
          ["`duplicate(false)`", "浅复制；嵌套 Resource、Array、Dictionary 多数仍共享。", "只想复制资源外壳或简单属性。"],
          ["`duplicate(true)`", "深复制数组/字典，并按 INTERNAL 规则复制内置子资源。", "复制场景内子资源，不复制外部大资源。"],
          ["`DEEP_DUPLICATE_NONE`", "数组/字典可复制，但子资源仍指向原对象。", "保留共享资源引用。"],
          ["`DEEP_DUPLICATE_INTERNAL`", "只复制 built-in/local 子资源。", "默认安全选择，避免外部资源膨胀。"],
          ["`DEEP_DUPLICATE_ALL`", "能遇到的子资源都复制，外部资源也复制。", "确实需要完全独立资源图时。"],
          ["`ALWAYS/NEVER_DUPLICATE`", "属性级标记覆盖普通 deep 参数。", "资源子类声明固定复制规则。"]
        ]
      },
      {
        type: "heading",
        title: "changed 信号、重载和编辑器状态"
      },
      {
        type: "paragraph",
        text: "`emit_changed()` 会发出 `changed` 信号。内置资源会在某些属性变化时自动调用；自定义 Resource 的普通导出变量不会天然知道“有意义的变化”，通常要在 setter 里手动调用 `emit_changed()`，否则依赖它的节点、预览或编辑器 UI 可能不会刷新。"
      },
      {
        type: "paragraph",
        text: "`reload_from_file()` 会用 `CACHE_MODE_IGNORE` 从路径重新加载同类资源，再 `copy_from()` 到当前对象。`copy_from()` 会先 `reset_state()`，再复制 `PROPERTY_USAGE_STORAGE` 属性，并用 `_block_emit_changed()` / `_unblock_emit_changed()` 把过程中多次变化合并成一次信号。"
      },
      {
        type: "code",
        code: [
          "extends Resource",
          "",
          "@export var damage := 10:",
          "    set(value):",
          "        if damage == value:",
          "            return",
          "        damage = value",
          "        emit_changed()"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个 setter 不是为了“让值能保存”，而是为了通知使用者资源内容变了。保存仍取决于属性是否带 storage usage；刷新则取决于谁监听了 `changed`。"
      },
      {
        type: "heading",
        title: "Resource 和 RID 的关系"
      },
      {
        type: "paragraph",
        text: "`get_rid()` 是 Resource 通向底层 Server 的统一桥。Resource 本身不是 RID，也不拥有所有 Server 对象；但 Texture2D、Mesh、Font 等高层资源可以重写 `_get_rid()`，返回 RenderingServer、TextServer 或其他 Server 管理的底层句柄。"
      },
      {
        type: "paragraph",
        text: "读源码时要把这两层分开：Resource 是可序列化、可引用计数、可被编辑器和脚本持有的高层对象；RID 是 Server 内部对象句柄，释放规则由对应 Server 管。把 RID 当 Resource 或把 Resource 当 RID，都会看错生命周期。"
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：共享资源会共享状态"
      },
      {
        type: "code",
        code: [
          "var mat := load(\"res://enemy_material.tres\")",
          "$EnemyA.material_override = mat",
          "$EnemyB.material_override = mat",
          "",
          "mat.albedo_color = Color.RED",
          "# 两个节点看到的是同一个 Resource 实例的变化。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "如果不希望共享状态，应该复制资源或把资源设为 local-to-scene，而不是以为再次 load 同一路径会得到新对象。"
      },
      {
        type: "subheading",
        title: "案例二：接管路径只影响内存缓存"
      },
      {
        type: "code",
        code: [
          "var replacement := Resource.new()",
          "replacement.take_over_path(\"res://config/runtime.tres\")",
          "",
          "var loaded := load(\"res://config/runtime.tres\")",
          "# 默认缓存模式下，loaded 会拿到 replacement。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`take_over_path()` 不会写磁盘文件。它只是让当前进程里后续同路径加载优先返回这个资源。"
      },
      {
        type: "subheading",
        title: "案例三：每个场景实例得到独立资源"
      },
      {
        type: "code",
        code: [
          "extends Resource",
          "",
          "@export var seed := 0",
          "",
          "func _setup_local_to_scene():",
          "    seed = randi()"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "把这个资源保存到场景里并设置 `resource_local_to_scene = true` 后，每次 PackedScene 实例化都会复制一份，并调用 `_setup_local_to_scene()`。"
      },
      {
        type: "heading",
        title: "Resource 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["RefCounted", "引用计数生命周期。", "不提供路径、缓存、保存和资源复制规则。"],
          ["Resource", "可序列化数据对象、路径缓存入口、复制规则、changed 信号和可选 RID 桥接。", "不进入 SceneTree，也不代表运行中的节点。"],
          ["Node", "场景树父子关系、生命周期、process、owner 和运行时行为。", "不是资源缓存对象，不能靠 Ref 自动释放。"],
          ["PackedScene", "一种 Resource，保存可实例化 Node 树的 SceneState。", "加载后不是 Node；要 `instantiate()` 才生成节点。"],
          ["ResourceLoader", "按路径和格式加载 Resource，并处理缓存策略。", "不是 Resource 基类，也不保存资源内容。"],
          ["ResourceCache", "当前进程内按路径保存 Resource 指针。", "不是磁盘缓存，也不复制资源内容。"],
          ["RID", "Server 内部对象句柄。", "不是 RefCounted，不负责高层资源序列化。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Resource 就是 RefCounted。Resource 继承 RefCounted，但额外提供路径、缓存、保存、复制、changed 和 local-to-scene 语义。",
          "误区二：Resource 是 Node。Resource 是数据对象，不能进入场景树，也没有 enter_tree/ready/process。",
          "误区三：再次 load 同一路径会重新读文件并创建新对象。默认会优先复用 ResourceCache。",
          "误区四：复制 Ref 就复制资源内容。复制 Ref 只是增加强引用，内容仍共享。",
          "误区五：`resource_path` 是普通显示字段。它是 ResourceCache 的键，会影响后续加载。",
          "误区六：`take_over_path()` 会保存文件。它只接管内存缓存入口。",
          "误区七：深拷贝会复制所有资源。默认只复制内置/本地子资源，外部资源通常仍共享。",
          "误区八：custom Resource 属性变化会自动发 `changed`。自定义资源通常需要在 setter 中手动 `emit_changed()`。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/io/resource.h:52` 的类声明，确认它继承 RefCounted，并标出 name、path_cache、scene_unique_id、local_to_scene 等字段。",
          "读 `core/register_core_types.cpp:148` 和 `Resource::register_custom_data_to_otdb()`，确认 Resource 何时注册、默认扩展如何进入 ClassDB。",
          "读 `Resource::_bind_methods()`，把脚本 API、属性和信号与 C++ 方法对上。",
          "读 `Resource::set_path()`、`take_over_path()`、`ResourceCache::has()`、`get_ref()`，理解同一路径为什么共享对象。",
          "读 `ResourceLoader::_load_start()` 和加载完成后的 `set_path()` 路径，把 cache mode 和 ResourceCache 对上。",
          "读 `ResourceSaver::save()`，理解保存器如何识别 Resource 并写回文件。",
          "读 `_duplicate_recursive()`、`duplicate()`、`duplicate_deep()` 和 `duplicate_for_local_scene()`，理解浅复制、深复制、本地场景资源的区别。",
          "最后读 `emit_changed()`、`copy_from()`、`reload_from_file()`、`get_rid()`，补上编辑器刷新、文件重载和 Server 句柄边界。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Resource 是 Godot 的可序列化共享数据基类：它借 RefCounted 管生命周期，借 resource_path 接入 ResourceCache，借 Loader/Saver 进出文件系统，并用复制规则、changed 信号和 local-to-scene 语义处理资源共享与隔离。"
      }
    ]
  },
  {
    id: "messagequeue",
    title: "MessageQueue",
    aliases: ["MessageQueue", "CallQueue", "消息队列", "延迟调用", "call_deferred", "set_deferred", "CONNECT_DEFERRED"],
    summary: "Godot 的延迟调用队列，把 call/set/notification 暂存起来，在主循环或场景树的安全点统一 flush。",
    article: [
      {
        type: "lead",
        text: "MessageQueue 是 Godot 里“先记下来，稍后再执行”的核心设施。`call_deferred()`、`set_deferred()`、延迟信号连接和 SceneTree 的延迟组调用，最终都会把一次调用、一次属性设置或一次通知塞进这个队列，然后等主循环或场景树走到安全点时 `flush()`。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 MessageQueue 想成游戏引擎内部的待办事项箱。你正在遍历场景树、处理物理、发射信号时，如果马上改节点状态可能把当前流程弄乱，于是 Godot 先把“等会儿调用这个方法”“等会儿设置这个属性”写进待办箱。当前这轮关键工作结束后，引擎再打开待办箱逐条处理。"
      },
      {
        type: "paragraph",
        text: "`call_deferred()` 的重点不是“开一个新线程”，也不是“下一行马上执行”。它只是排队：当前函数继续执行，队列里的调用要等之后的 flush 点才运行。如果目标对象在 flush 前已经被释放，MessageQueue 会重新检查目标，通常会跳过这条消息。"
      },
      {
        type: "flow",
        title: "待办事项箱心智模型",
        steps: [
          { title: "你发出延迟请求", text: "脚本或 C++ 调用 `call_deferred`、`set_deferred`、延迟信号或延迟组调用。" },
          { title: "MessageQueue 记账", text: "队列保存目标 ObjectID、方法/属性名、参数副本或通知编号。" },
          { title: "当前流程继续", text: "调用者不会等队列执行，当前遍历、信号或物理步骤继续走完。" },
          { title: "安全点 flush", text: "SceneTree 或 MainLoop 到达固定位置，调用 `MessageQueue::flush()`。" },
          { title: "重新找目标再执行", text: "目标还活着就 call/set/notification；目标不在了就跳过。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "源码上，`MessageQueue` 继承 `CallQueue`，定义在 `core/object/message_queue.h:150`。真正的数据结构和操作都在 `CallQueue` 里：4096 字节一页的 `Page`、保存每页已用字节数的 `page_bytes`、当前使用页数 `pages_used`、最大页数 `max_pages`、互斥锁 `mutex` 和是否正在 flush 的 `flushing`。"
      },
      {
        type: "paragraph",
        text: "`Message` 结构在 `core/object/message_queue.h:83`：它保存一个 `Callable callable`、一个 `int16_t type`，以及 union 里的 `notification` 或 `args`。call 和 set 后面紧跟若干个 placement-new 出来的 `Variant` 参数；notification 不需要 Variant 参数，只保存通知编号。"
      },
      {
        type: "table",
        title: "哪些入口会写入 MessageQueue",
        headers: ["入口", "源码位置", "写入什么", "典型用途"],
        rows: [
          ["`Object::call_deferredp()`", "`core/object/object.cpp:1998`", "`push_callp(this, method, args, argcount)`", "脚本的 `call_deferred`。"],
          ["`Object::set_deferred()`", "`core/object/object.cpp:2002`", "`push_set(this, property, value)`", "延迟设置属性。"],
          ["`CONNECT_DEFERRED` 信号", "`core/object/object.cpp:1287`", "`push_callablep(callable, args, argc, true)`", "信号发射时不立刻调用连接目标。"],
          ["`SceneTree::call_group_flags()`", "`scene/main/scene_tree.cpp:422`、`:440`", "`push_callp(node, function, args, argcount)`", "延迟调用组内节点方法。"],
          ["`SceneTree::notify_group_flags()`", "`scene/main/scene_tree.cpp:489`、`:502`", "`push_notification(node, notification)`", "延迟给组内节点发通知。"],
          ["`SceneTree::set_group_flags()`", "`scene/main/scene_tree.cpp:551`、`:564`", "`push_set(node, name, value)`", "延迟设置组内节点属性。"]
        ]
      },
      {
        type: "table",
        title: "队列里的消息长什么样",
        headers: ["字段/存储", "含义", "为什么这样设计"],
        rows: [
          ["`Callable callable`", "保存目标 ObjectID 和方法名，也可以保存自定义 Callable。", "flush 时通过 Callable 重新取目标对象，不长期依赖裸指针。"],
          ["`type`", "`TYPE_CALL`、`TYPE_NOTIFICATION`、`TYPE_SET` 加上一些 flag。", "同一个队列能承载方法调用、通知和属性设置。"],
          ["`args`", "call/set 消息后方连续存放 `Variant` 参数副本。", "调用者返回后，原参数可能失效，队列必须自己持有副本。"],
          ["`notification`", "notification 消息只存一个 int16 通知编号。", "通知不需要 Variant 参数，消息更小。"],
          ["`FLAG_SHOW_ERROR`", "延迟调用失败时打印可读错误。", "`call_deferred` 和延迟信号常需要把失败暴露给开发者。"],
          ["`FLAG_NULL_IS_OK`", "允许无对象的合法 Callable，例如某些静态方法。", "不是所有 Callable 都必须有实例目标。"]
        ]
      },
      {
        type: "flow",
        title: "`call_deferred()` 到实际调用的路径",
        steps: [
          { title: "脚本调用", text: "`node.call_deferred(\"refresh\", value)` 进入 Object 的绑定函数。" },
          { title: "Object 转发", text: "`Object::_call_deferred_bind()` 或 `call_deferredp()` 调 `MessageQueue::get_singleton()->push_callp(...)`。" },
          { title: "构造 Callable", text: "`push_callp(ObjectID, method, ...)` 包成 `Callable(p_id, p_method)`。" },
          { title: "复制参数", text: "`push_callablep()` 在队列页内 placement-new `Message` 和参数 `Variant`。" },
          { title: "后续 flush", text: "`flush()` 取出消息，`Callable::callp()` 真正执行。" }
        ]
      },
      {
        type: "heading",
        title: "push 阶段：把消息写进分页内存"
      },
      {
        type: "paragraph",
        text: "`push_callablep()` 是 call 类消息的核心入口，定义在 `core/object/message_queue.cpp:84`。它先计算 `room_needed = sizeof(Message) + sizeof(Variant) * p_argcount`，如果单条消息比 4096 字节页面还大，就直接报错；如果当前页空间不够，就加新页。达到 `max_pages` 时会打印统计并返回 `ERR_OUT_OF_MEMORY`。"
      },
      {
        type: "paragraph",
        text: "写入消息时，MessageQueue 使用 placement new：先在当前页的空闲字节位置构造 `Message`，再在 `message + 1` 后面连续构造参数 `Variant`。所以 MessageQueue 不是把每条消息做成单独堆对象，而是用一页一页的连续内存减少碎片和分配次数。"
      },
      {
        type: "table",
        title: "三类 push 的差异",
        headers: ["push API", "消息类型", "参数存储", "flush 时动作"],
        rows: [
          ["`push_callp()` / `push_callablep()`", "`TYPE_CALL`", "保存 0 个或多个 `Variant` 参数。", "调用 `_call_function()`，内部走 `Callable::callp()`。"],
          ["`push_set()`", "`TYPE_SET`", "保存 1 个 `Variant` 属性值。", "目标存在时执行 `target->set(property, value)`。"],
          ["`push_notification()`", "`TYPE_NOTIFICATION`", "不保存 Variant，只保存通知编号。", "目标存在时执行 `target->notification(notification)`。"]
        ]
      },
      {
        type: "code",
        code: [
          "// message_queue.cpp 的简化心智模型，不是逐字源码。",
          "room_needed = sizeof(Message) + sizeof(Variant) * arg_count;",
          "page = pages[pages_used - 1];",
          "Message *msg = memnew_placement(buffer_end, Message);",
          "msg->callable = callable;",
          "msg->type = TYPE_CALL;",
          "",
          "Variant *args = (Variant *)(msg + 1);",
          "for (int i = 0; i < arg_count; i++) {",
          "    memnew_placement(&args[i], Variant(*input_args[i]));",
          "}"
        ].join("\n")
      },
      {
        type: "heading",
        title: "flush 阶段：安全点执行"
      },
      {
        type: "paragraph",
        text: "`CallQueue::flush()` 定义在 `core/object/message_queue.cpp:224`。它先加锁并检查是否已经在 flush；如果正在 flush，会返回 `ERR_BUSY`。真正执行每条消息前，它会先算出当前消息占多少字节，并把 offset 提前推进。这个“先推进再执行”的细节很重要：如果消息执行过程中又把新消息加入队列，队列状态仍然能继续前进。"
      },
      {
        type: "paragraph",
        text: "执行消息前，`flush()` 会通过 `message->callable.get_object()` 重新取目标对象。这里会走 ObjectDB/Callable 的对象查找语义，而不是相信当初入队时的裸指针。如果目标对象已经释放，call/set/notification 通常直接跳过。执行期间队列会临时解锁，执行完再析构参数 Variant 和 Message。"
      },
      {
        type: "flow",
        title: "`flush()` 的执行流程",
        steps: [
          { title: "锁住队列", text: "确认有页面且没有递归 flush，然后设置 `flushing = true`。" },
          { title: "读取当前消息", text: "从 page + offset 解释出 `Message`，计算它占用的字节数。" },
          { title: "先推进 offset", text: "提前越过当前消息，允许当前调用过程中继续向队列追加消息。" },
          { title: "重新取目标", text: "用 Callable 取 Object，目标不存在就跳过实例相关消息。" },
          { title: "解锁并执行", text: "call、notification、set 三选一执行。" },
          { title: "析构并继续", text: "销毁参数 Variant 和 Message，回到队列锁内继续下一条。" },
          { title: "清空页状态", text: "结束后保留第一页复用，`page_bytes[0] = 0`、`pages_used = 1`。" }
        ]
      },
      {
        type: "table",
        title: "flush 时三种消息怎么处理",
        headers: ["消息", "目标不存在时", "目标存在时", "错误处理"],
        rows: [
          ["`TYPE_CALL`", "一般跳过；如果有 `FLAG_NULL_IS_OK`，仍可调用合法的无实例 Callable。", "把参数数组交给 `Callable::callp()`。", "`FLAG_SHOW_ERROR` 开启时打印调用错误文本。"],
          ["`TYPE_SET`", "跳过。", "调用 `target->set(property, value)`。", "属性是否有效由 Object 的 set 路径处理。"],
          ["`TYPE_NOTIFICATION`", "跳过。", "调用 `target->notification(notification)`。", "负数通知在 push 阶段会被拒绝。"]
        ]
      },
      {
        type: "heading",
        title: "它和主循环、场景树的关系"
      },
      {
        type: "paragraph",
        text: "MessageQueue 自己只负责排队和 flush，不决定“每帧哪里 flush”。具体安全点由 MainLoop、SceneTree 或少数资源加载路径调用。`SceneTree::physics_process()` 在物理处理和若干通知后调用 `MessageQueue::get_singleton()->flush()`，位置在 `scene/main/scene_tree.cpp:658`；普通 process 路径在 `scene/main/scene_tree.cpp:715` 和 `:722` 也会 flush。"
      },
      {
        type: "paragraph",
        text: "这也是为什么 `call_deferred()` 常被用来避免“正在遍历场景树时马上改树”的问题：它把修改推迟到引擎安排好的位置。但它不等同于 `queue_free()`。删除队列由 SceneTree 的 `_flush_delete_queue()` 处理，常常靠近 MessageQueue flush，但职责不同。"
      },
      {
        type: "flow",
        title: "一帧里 deferred 调用的大致位置",
        steps: [
          { title: "用户脚本或引擎代码", text: "产生 `call_deferred`、延迟信号、组延迟调用。" },
          { title: "MessageQueue 累积消息", text: "消息先放进队列，当前遍历继续。" },
          { title: "SceneTree 安全点", text: "process/physics_process 中固定位置调用 `flush()`。" },
          { title: "执行延迟消息", text: "MessageQueue 重新查对象并执行 call/set/notification。" },
          { title: "删除队列等其他阶段", text: "SceneTree 还会在自己的时机处理 queued delete、timer、tween 等。"}
        ]
      },
      {
        type: "heading",
        title: "线程单例和局部队列"
      },
      {
        type: "paragraph",
        text: "`MessageQueue` 有一个 `main_singleton`，也有一个 `thread_local CallQueue *thread_singleton`，入口在 `core/object/message_queue.h:152`。`get_singleton()` 会优先返回当前线程覆盖的队列，否则返回主队列。`set_thread_singleton_override()` 在 `core/object/message_queue.cpp:490`，用于特定线程临时接管消息队列。"
      },
      {
        type: "paragraph",
        text: "这不意味着你可以从任意线程随便改场景树。MessageQueue 的队列写入有锁，但对象本身、SceneTree 和大多数 Node API 仍有线程约束。线程 override 更像是给特定执行环境准备一条局部 CallQueue，而不是把所有 Godot 对象 API 变成线程安全。"
      },
      {
        type: "table",
        title: "MessageQueue 和相邻概念的边界",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "提供 `call_deferredp()`、`set_deferred()`、信号发射和 ObjectID。", "不保存延迟消息本身。"],
          ["Callable", "封装目标、方法和调用逻辑。", "不决定何时执行。"],
          ["Signal", "`CONNECT_DEFERRED` 时把 Callable 推入 MessageQueue。", "不自己维护延迟队列。"],
          ["ObjectDB", "让 flush 时可以从 ObjectID 重新确认目标还活着。", "不拥有对象，也不保证对象一定存在。"],
          ["MessageQueue", "保存延迟 call/set/notification，并在 flush 时执行。", "不管理对象生命周期，也不替代线程同步。"],
          ["SceneTree 删除队列", "处理 `queue_free()` 产生的待删除 Node。", "不负责普通延迟方法调用。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：延迟添加子节点"
      },
      {
        type: "code",
        code: [
          "func spawn_later(child: Node) -> void:",
          "    call_deferred(\"add_child\", child)",
          "    print(\"这里会先执行，child 还不一定已经进入树\")",
          "",
          "func _ready() -> void:",
          "    var marker := Node2D.new()",
          "    spawn_later(marker)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这类写法适合避免在某些通知、遍历或初始化流程中马上改场景树。入队时 MessageQueue 会复制参数 Variant；真正的 `add_child` 要等后续 flush。"
      },
      {
        type: "subheading",
        title: "案例二：延迟设置属性"
      },
      {
        type: "code",
        code: [
          "func disable_collision_later(shape: CollisionShape2D) -> void:",
          "    shape.set_deferred(\"disabled\", true)",
          "",
          "# 等价心智：先把 disabled=true 这个 set 消息排队，",
          "# 后面安全点再走 Object::set 路径。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`set_deferred` 最终进入 `CallQueue::push_set()`，消息只带一个属性值 Variant。flush 时目标还存在才调用 `target->set(property, value)`。"
      },
      {
        type: "subheading",
        title: "案例三：C++ 里直接推入延迟调用"
      },
      {
        type: "code",
        code: [
          "Variant value = 42;",
          "const Variant *args[] = { &value };",
          "",
          "MessageQueue::get_singleton()->push_callp(",
          "        some_object->get_instance_id(),",
          "        SNAME(\"refresh_from_queue\"),",
          "        args,",
          "        1,",
          "        true);"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Object* overload 最终也会转成 ObjectID。这样队列不会长期持有裸指针；flush 时会重新判断目标是否仍然有效。"
      },
      {
        type: "subheading",
        title: "案例四：延迟信号连接"
      },
      {
        type: "code",
        code: [
          "button->connect(",
          "        SNAME(\"pressed\"),",
          "        callable_mp(this, &MyPanel::_on_pressed),",
          "        CONNECT_DEFERRED);",
          "",
          "// 信号发射时不立即调用 _on_pressed，",
          "// Object::emit_signalp() 会把 callable 推入 MessageQueue。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这在信号发射期间可能修改连接对象、释放对象或触发复杂 UI 状态变化时很有用。源码里 `CONNECT_DEFERRED` 的分支在 `core/object/object.cpp:1287`。"
      },
      {
        type: "subheading",
        title: "案例五：SceneTree 延迟组调用"
      },
      {
        type: "code",
        code: [
          "get_tree().call_group_flags(",
          "        SceneTree.GROUP_CALL_DEFERRED,",
          "        \"enemies\",",
          "        \"refresh_target\",",
          "        player)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "组调用会先复制组内节点列表，再按 flag 决定直接调用还是推入 MessageQueue。延迟版本能避开遍历组时立刻修改节点状态带来的风险。"
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：`call_deferred` 是多线程。它不是开新线程，只是排队到之后的 flush 点。",
          "误区二：延迟调用一定在下一行之后马上执行。当前函数会继续执行，但真正执行时间取决于主循环或 SceneTree 的 flush 位置。",
          "误区三：MessageQueue 会保证目标对象活到执行时。队列主要保存 ObjectID/Callable 和参数副本，不拥有目标对象。",
          "误区四：入队参数没有成本。call/set 消息会复制 Variant 参数，大对象或大量消息会增加内存压力。",
          "误区五：可以无限排队。队列最大大小来自项目设置 `memory/limits/message_queue/max_size_mb`，超限会报错并打印统计。",
          "误区六：`set_deferred` 绕过属性系统。flush 时仍然走 `Object::set`，setter、脚本属性和错误行为仍按 Object 规则处理。",
          "误区七：MessageQueue 等于删除队列。`queue_free()` 的删除队列是 SceneTree 的另一套机制，只是经常在相近安全点处理。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/object/message_queue.h:41` 的 `CallQueue` 字段，确认页面、锁、消息结构和类型枚举。",
          "读 `MessageQueue::get_singleton()` 和 `set_thread_singleton_override()`，理解主队列和线程局部队列。",
          "读 `core/object/object.cpp:1998` 和 `:2002`，把脚本里的 `call_deferred`、`set_deferred` 对上源码入口。",
          "读 `message_queue.cpp:84` 的 `push_callablep()`，重点看空间计算、加页、placement new 和 Variant 参数复制。",
          "读 `message_queue.cpp:224` 的 `flush()`，重点看预推进 offset、重新取目标、解锁执行和析构消息。",
          "最后读 `scene/main/scene_tree.cpp:658`、`:715`、`:722`，把队列执行点放回每帧流程里理解。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "MessageQueue 是 Godot 的延迟执行缓冲区：它把 call、set 和 notification 变成可稍后处理的消息，在引擎安排好的安全点重新确认目标并执行，从而避免在敏感遍历或信号流程中立刻改对象状态。"
      }
    ]
  },
  {
    id: "resourceloader",
    title: "ResourceLoader",
    aliases: ["ResourceLoader", "ResourceFormatLoader", "ResourceCache", "资源加载", "load", "load_threaded_request", "资源缓存"],
    summary: "Godot 的资源加载调度器：标准化路径、处理 remap/UID/cache，选择合适的 ResourceFormatLoader，并支持同步或线程加载。",
    article: [
      {
        type: "lead",
        text: "ResourceLoader 是 Godot 资源系统的总入口。用户说“加载 res://main.tscn”，ResourceLoader 负责把路径变成引擎能理解的本地路径，检查缓存和正在加载的任务，处理 UID、翻译 remap、导入产物，再把文件交给合适的 ResourceFormatLoader。真正解析 .tscn、.res、图片、shader 的不是 ResourceLoader 本身，而是被它管理的一组格式加载器。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 ResourceLoader 想成资源图书馆的前台。你给它一本书的地址，例如 `res://levels/main.tscn`，它先查这本书是不是已经借过并放在缓存里；如果没有，再根据书的格式找负责这类文件的馆员，比如文本场景加载器、二进制资源加载器、图片加载器。馆员把文件读成一个 `Resource` 对象后，ResourceLoader 再把结果交给你。"
      },
      {
        type: "paragraph",
        text: "所以 `ResourceLoader.load()` 不是简单的“打开文件并返回字节”。它返回的是 Godot 的 Resource 对象，比如 `PackedScene`、`Texture2D`、`Script`、`Material`。同一个路径通常会复用同一个缓存资源；如果你想绕过或替换缓存，要显式选择 cache mode。"
      },
      {
        type: "flow",
        title: "从路径到 Resource 的直觉流程",
        steps: [
          { title: "用户给路径", text: "`ResourceLoader.load(\"res://main.tscn\")` 或 `preload()`/场景实例化间接触发加载。" },
          { title: "前台整理地址", text: "相对路径、UID、项目路径会被转成本地 `res://` 路径。" },
          { title: "先看缓存", text: "默认 `CACHE_MODE_REUSE` 会优先复用 ResourceCache 里的对象。" },
          { title: "找格式加载器", text: "遍历注册的 ResourceFormatLoader，按扩展名和 type hint 找能处理的加载器。" },
          { title: "解析成 Resource", text: "具体加载器读取文件、依赖和子资源，构造出 Ref<Resource>。" },
          { title: "登记并返回", text: "成功后设置路径、进入缓存，调用者拿到强引用。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`ResourceLoader` 声明在 `core/io/resource_loader.h:103`，内部有最多 64 个 `ResourceFormatLoader` 的静态数组。`ResourceFormatLoader` 声明在 `core/io/resource_loader.h:48`，是所有具体格式加载器的基类，提供 `recognize_path()`、`load()`、`get_dependencies()`、`get_resource_type()`、`get_resource_uid()` 等接口。"
      },
      {
        type: "paragraph",
        text: "`ResourceLoader::load()` 在 `core/io/resource_loader.cpp:513`。它并不直接读文件，而是创建一个 `LoadToken`，调用 `_load_start()` 建立或复用加载任务，再调用 `_load_complete()` 等任务完成并拿回 `Ref<Resource>`。真正遍历加载器的函数是 `_load()`，入口在 `core/io/resource_loader.cpp:275`。"
      },
      {
        type: "table",
        title: "ResourceLoader 里的核心角色",
        headers: ["角色", "源码位置", "负责什么", "常见误解"],
        rows: [
          ["`ResourceLoader`", "`core/io/resource_loader.h:103`", "静态加载入口、任务管理、缓存策略、loader 调度、线程加载 API。", "它不是某一种资源格式的解析器。"],
          ["`ResourceFormatLoader`", "`core/io/resource_loader.h:48`", "具体格式加载器基类，负责识别路径并把文件解析成 Resource。", "不是只有文本场景一个加载器。"],
          ["`ResourceCache`", "`core/io/resource.h:197`", "按路径保存已加载 Resource 的弱表，配合 Ref 返回强引用。", "不是磁盘缓存，也不是导入缓存。"],
          ["`LoadToken`", "`core/io/resource_loader.h:134`", "加载任务的引用计数令牌，保证任务完成前状态不会被过早清理。", "不是用户可直接操作的资源。"],
          ["`ThreadLoadTask`", "`core/io/resource_loader.h:181`", "保存线程加载状态、进度、父子任务、结果和错误。", "线程加载不是把 Resource 变成 Node 或自动进树。"]
        ]
      },
      {
        type: "table",
        title: "常用公开 API",
        headers: ["API", "源码入口", "作用", "返回/状态"],
        rows: [
          ["`load(path, type_hint, cache_mode)`", "`resource_loader.cpp:513`", "同步加载资源；如果在 WorkerThreadPool 任务里调用，会改用新任务避免死锁。", "`Ref<Resource>`，失败为空。"],
          ["`load_threaded_request(path, type_hint, use_sub_threads, cache_mode)`", "`resource_loader.cpp:488`", "发起后台加载请求。", "`OK` 或 `FAILED`。"],
          ["`load_threaded_get_status(path, &progress)`", "`resource_loader.cpp:675`", "查询后台加载状态和依赖进度。", "`THREAD_LOAD_*` 枚举。"],
          ["`load_threaded_get(path)`", "`resource_loader.cpp:720`", "领取后台加载结果；未完成时可能等待。", "`Ref<Resource>`。"],
          ["`exists(path, type_hint)`", "`resource_loader.cpp:1038`", "先查 ResourceCache，再问能识别该路径的加载器。", "`bool`。"],
          ["`add_resource_format_loader(loader, at_front)`", "`resource_loader.cpp:1062`", "注册格式加载器，`at_front` 会让它优先匹配。", "影响后续 loader 选择顺序。"]
        ]
      },
      {
        type: "heading",
        title: "同步 load 的源码路径"
      },
      {
        type: "flow",
        title: "`ResourceLoader::load()` 的调用链",
        steps: [
          { title: "`load()`", text: "设置错误初值，决定当前线程模式，然后调用 `_load_start()`。" },
          { title: "`_validate_local_path()`", text: "把 UID 文本、相对路径或系统路径转成项目本地路径。" },
          { title: "`_load_start()`", text: "检查已有任务和 ResourceCache，创建 `ThreadLoadTask` 和 `LoadToken`。" },
          { title: "`_run_load_task()`", text: "同步模式下当前线程直接执行；线程模式下由 WorkerThreadPool 执行。" },
          { title: "`_path_remap()`", text: "处理翻译 remap、ResourceUID 和 `.remap` 文件。" },
          { title: "`_load()`", text: "遍历 loader 数组，找到能识别路径的 ResourceFormatLoader 并调用它。" },
          { title: "`_load_complete()`", text: "等待任务完成，返回任务里的 `Ref<Resource>` 和错误码。" }
        ]
      },
      {
        type: "paragraph",
        text: "`_validate_local_path()` 在 `core/io/resource_loader.cpp:477`。如果路径是 ResourceUID 文本，它会查 `ResourceUID` 得到真实路径；如果是相对路径，就拼成 `res://...`；否则用 `ProjectSettings::localize_path()` 尽量转成项目路径。读加载 bug 时第一步就要确认这里之后的 local_path 是什么。"
      },
      {
        type: "paragraph",
        text: "`_load()` 的核心逻辑很短：遍历 `loader[0..loader_count)`，先调用 `loader[i]->recognize_path(p_path, p_type_hint)`，匹配后调用 `loader[i]->load(...)`；如果返回有效资源就停止。`ResourceFormatLoader::recognize_path()` 在 `resource_loader.cpp:62`，默认按扩展名和 type hint 判断，也允许脚本虚函数覆盖。"
      },
      {
        type: "flow",
        title: "loader 选择过程",
        steps: [
          { title: "注册 loader", text: "core、scene、modules 在 register_types 阶段调用 `add_resource_format_loader()`。" },
          { title: "按顺序遍历", text: "ResourceLoader 维护最多 64 个 loader，`at_front=true` 的加载器优先。" },
          { title: "识别路径", text: "`recognize_path()` 看扩展名、type hint 或自定义识别逻辑。" },
          { title: "尝试加载", text: "匹配的 loader 执行 `load()`，失败则继续尝试后续 loader。" },
          { title: "返回第一个有效结果", text: "一旦得到有效 `Ref<Resource>`，ResourceLoader 停止遍历。" }
        ]
      },
      {
        type: "code",
        code: [
          "// resource_loader.cpp 的简化心智模型，不是逐字源码。",
          "Ref<Resource> ResourceLoader::_load(path, original_path, type_hint, cache_mode) {",
          "    for (int i = 0; i < loader_count; i++) {",
          "        if (!loader[i]->recognize_path(path, type_hint)) {",
          "            continue;",
          "        }",
          "",
          "        Ref<Resource> res = loader[i]->load(path, original_path, ...);",
          "        if (res.is_valid()) {",
          "            return res;",
          "        }",
          "    }",
          "    return Ref<Resource>();",
          "}"
        ].join("\n")
      },
      {
        type: "heading",
        title: "缓存模式：为什么同一路径常返回同一个对象"
      },
      {
        type: "paragraph",
        text: "默认 cache mode 是 `CACHE_MODE_REUSE`。`_load_start()` 在 `core/io/resource_loader.cpp:538` 会先查 `ResourceCache::get_ref(local_path)`；如果已有有效资源，任务直接标记为 `THREAD_LOAD_LOADED` 并返回缓存对象。`Resource::set_path()` 在 `core/io/resource.cpp:79` 会把资源路径登记到 `ResourceCache::resources`，这就是同一路径复用的基础。"
      },
      {
        type: "paragraph",
        text: "`CACHE_MODE_REPLACE` 的行为更微妙：如果加载完成时缓存里已有旧资源，ResourceLoader 会用新资源 `copy_from()` 到旧资源，再把任务结果替换成旧资源。这样外部已经持有旧资源引用的代码仍然看到同一个对象实例，只是内容被刷新。`IGNORE` 则不把结果登记进缓存，只设置 path cache。"
      },
      {
        type: "table",
        title: "CacheMode 的使用心智",
        headers: ["模式", "是否复用缓存", "加载成功后怎么处理路径", "适合场景"],
        rows: [
          ["`CACHE_MODE_REUSE`", "是。命中 ResourceCache 直接返回。", "资源设置 `set_path(local_path)` 并进入缓存。", "绝大多数运行时加载。"],
          ["`CACHE_MODE_IGNORE`", "否。即使有缓存也尝试新加载。", "只设置 `set_path_cache(local_path)`，不接管缓存路径。", "临时读取、比较文件、避免影响已有资源。"],
          ["`CACHE_MODE_REPLACE`", "不直接复用，但会关注已有缓存对象。", "如果已有对象，`copy_from()` 到旧对象并返回旧对象。", "编辑器刷新、热重载资源内容。"],
          ["`CACHE_MODE_IGNORE_DEEP`", "顶层和依赖都倾向忽略缓存。", "深层行为由具体 loader 传播处理。", "需要重新读完整依赖树。"],
          ["`CACHE_MODE_REPLACE_DEEP`", "顶层和依赖都倾向替换。", "深层行为由具体 loader 传播处理。", "批量刷新资源树。"]
        ]
      },
      {
        type: "heading",
        title: "路径 remap、UID 和导入产物"
      },
      {
        type: "paragraph",
        text: "`ResourceLoader::_path_remap()` 在 `core/io/resource_loader.cpp:1234`。它先看翻译 remap：例如同一张图片在不同 locale 下换成本地化版本；然后通过 `ResourceUID::ensure_path()` 处理 UID；最后检查 `path + \".remap\"` 文件，读取其中的 `path` 指向真正要加载的资源。"
      },
      {
        type: "paragraph",
        text: "`import_remap()` 在 `core/io/resource_loader.cpp:1322`，会询问 `ResourceFormatImporter` 是否识别该路径，如果识别，就返回导入后的内部资源路径。图片、压缩纹理等资源经常不是直接读原始源文件，而是读编辑器导入阶段生成的产物。"
      },
      {
        type: "flow",
        title: "路径可能被改写的地方",
        steps: [
          { title: "用户路径", text: "`res://icon.png`、相对路径或 `uid://...`。" },
          { title: "本地化", text: "`_validate_local_path()` 统一成项目路径。" },
          { title: "翻译 remap", text: "根据当前 locale 把资源换成本地化资源。" },
          { title: "ResourceUID", text: "`ResourceUID::ensure_path()` 把 UID 文本转成真实路径。" },
          { title: "`.remap` 文件", text: "导出或编辑器流程可把路径继续指到另一个资源。" },
          { title: "import remap", text: "需要时由 `ResourceFormatImporter` 指向导入缓存产物。" }
        ]
      },
      {
        type: "heading",
        title: "线程加载不是另一个 ResourceLoader"
      },
      {
        type: "paragraph",
        text: "线程加载仍然走同一套 ResourceLoader 逻辑，只是任务由 `WorkerThreadPool` 或当前线程执行。`load_threaded_request()` 调 `_load_start()`，传入 `LOAD_THREAD_SPAWN_SINGLE` 或 `LOAD_THREAD_DISTRIBUTE`；`load_threaded_get_status()` 查询 `ThreadLoadTask::status` 和依赖进度；`load_threaded_get()` 最终调用 `_load_complete_inner()` 领取结果。"
      },
      {
        type: "paragraph",
        text: "`_run_load_task()` 在 `core/io/resource_loader.cpp:337`。如果加载发生在非主线程，并且当前线程还没有自己的 MessageQueue，它会临时创建一个 `CallQueue` 作为 thread singleton override，加载结束后 flush 这个局部队列。资源加载期间产生的 changed 连接也会被收集，必要时通过主 MessageQueue 迁回主线程。"
      },
      {
        type: "flow",
        title: "线程加载请求的生命周期",
        steps: [
          { title: "发起请求", text: "`load_threaded_request(path)` 创建或复用 user token。" },
          { title: "登记任务", text: "`thread_load_tasks[local_path]` 保存状态、进度、错误、结果和依赖子任务。" },
          { title: "后台执行", text: "WorkerThreadPool 调 `_run_load_task()`，里面仍走 `_path_remap()` 和 `_load()`。" },
          { title: "轮询状态", text: "`load_threaded_get_status()` 返回 in progress/failed/loaded，并计算依赖进度。" },
          { title: "领取结果", text: "`load_threaded_get()` 等任务结束，调用 `_load_complete_inner()` 返回 `Ref<Resource>`。" },
          { title: "清理 token", text: "用户引用计数归零后从 `user_load_tokens` 移除。" }
        ]
      },
      {
        type: "table",
        title: "线程加载状态",
        headers: ["状态", "含义", "调用者该怎么理解"],
        rows: [
          ["`THREAD_LOAD_INVALID_RESOURCE`", "没有这个用户请求，或结果已经被领取。", "路径不匹配、未 request、重复 get 后再查都可能出现。"],
          ["`THREAD_LOAD_IN_PROGRESS`", "任务还在执行。", "可以继续轮询进度；同一帧主线程反复轮询可能触发 `_ensure_load_progress()`。"],
          ["`THREAD_LOAD_FAILED`", "加载任务完成但失败。", "调用 `load_threaded_get()` 也只能得到空资源和错误码。"],
          ["`THREAD_LOAD_LOADED`", "资源已经加载成功。", "可以调用 `load_threaded_get()` 领取结果。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：加载并实例化场景"
      },
      {
        type: "code",
        code: [
          "func spawn_level() -> void:",
          "    var packed := ResourceLoader.load(\"res://levels/main.tscn\", \"PackedScene\") as PackedScene",
          "    if packed == null:",
          "        push_error(\"Level load failed\")",
          "        return",
          "",
          "    var root := packed.instantiate()",
          "    add_child(root)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这里 ResourceLoader 返回的是 `PackedScene` 资源，不是已经进树的 Node。真正创建节点树发生在 `PackedScene.instantiate()`，进入 SceneTree 则发生在 `add_child()`。"
      },
      {
        type: "subheading",
        title: "案例二：后台加载大场景"
      },
      {
        type: "code",
        code: [
          "var path := \"res://levels/boss_room.tscn\"",
          "",
          "func _ready() -> void:",
          "    ResourceLoader.load_threaded_request(path, \"PackedScene\")",
          "",
          "func _process(_delta: float) -> void:",
          "    var progress := []",
          "    var status := ResourceLoader.load_threaded_get_status(path, progress)",
          "    if status == ResourceLoader.THREAD_LOAD_LOADED:",
          "        var packed := ResourceLoader.load_threaded_get(path) as PackedScene",
          "        add_child(packed.instantiate())"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`load_threaded_get()` 不是纯查询；如果资源还没完成，它可能等待。用户界面里通常先用 status 判断，再在 loaded 时 get。"
      },
      {
        type: "subheading",
        title: "案例三：C++ 同步加载"
      },
      {
        type: "code",
        code: [
          "Error err = OK;",
          "Ref<Resource> res = ResourceLoader::load(",
          "        \"res://levels/main.tscn\",",
          "        \"PackedScene\",",
          "        ResourceLoader::CACHE_MODE_REUSE,",
          "        &err);",
          "",
          "Ref<PackedScene> scene = res;",
          "if (scene.is_valid()) {",
          "    Node *root = scene->instantiate();",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`type_hint` 帮助选择 loader，但你仍然应该检查返回资源是否能转换成目标类型。加载失败时 `Ref<Resource>` 为空，`err` 带错误原因。"
      },
      {
        type: "subheading",
        title: "案例四：注册一个格式加载器"
      },
      {
        type: "code",
        code: [
          "Ref<MyResourceFormatLoader> loader;",
          "loader.instantiate();",
          "",
          "// at_front=true 会让它在已有 loader 前面优先匹配。",
          "ResourceLoader::add_resource_format_loader(loader, true);"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Godot 自己也在注册阶段这样添加加载器。比如 `scene/register_scene_types.cpp:418` 把文本资源加载器加到前面，使 `.tscn`、`.tres` 这类文本资源能被优先识别。"
      },
      {
        type: "heading",
        title: "ResourceLoader 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Resource", "被加载出来的资源对象，通常继承 RefCounted。", "不负责全局选择哪个加载器。"],
          ["ResourceLoader", "路径标准化、缓存策略、任务管理、格式加载器调度。", "不直接解析每一种文件格式。"],
          ["ResourceFormatLoader", "识别并解析某类资源文件。", "不管理所有加载任务和用户线程 API。"],
          ["ResourceCache", "按路径保存当前已加载资源。", "不是导入缓存，也不是磁盘文件。"],
          ["ResourceUID", "把稳定 UID 映射到真实资源路径。", "不解析资源内容。"],
          ["ResourceFormatImporter", "把源资源路径映射到导入产物。", "不是用户脚本通常直接调用的运行时加载入口。"],
          ["PackedScene", "加载出来的场景模板资源。", "不是已经挂到 SceneTree 的 Node。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：ResourceLoader 就是读文件函数。它是加载调度器，具体解析由 ResourceFormatLoader 完成。",
          "误区二：`load(\"res://x.tscn\")` 返回 Node。它返回 PackedScene；要 `instantiate()` 后才有节点。",
          "误区三：同一路径每次都会生成新对象。默认 `CACHE_MODE_REUSE` 会复用 ResourceCache。",
          "误区四：type hint 等于强制类型。它帮助选择加载器，但返回值仍要检查和转换。",
          "误区五：线程加载永远不阻塞。`load_threaded_get()` 在结果未完成时可能等待。",
          "误区六：运行时读的一定是源文件。路径可能经过 UID、翻译 remap、`.remap` 或导入产物改写。",
          "误区七：ResourceCache 是磁盘缓存。它是当前进程内按路径登记的 Resource 对象表。",
          "误区八：后台加载可以顺便改 SceneTree。资源加载可以在线程里跑，但 SceneTree/Node 操作仍要尊重主线程和安全点。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/io/resource_loader.h:48` 和 `:103`，区分 ResourceFormatLoader 和 ResourceLoader。",
          "读 `ResourceLoader::load()`、`_load_start()`、`_run_load_task()`、`_load_complete_inner()`，串起同步和线程加载共用的任务模型。",
          "读 `_validate_local_path()` 和 `_path_remap()`，确认路径为什么可能不是你传入的原始字符串。",
          "读 `_load()` 和 `ResourceFormatLoader::recognize_path()`，理解 loader 数组如何选择具体解析器。",
          "读 `Resource::set_path()` 和 `ResourceCache::get_ref()`，把 cache mode 和路径登记对上。",
          "最后读 `scene/resources/resource_format_text.cpp:1422` 或 `core/io/resource_format_binary.cpp:1148`，看具体格式加载器如何解析依赖和子资源。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "ResourceLoader 是 Godot 的资源加载前台和调度器：它负责路径、缓存、任务和格式加载器选择；真正把文件解析成 Resource 的，是背后的 ResourceFormatLoader。"
      }
    ]
  },
  {
    id: "packedscene",
    title: "PackedScene",
    aliases: ["PackedScene", "SceneState", "场景资源", ".tscn", ".scn", "instantiate", "pack", "_bundled"],
    summary: "Godot 的场景模板资源：文件加载后变成 PackedScene/SceneState，调用 instantiate() 才创建真正的 Node 树。",
    article: [
      {
        type: "lead",
        text: "PackedScene 是 Godot 场景文件加载后的资源对象。`.tscn` 或 `.scn` 不是直接变成一棵正在运行的节点树，而是先变成一个 PackedScene；PackedScene 内部持有 SceneState，SceneState 保存节点表、属性表、连接表、资源引用、owner 和子场景实例信息。只有调用 `instantiate()`，才会按这些数据创建真实 Node 对象。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 PackedScene 想成“场景蓝图”。蓝图里写着：根节点是什么类型、有哪些子节点、每个节点有哪些属性、谁连了谁的信号、哪些子场景被嵌进来了。但蓝图本身不是房子。你要调用 `instantiate()`，Godot 才会照着蓝图盖出一棵新的 Node 树。"
      },
      {
        type: "paragraph",
        text: "`ResourceLoader.load(\"res://main.tscn\")` 返回的是 PackedScene，不是 Node；`packed.instantiate()` 返回 Node 树，但这棵树还没有进入 SceneTree；只有再 `add_child(root)`，节点才进入运行中的场景树并开始收到 enter_tree、ready、process 等生命周期通知。"
      },
      {
        type: "flow",
        title: "从 .tscn 到运行中节点的三段",
        steps: [
          { title: "磁盘文件", text: "`.tscn`/`.scn` 保存节点、属性、连接和资源。" },
          { title: "PackedScene", text: "ResourceLoader 加载后得到一个 PackedScene 资源。" },
          { title: "SceneState", text: "PackedScene 内部的 SceneState 保存可实例化数据。" },
          { title: "instantiate", text: "按 SceneState 创建一棵新的 Node 子树。" },
          { title: "add_child", text: "把 Node 子树接入 SceneTree，生命周期才开始运行。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`PackedScene` 声明在 `scene/resources/packed_scene.h:246`，继承 `Resource`，内部只有一个关键字段：`Ref<SceneState> state`。`SceneState` 声明在 `scene/resources/packed_scene.h:38`，它不是 Object/Node，而是 RefCounted 数据对象，专门保存“如何重建节点树”。"
      },
      {
        type: "paragraph",
        text: "这层拆分很重要：PackedScene 负责作为 Resource 暴露给 ResourceLoader、ResourceCache、脚本和编辑器；SceneState 负责保存和解析节点、属性、信号连接、NodePath、子资源和编辑器状态。"
      },
      {
        type: "table",
        title: "PackedScene 和 SceneState 的分工",
        headers: ["组件", "源码位置", "负责什么", "不负责什么"],
        rows: [
          ["`PackedScene`", "`scene/resources/packed_scene.h:246`", "Resource 外壳，提供 `pack()`、`instantiate()`、`get_state()`、路径和重载逻辑。", "不直接保存每个节点字段的细节。"],
          ["`SceneState`", "`scene/resources/packed_scene.h:38`", "保存节点表、属性表、连接表、NodePath、子场景实例和 editable instances。", "不是运行中的场景树。"],
          ["`NodeData`", "`packed_scene.h:58`", "单个节点的 parent、owner、type、name、instance、index、properties、groups。", "不保存真实 Node 指针。"],
          ["`ConnectionData`", "`packed_scene.h:83`", "信号连接的 from/to/signal/method/flags/unbinds/binds。", "不在加载阶段立即发射信号。"],
          ["`_bundled`", "`packed_scene.cpp:2621`", "序列化用的内部 Dictionary 属性，保存 SceneState 的压缩表示。", "不是用户应手写依赖的公开业务数据格式。"]
        ]
      },
      {
        type: "table",
        title: "SceneState 保存了哪些表",
        headers: ["表/字段", "含义", "为什么不直接保存对象"],
        rows: [
          ["`names`", "节点名、类型名、属性名、组名、信号名、方法名的 StringName 池。", "用索引复用字符串，减少重复。"],
          ["`variants`", "属性值、资源引用、子场景引用、绑定参数等 Variant 池。", "统一保存不同类型的值。"],
          ["`node_paths` / `id_paths`", "跨节点引用、owner、连接端点等路径。", "场景文件不能长期保存运行时指针。"],
          ["`nodes`", "每个节点的 NodeData。", "重建时按表顺序创建或复用节点。"],
          ["`connections`", "持久化信号连接。", "实例化后再按 NodePath/索引找到双方并 connect。"],
          ["`editable_instances`", "编辑器里允许展开编辑的子场景实例路径。", "运行时通常不需要这层编辑状态。"],
          ["`base_scene_idx`", "继承场景的基场景资源索引。", "支持 inherited scene 只保存差异。"]
        ]
      },
      {
        type: "heading",
        title: "加载：文件如何变成 PackedScene"
      },
      {
        type: "paragraph",
        text: "文本场景由 `ResourceFormatLoaderText` 解析。`ResourceLoaderText::_parse_node_tag()` 在 `scene/resources/resource_format_text.cpp:184`，遇到 `[node]` 标签时调用 `SceneState::add_name()`、`add_node_path()`、`add_value()`、`add_node()`、`add_node_property()` 等 build API，把文本标签转成 SceneState 内部表。遇到 `[connection]` 标签时调用 `add_connection()`。"
      },
      {
        type: "paragraph",
        text: "二进制资源也会走类似结果：`ResourceFormatLoaderBinary::load()` 在 `core/io/resource_format_binary.cpp:1148`，如果资源类型是 PackedScene，会拿到 PackedScene 的 state 并填入场景数据。也就是说，`.tscn` 和 `.scn` 文件格式不同，但加载后的抽象目标都是 PackedScene/SceneState。"
      },
      {
        type: "flow",
        title: "文本 .tscn 解析成 SceneState",
        steps: [
          { title: "ResourceLoader 选中 Text loader", text: "`.tscn` 被 ResourceFormatLoaderText 识别为 PackedScene。" },
          { title: "读取 node 标签", text: "`_parse_node_tag()` 读取 name、type、parent、owner、instance、groups。" },
          { title: "写入节点表", text: "调用 `add_node()` 保存 parent/owner/type/name/instance/index/unique_id。" },
          { title: "读取属性赋值", text: "每个属性名和值进入 names/variants，再挂到 NodeData.properties。" },
          { title: "读取 connection 标签", text: "from/to/signal/method/binds/flags 进入 ConnectionData。" },
          { title: "返回 PackedScene", text: "解析完成后 ResourceLoader 得到可缓存的 PackedScene Resource。" }
        ]
      },
      {
        type: "code",
        code: [
          "# .tscn 的极简示意，不是完整文件。",
          "[gd_scene format=3]",
          "",
          "[node name=\"Main\" type=\"Node2D\"]",
          "",
          "[node name=\"Player\" type=\"CharacterBody2D\" parent=\".\"]",
          "position = Vector2(64, 96)",
          "",
          "[connection signal=\"pressed\" from=\"Button\" to=\".\" method=\"_on_button_pressed\"]"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这些文本标签不会直接创建 Node。加载器只是把它们拆进 SceneState：节点、属性和连接分别进不同表，等 instantiate 时再重建。"
      },
      {
        type: "heading",
        title: "实例化：蓝图如何变成 Node 树"
      },
      {
        type: "paragraph",
        text: "`PackedScene::instantiate()` 在 `scene/resources/packed_scene.cpp:2507`，它调用 `state->instantiate()`；实例化成功后，如果是编辑器 edit state，会给根节点设置 scene instance state；如果 PackedScene 不是 built-in 资源，还会把 `scene_file_path` 写到根节点；最后发送 `Node::NOTIFICATION_SCENE_INSTANTIATED`。"
      },
      {
        type: "paragraph",
        text: "`SceneState::instantiate()` 在 `packed_scene.cpp:155`。它按 `nodes` 顺序创建或复用节点：普通节点通过 `ClassDB::instantiate(type)` 创建；子场景实例会拿 `PackedScene` 再递归 instantiate；继承或实例覆盖里的 `TYPE_INSTANTIATED` 节点会尝试从已有实例里找回。所有节点创建完后，再处理 deferred NodePath 属性、local-to-scene 资源、信号连接和 editable instances。"
      },
      {
        type: "flow",
        title: "`PackedScene.instantiate()` 的主要步骤",
        steps: [
          { title: "检查 SceneState", text: "`can_instantiate()` 要求 node count 大于 0。" },
          { title: "按节点表循环", text: "每个 NodeData 决定创建新节点、实例化子场景、复用继承节点或创建 MissingNode。" },
          { title: "恢复属性和脚本", text: "按 properties 取 Variant 值，必要时处理 script、MissingResource、Array/Dictionary。" },
          { title: "建立父子关系", text: "调用 `_add_child_nocheck()`，设置名称、index 和 owner。" },
          { title: "延后解析 Node 引用", text: "Node 类型属性先存 NodePath，所有节点创建完后再转回 Node。" },
          { title: "连接信号", text: "遍历 ConnectionData，用 Callable 重新 connect。" },
          { title: "返回根节点", text: "得到一棵还未必进入 SceneTree 的 Node 子树。" }
        ]
      },
      {
        type: "table",
        title: "instantiate 时几类节点怎么处理",
        headers: ["NodeData 情况", "源码行为", "结果"],
        rows: [
          ["根节点继承 base scene", "`base_scene_idx >= 0` 时先实例化基 PackedScene。", "根节点来自基场景，再叠加当前场景差异。"],
          ["`n.instance >= 0`", "把 variants 里的 PackedScene 或 placeholder 路径实例化。", "子场景作为当前场景的一个节点分支。"],
          ["`n.type == TYPE_INSTANTIATED`", "从已有实例/继承场景中按名称或 unique scene id 找节点。", "不新建节点，只恢复覆盖数据。"],
          ["普通节点", "调用 `ClassDB::instantiate(snames[n.type])`。", "创建对应 C++ Node 类型。"],
          ["类型缺失", "可创建 MissingNode，或退化成 Node/Node2D/Node3D/Control placeholder。", "编辑器能保留未知节点数据，避免立刻丢失。"]
        ]
      },
      {
        type: "heading",
        title: "pack：运行中的 Node 树如何变回场景资源"
      },
      {
        type: "paragraph",
        text: "`PackedScene::pack()` 在 `packed_scene.cpp:2473`，只是转发给 `SceneState::pack()`。`SceneState::pack()` 在 `packed_scene.cpp:1343`，会先 `clear()`，然后递归 `_parse_node()` 保存节点，再 `_parse_connections()` 保存持久化连接，最后把 name_map、variant_map、nodepath_map 写回 `names`、`variants`、`node_paths` 和 `id_paths`。"
      },
      {
        type: "paragraph",
        text: "`_parse_node()` 在 `packed_scene.cpp:792`。它只保存属于当前 owner 的节点，或 editable instance 允许保存的子场景内容。属性只保存 `PROPERTY_USAGE_STORAGE` 或缺失资源相关数据，并且会和默认值、继承/实例状态比较，没变化的通常不保存。Node 类型属性会先转成 NodePath，等 instantiate 完再转回 Node。"
      },
      {
        type: "flow",
        title: "`PackedScene.pack(root)` 的路径",
        steps: [
          { title: "清空旧 state", text: "`SceneState::pack()` 先 `clear()`。" },
          { title: "处理继承场景", text: "如果根节点有 inherited state，先把基 PackedScene 存进 variants。" },
          { title: "递归解析节点", text: "`_parse_node()` 保存当前场景拥有的节点和必要覆盖数据。" },
          { title: "收集属性和组", text: "只保存可存储、持久化、与默认/基状态不同的数据。" },
          { title: "解析持久连接", text: "`_parse_connections()` 保存 `CONNECT_PERSIST` 的信号连接。" },
          { title: "压缩索引表", text: "把 name/variant/nodepath 映射写回 SceneState 的数组。" }
        ]
      },
      {
        type: "table",
        title: "pack 时会保存什么",
        headers: ["数据", "保存条件", "容易踩坑"],
        rows: [
          ["节点", "属于当前 scene owner，或是 editable instance 的必要数据。", "只 add_child 不 set owner，通常不会保存进当前场景。"],
          ["属性", "有 `PROPERTY_USAGE_STORAGE`，且不是默认值或需要保留 local resource。", "运行时临时状态、非存储属性不会进 .tscn。"],
          ["组", "节点组必须是 persistent。", "运行时临时 group 不会保存。"],
          ["信号连接", "连接必须带 `CONNECT_PERSIST`，并且不重复保存来自基场景/实例的连接。", "纯运行时 connect 默认不一定被场景保存。"],
          ["子场景实例", "保存 PackedScene 引用或 placeholder 路径，以及本地覆盖。", "子场景本体不会被复制成普通节点，除非 editable/inherited 规则要求保存差异。"],
          ["Node 引用属性", "先保存为 NodePath 或 id path。", "实例化后才恢复成真实 Node 对象。"]
        ]
      },
      {
        type: "heading",
        title: "_bundled：SceneState 的序列化接口"
      },
      {
        type: "paragraph",
        text: "PackedScene 绑定了一个内部属性 `_bundled`，位置在 `packed_scene.cpp:2621`。它的 setter/getter 分别调用 `SceneState::set_bundled_scene()` 和 `get_bundled_scene()`。这两个函数把 SceneState 压成 Dictionary：`names`、`variants`、`node_count`、`nodes`、`conn_count`、`conns`、`node_paths`、`id_paths`、`editable_instances`、`base_scene`、`version`。"
      },
      {
        type: "paragraph",
        text: "文本和二进制资源保存器可以把这个结构写进文件，加载器也可以从文件重建它。你读 .tscn 时看到的是可读文本标签；你读 PackedScene 内部时看到的是更适合运行时构建的索引数组。"
      },
      {
        type: "table",
        title: "_bundled 字典和 SceneState 字段",
        headers: ["字典键", "对应字段", "说明"],
        rows: [
          ["`names`", "`names`", "StringName 池。"],
          ["`variants`", "`variants`", "属性值、资源、绑定参数。"],
          ["`nodes` / `node_count`", "`nodes`", "压平的 NodeData 数组。"],
          ["`conns` / `conn_count`", "`connections`", "压平的 ConnectionData 数组。"],
          ["`node_paths`", "`node_paths`", "NodePath 池。"],
          ["`id_paths` / `node_ids`", "`id_paths` / `ids`", "unique scene id 辅助恢复路径。"],
          ["`editable_instances`", "`editable_instances`", "编辑器可展开实例。"],
          ["`base_scene`", "`base_scene_idx`", "继承场景资源索引。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：加载蓝图并创建实例"
      },
      {
        type: "code",
        code: [
          "func spawn_enemy() -> void:",
          "    var packed := load(\"res://enemy.tscn\") as PackedScene",
          "    if packed == null:",
          "        return",
          "",
          "    var enemy := packed.instantiate()",
          "    add_child(enemy)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`load()` 得到蓝图，`instantiate()` 得到节点，`add_child()` 才进入树。这三个动作不要混在一起理解。"
      },
      {
        type: "subheading",
        title: "案例二：运行时打包一个节点树"
      },
      {
        type: "code",
        code: [
          "func save_branch(root: Node) -> void:",
          "    var packed := PackedScene.new()",
          "    var err := packed.pack(root)",
          "    if err == OK:",
          "        ResourceSaver.save(packed, \"res://saved_branch.tscn\")"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "如果子节点没有正确设置 `owner`，它可能在树里能运行，却不会被 pack 进场景。编辑器里“动态节点没保存”常常就是 owner 问题。"
      },
      {
        type: "subheading",
        title: "案例三：检查 PackedScene 的 SceneState"
      },
      {
        type: "code",
        code: [
          "var packed := load(\"res://main.tscn\") as PackedScene",
          "var state := packed.get_state()",
          "",
          "for i in state.get_node_count():",
          "    print(i, state.get_node_path(i), state.get_node_type(i))"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这适合调试“场景文件里到底保存了哪些节点”。它读的是 SceneState 表，不会把场景实例化进当前 SceneTree。"
      },
      {
        type: "subheading",
        title: "案例四：C++ 里实例化场景"
      },
      {
        type: "code",
        code: [
          "Ref<PackedScene> packed = ResourceLoader::load(\"res://ui/menu.tscn\", \"PackedScene\");",
          "if (packed.is_valid() && packed->can_instantiate()) {",
          "    Node *menu = packed->instantiate();",
          "    parent->add_child(menu);",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "C++ 里同样要区分 Resource 和 Node。`Ref<PackedScene>` 是资源强引用，`instantiate()` 返回裸 `Node *`，之后通常交给父节点/SceneTree 管生命周期。"
      },
      {
        type: "heading",
        title: "PackedScene 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["ResourceLoader", "按路径加载 `.tscn`/`.scn` 并返回 PackedScene。", "不把场景直接挂进 SceneTree。"],
          ["PackedScene", "作为 Resource 保存一份 SceneState，并提供 pack/instantiate。", "不是运行中的节点树。"],
          ["SceneState", "保存节点、属性、连接、路径、实例和编辑信息。", "不是脚本运行环境，也不接收 process。"],
          ["Node", "实例化后真实存在的运行时对象。", "不自动保存进 PackedScene，除非 pack 且 owner/属性条件满足。"],
          ["SceneTree", "接管已 add_child 的 Node 生命周期和每帧调度。", "不解析 .tscn 文件。"],
          ["owner", "决定 pack 时节点属于哪个可保存场景。", "不是 parent，也不是内存所有权。"],
          ["ResourceCache", "复用同一路径的 PackedScene 资源。", "不复用每次 instantiate 生成的 Node 树。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：PackedScene 是 Node。它是 Resource；实例化后才得到 Node。",
          "误区二：ResourceLoader.load 会自动把场景加入树。它只返回 PackedScene，进入树要 add_child。",
          "误区三：同一个 PackedScene 只能实例化一次。它是蓝图，可以多次 instantiate 出多棵 Node 树。",
          "误区四：多次 instantiate 会复制所有资源。普通 Resource 可能共享；`local_to_scene` 资源会按规则复制或配置。",
          "误区五：pack 会保存树下所有节点。只有 owner/可编辑实例/覆盖规则允许的数据会保存。",
          "误区六：运行时属性都会进 .tscn。只有可存储属性、持久 group 和持久连接会保存。",
          "误区七：NodePath 属性一开始就是 Node 指针。文件里保存路径，实例化后等所有节点存在了再恢复引用。",
          "误区八：编辑器 edit_state 可以在普通模板构建里随便用。非 tools 构建下 edit_state 不是运行时功能。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/resources/packed_scene.h:38` 的 SceneState 字段，理解它保存的是表而不是 Node 指针。",
          "读 `PackedScene` 类声明，确认它只是 Resource 外壳和 `Ref<SceneState>`。",
          "读 `ResourceLoaderText::_parse_node_tag()`，把 .tscn 标签如何写进 SceneState 对上。",
          "读 `PackedScene::instantiate()` 和 `SceneState::instantiate()`，理解蓝图如何创建 Node 树。",
          "读 `SceneState::pack()`、`_parse_node()`、`_parse_connections()`，理解保存场景为什么依赖 owner、storage 属性和 persistent 连接。",
          "最后读 `_bundled` 的 getter/setter，理解场景文件和 SceneState 内部表之间如何互相转换。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "PackedScene 是可实例化的场景蓝图；SceneState 保存蓝图细节，`instantiate()` 才创建 Node 树，`pack()` 则把符合保存规则的 Node 树重新压回蓝图。"
      }
    ]
  },
  {
    id: "scenestate",
    title: "SceneState",
    aliases: ["SceneState", "场景状态", "节点表", "连接表", "_bundled", "NodeData", "ConnectionData"],
    summary: "PackedScene 内部的场景数据表：用 names、variants、nodes、connections、node_paths 等数组描述一棵可实例化的 Node 树。",
    article: [
      {
        type: "lead",
        text: "SceneState 是 PackedScene 里的“蓝图数据表”。它不运行脚本、不进入 SceneTree，也不保存真实 Node 指针；它用一组紧凑数组记录场景节点、属性、组、信号连接、NodePath、子场景实例和继承/编辑信息。PackedScene 的 `instantiate()` 和 `pack()` 本质上都是围绕 SceneState 做“读表建树”和“读树填表”。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 PackedScene 是一本施工说明书，SceneState 就是说明书里的表格页：第 0 行是什么节点，第 1 行的父节点是谁，第 2 行有哪些属性，哪个按钮的 pressed 信号连到哪个方法。Godot 不在文件里保存“真实节点对象”，而是保存这些表格，等需要时再照表创建对象。"
      },
      {
        type: "paragraph",
        text: "理解 SceneState 的关键是：它保存的是索引。节点类型、属性名、信号名都不直接重复写字符串，而是先放进 `names`；属性值、资源引用、绑定参数先放进 `variants`；节点表和连接表只保存这些池子的下标。"
      },
      {
        type: "flow",
        title: "SceneState 的表格心智模型",
        steps: [
          { title: "名字池", text: "`names` 保存 Main、Player、position、pressed 等 StringName。" },
          { title: "值池", text: "`variants` 保存 Vector2、Resource、PackedScene、绑定参数等 Variant。" },
          { title: "节点表", text: "`nodes` 每行描述一个节点：父节点、owner、类型、名字、属性、组。" },
          { title: "连接表", text: "`connections` 每行描述一条持久信号连接。" },
          { title: "路径表", text: "`node_paths` 和 `id_paths` 辅助跨节点引用和路径恢复。" },
          { title: "实例化", text: "Godot 读这些表，重新创建 Node 树。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`SceneState` 声明在 `scene/resources/packed_scene.h:38`，继承 `RefCounted`。它的核心字段包括 `names`、`variants`、`node_paths`、`id_paths`、`ids`、`editable_instances`、`nodes` 和 `connections`。这些字段构成了 PackedScene 的真实内容。"
      },
      {
        type: "paragraph",
        text: "SceneState 同时提供两类 API：加载器和 pack 过程使用 build API，例如 `add_name()`、`add_value()`、`add_node()`、`add_connection()`；编辑器、调试器或脚本检查场景内容时使用 unbuild API，例如 `get_node_count()`、`get_node_path()`、`get_node_property_value()`、`get_connection_source()`。"
      },
      {
        type: "table",
        title: "SceneState 的主要数组",
        headers: ["字段", "保存什么", "典型读取位置"],
        rows: [
          ["`names`", "节点名、类型名、属性名、组名、信号名、方法名。", "`get_node_type()`、`get_node_name()`、`get_connection_signal()`。"],
          ["`variants`", "属性值、资源引用、子场景 PackedScene、连接 binds。", "`get_node_property_value()`、`get_connection_binds()`。"],
          ["`node_paths`", "parent、owner、连接 from/to、Node 引用属性的路径。", "`get_node_path()`、`get_connection_source()`。"],
          ["`id_paths` / `ids`", "unique scene id 路径，用来在路径失效时恢复节点。", "`_recover_node_path_index()`、`get_node_id_path()`。"],
          ["`nodes`", "一行一个 NodeData。", "`SceneState::instantiate()` 和 unbuild API。"],
          ["`connections`", "一行一个 ConnectionData。", "实例化末尾重新 connect。"],
          ["`editable_instances`", "编辑器可展开子场景实例路径。", "实例化结束后 `set_editable_instance()`。"],
          ["`base_scene_idx`", "继承场景的 PackedScene 在 variants 中的索引。", "继承场景实例化和 `get_base_scene_state()`。"]
        ]
      },
      {
        type: "table",
        title: "NodeData 一行代表什么",
        headers: ["字段", "含义", "注意点"],
        rows: [
          ["`parent`", "父节点索引，或带 `FLAG_ID_IS_PATH` 的 NodePath 索引。", "跨实例/继承边界时可能不是普通节点下标。"],
          ["`owner`", "保存场景时的 owner 信息。", "影响 pack/保存，不等于 parent。"],
          ["`type`", "节点类型名在 names 里的索引，或 `TYPE_INSTANTIATED`。", "TYPE_INSTANTIATED 表示节点来自已有实例/继承，不应新建。"],
          ["`name`", "节点名在 names 里的索引。", "get_node_path 依赖它拼路径。"],
          ["`instance`", "子场景 PackedScene 或 placeholder 路径在 variants 里的索引。", "可带 `FLAG_INSTANCE_IS_PLACEHOLDER`。"],
          ["`properties`", "属性名和值的索引对。", "Node 引用属性会用 `FLAG_PATH_PROPERTY_IS_NODE` 标记。"],
          ["`groups`", "持久组名索引。", "非 persistent group 不会进入 SceneState。"]
        ]
      },
      {
        type: "table",
        title: "ConnectionData 一行代表什么",
        headers: ["字段", "含义", "实例化时怎么用"],
        rows: [
          ["`from` / `to`", "信号源和目标节点索引，或 NodePath 索引。", "用 `NODE_FROM_ID` 找回 cfrom/cto。"],
          ["`signal`", "信号名在 names 里的索引。", "传给 `connect()` 的信号名。"],
          ["`method`", "目标方法名在 names 里的索引。", "构造 `Callable(cto, method)`。"],
          ["`flags`", "连接 flags。", "实例化时会加 `CONNECT_PERSIST` 和可能的 `CONNECT_INHERITED`。"],
          ["`unbinds`", "Callable unbind 参数数量。", "恢复 `callable.unbind(unbinds)`。"],
          ["`binds`", "绑定参数在 variants 里的索引列表。", "恢复 `callable.bindv(binds)`。"]
        ]
      },
      {
        type: "heading",
        title: "build API：谁会往 SceneState 里写数据"
      },
      {
        type: "paragraph",
        text: "加载 `.tscn` 时，`ResourceLoaderText::_parse_node_tag()` 会把文本标签翻译成 build API 调用：`add_name()`、`add_node_path()`、`add_value()`、`add_node()`、`add_node_property()`、`add_node_group()` 和 `add_connection()`。这些函数集中在 `scene/resources/packed_scene.cpp:2300` 附近。"
      },
      {
        type: "paragraph",
        text: "`PackedScene.pack(root)` 也会写 SceneState，但路径不同：`SceneState::pack()` 先递归 `_parse_node()` 和 `_parse_connections()`，收集临时 `name_map`、`variant_map`、`node_map`、`nodepath_map`，最后一次性填回 `names`、`variants`、`nodes`、`connections`、`node_paths` 和 `id_paths`。"
      },
      {
        type: "flow",
        title: "两条写入 SceneState 的路径",
        steps: [
          { title: "加载文件", text: "ResourceFormatLoaderText/Binary 解析文件。" },
          { title: "调用 build API", text: "按标签直接 `add_node()`、`add_value()`、`add_connection()`。" },
          { title: "得到 SceneState", text: "PackedScene 持有填好的 state。" },
          { title: "pack 运行中节点", text: "`SceneState::pack()` 遍历 Node 树。" },
          { title: "收集映射", text: "名字、值、路径先进入临时 map 去重。" },
          { title: "写回数组", text: "最终生成和加载器相同形态的 SceneState 表。" }
        ]
      },
      {
        type: "code",
        code: [
          "// SceneState build API 的简化心智模型。",
          "int name = state->add_name(\"Player\");",
          "int type = state->add_name(\"CharacterBody2D\");",
          "int value = state->add_value(Vector2(64, 96));",
          "",
          "int node = state->add_node(parent, owner, type, name, instance, index, unique_id);",
          "state->add_node_property(node, state->add_name(\"position\"), value);"
        ].join("\n")
      },
      {
        type: "heading",
        title: "unbuild API：如何不实例化也能查看场景"
      },
      {
        type: "paragraph",
        text: "SceneState 暴露了一组只读查询方法，绑定在 `SceneState::_bind_methods()` 的 `packed_scene.cpp:2410` 附近。脚本可以通过 `packed.get_state()` 获取 SceneState，然后调用 `get_node_count()`、`get_node_path()`、`get_node_type()`、`get_node_property_count()`、`get_connection_count()` 等方法检查场景内容。"
      },
      {
        type: "paragraph",
        text: "这些查询不会创建 Node。它们只是把内部表的索引翻译回可读信息：例如 `get_node_path()` 从当前节点沿 parent 链向上拼出 NodePath；`get_node_property_name()` 用 `FLAG_PROP_NAME_MASK` 去掉 Node 引用标记后查 `names`；`get_connection_binds()` 把 binds 索引转回 Variant。"
      },
      {
        type: "table",
        title: "常用 unbuild API",
        headers: ["API", "读哪张表", "用途"],
        rows: [
          ["`get_node_count()`", "`nodes`", "知道场景保存了多少个节点记录。"],
          ["`get_node_path(i)`", "`nodes` + `node_paths` + `names`", "得到第 i 个节点相对根的路径。"],
          ["`get_node_type(i)`", "`nodes[i].type` + `names`", "知道节点类型；TYPE_INSTANTIATED 会返回空名。"],
          ["`get_node_property_count(i)`", "`nodes[i].properties`", "遍历某节点保存的属性。"],
          ["`get_node_property_value(i, p)`", "`variants`", "读属性值。"],
          ["`get_connection_count()`", "`connections`", "遍历持久信号连接。"],
          ["`get_connection_source(i)` / `target(i)`", "`connections` + `node_paths`", "读连接两端的路径。"],
          ["`get_sub_resources()`", "`variants`", "查内置子资源。"]
        ]
      },
      {
        type: "heading",
        title: "instantiate 如何读 SceneState"
      },
      {
        type: "paragraph",
        text: "`SceneState::instantiate()` 不直接读取 .tscn 文本，而是读 SceneState 表。它先准备 `ret_nodes` 数组，长度等于 node count；之后按顺序处理每个 NodeData：找到 parent，决定创建、实例化、复用或恢复节点，再设置属性、组、名字、父子关系和 owner。"
      },
      {
        type: "paragraph",
        text: "Node 引用属性会被特殊处理。保存时 `_parse_node()` 把 Node 对象属性转成 NodePath，并给属性名加 `FLAG_PATH_PROPERTY_IS_NODE`；实例化时先把这些属性放进 `deferred_node_paths`，等所有节点都创建完成后，再用 `get_node_or_null()` 把路径转回真实 Node。"
      },
      {
        type: "flow",
        title: "读表建树的顺序",
        steps: [
          { title: "准备 ret_nodes", text: "每个 SceneState 节点记录对应一个运行时 Node 指针槽位。" },
          { title: "创建或复用节点", text: "普通节点用 ClassDB，新实例用 PackedScene，继承覆盖用已有节点。" },
          { title: "设置属性", text: "按 properties 从 variants 取值并调用 `node->set()`。" },
          { title: "建立树关系", text: "把节点加到 parent 下，恢复 name、index、owner。" },
          { title: "解析 NodePath 属性", text: "所有节点存在后，把延迟属性从路径转回 Node。" },
          { title: "连接信号", text: "遍历 connections，构造 Callable 并 connect。" },
          { title: "返回根节点", text: "SceneState 本身仍然只是数据，返回的是新 Node 树。" }
        ]
      },
      {
        type: "heading",
        title: "_bundled：表如何存进 Resource"
      },
      {
        type: "paragraph",
        text: "`SceneState::get_bundled_scene()` 在 `packed_scene.cpp:1792`，会把内部表压成 Dictionary；`set_bundled_scene()` 在 `packed_scene.cpp:1656`，会从 Dictionary 还原内部表。PackedScene 在 `_bind_methods()` 里把 `_bundled` 作为内部存储属性绑定到这两个函数。"
      },
      {
        type: "flow",
        title: "_bundled 和文件格式的关系",
        steps: [
          { title: "SceneState 内部表", text: "`names`、`variants`、`nodes`、`connections` 等。" },
          { title: "get_bundled_scene", text: "压成 Dictionary：`names`、`nodes`、`conns`、`version`。" },
          { title: "Resource 保存器", text: "把 Dictionary 写进 `.tscn` 或 `.scn` 的资源数据。" },
          { title: "Resource 加载器", text: "读取文件后调用 build API 或 set_bundled_scene 还原。" },
          { title: "PackedScene 使用", text: "后续 pack/instantiate 都围绕这个 state 工作。" }
        ]
      },
      {
        type: "table",
        title: "索引和 flag 的设计目的",
        headers: ["标记/常量", "用途", "避免的问题"],
        rows: [
          ["`FLAG_ID_IS_PATH`", "表示某个 parent/owner/from/to 不是 node 下标，而是 node_paths 索引。", "跨实例边界时无法用本地 nodes 下标表示。"],
          ["`TYPE_INSTANTIATED`", "表示该节点来自实例/继承，当前 SceneState 只保存覆盖。", "避免把子场景内容复制成普通节点。"],
          ["`FLAG_INSTANCE_IS_PLACEHOLDER`", "表示 instance 字段保存的是 placeholder 路径。", "编辑器可延迟加载大型子场景。"],
          ["`FLAG_PATH_PROPERTY_IS_NODE`", "表示属性值是 NodePath，实例化后要转回 Node。", "文件不能保存运行时 Node 指针。"],
          ["`NO_PARENT_SAVED`", "pack 时表示父节点不在当前保存范围里。", "支持只保存必要覆盖数据。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：列出场景里的节点记录"
      },
      {
        type: "code",
        code: [
          "var packed := load(\"res://main.tscn\") as PackedScene",
          "var state := packed.get_state()",
          "",
          "for i in state.get_node_count():",
          "    print(i, state.get_node_path(i), state.get_node_name(i), state.get_node_type(i))"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个例子只读 SceneState，不创建 Node。它适合做工具脚本、检查场景结构、定位文件里到底保存了哪些节点。"
      },
      {
        type: "subheading",
        title: "案例二：检查某个节点保存了哪些属性"
      },
      {
        type: "code",
        code: [
          "func print_saved_properties(packed: PackedScene, idx: int) -> void:",
          "    var state := packed.get_state()",
          "    for p in state.get_node_property_count(idx):",
          "        print(state.get_node_property_name(idx, p), \" = \", state.get_node_property_value(idx, p))"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "如果属性没出现在这里，它就没有被场景保存。常见原因是属性没有 storage usage、值等于默认值、或属于实例/继承中不需要保存的覆盖。"
      },
      {
        type: "subheading",
        title: "案例三：检查持久信号连接"
      },
      {
        type: "code",
        code: [
          "var state := packed.get_state()",
          "for i in state.get_connection_count():",
          "    print(state.get_connection_source(i), \".\", state.get_connection_signal(i),",
          "            \" -> \", state.get_connection_target(i), \".\", state.get_connection_method(i))"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "SceneState 里的连接通常来自编辑器保存或带 `CONNECT_PERSIST` 的连接。运行时临时连接不会自动变成场景文件里的连接。"
      },
      {
        type: "subheading",
        title: "案例四：Node 引用为什么要延迟恢复"
      },
      {
        type: "code",
        code: [
          "# 保存时：某个属性看起来像 Node 引用。",
          "weapon.target = player",
          "",
          "# SceneState 里不能保存 player 指针，",
          "# 所以保存为 NodePath，并在属性名上打 FLAG_PATH_PROPERTY_IS_NODE。",
          "",
          "# 实例化时：所有节点创建完后，再把 NodePath 解析成 Node。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "如果一开始就尝试恢复 Node 引用，被引用的节点可能还没创建。SceneState 用 deferred node path 列表解决这个顺序问题。"
      },
      {
        type: "heading",
        title: "SceneState 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["PackedScene", "持有 SceneState，并作为 Resource 暴露 pack/instantiate。", "不亲自维护每个节点字段。"],
          ["SceneState", "保存可实例化场景的结构化数据表。", "不运行节点生命周期。"],
          ["ResourceFormatLoaderText", "把 .tscn 文本标签读成 SceneState。", "不负责 SceneTree 生命周期。"],
          ["Node", "实例化后的运行时对象。", "不是 SceneState 表里长期保存的指针。"],
          ["NodePath", "文件/表里跨节点引用的可序列化路径。", "不保证目标在运行时一定存在。"],
          ["owner", "决定 pack 保存范围。", "不是 SceneState 的内存所有权。"],
          ["ResourceCache", "缓存 PackedScene 资源。", "不缓存每次实例化生成的 Node 树。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：SceneState 是场景树。它只是数据表，不是运行中的 Node 树。",
          "误区二：SceneState 保存 Node 指针。它保存索引、NodePath、Variant 和资源引用。",
          "误区三：get_state() 会实例化场景。它只是返回 PackedScene 内部状态。",
          "误区四：nodes 数组等于所有可见子节点。它保存的是当前场景需要记录的数据，实例/继承内容可能只保存覆盖。",
          "误区五：属性表里没有就是引擎丢了。很多属性因为默认值、usage 或实例继承规则不会保存。",
          "误区六：connections 里有所有运行时连接。只有持久化连接会进入场景数据。",
          "误区七：NodePath 和 unique id 是同一件事。NodePath 是主要路径，id path 是恢复和兼容辅助。",
          "误区八：可以随便改 `_bundled`。它是内部序列化结构，手改很容易破坏版本、索引和引用一致性。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `packed_scene.h:38` 到 `ConnectionData`，把 SceneState 的表结构画出来。",
          "读 `add_name()`、`add_value()`、`add_node()`、`add_connection()`，理解 build API 怎么填表。",
          "读 `ResourceLoaderText::_parse_node_tag()`，把 .tscn 标签和 build API 对上。",
          "读 `get_node_path()`、`get_node_property_name()`、`get_connection_binds()`，理解 unbuild API 如何把索引还原成可读信息。",
          "读 `SceneState::instantiate()`，看读表建树、延迟 NodePath 属性和信号连接恢复。",
          "最后读 `set_bundled_scene()` / `get_bundled_scene()`，理解内部表和 `_bundled` Dictionary 的转换。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "SceneState 是 PackedScene 的结构化数据核心：它用索引表保存场景蓝图，让 Godot 能在加载、保存、检查和实例化之间可靠地转换。"
      }
    ]
  },
  {
    id: "node",
    title: "Node",
    aliases: ["Node", "节点", "owner", "queue_free", "_ready", "_enter_tree", "_process", "add_child"],
    summary: "Godot 场景树里的运行时基本单位：继承 Object，负责父子关系、owner、组、路径、生命周期通知和 process 开关。",
    article: [
      {
        type: "lead",
        text: "Node 是 Godot 场景层最核心的运行时对象。它继承 Object，所以有 ClassDB、属性、方法、信号和 ObjectID；但它额外拥有父子关系、owner、组、SceneTree 指针、生命周期通知、process/physics_process 开关、场景实例状态和删除队列语义。PackedScene.instantiate() 创建出来的就是一棵 Node 子树。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Node 想成游戏世界里的一块积木。一个 Node 可以有子 Node，子 Node 又可以有自己的子 Node，最终组成一棵树。按钮、角色、摄像机、灯光、音频播放器都可以是节点或节点子类。只有节点进入 SceneTree，它才会开始收到 `_enter_tree()`、`_ready()`、`_process()` 等回调。"
      },
      {
        type: "paragraph",
        text: "Node 不是 Resource。PackedScene 是蓝图资源，Node 是蓝图实例化后的真实对象。Resource 通常靠 RefCounted 管生命周期；Node 通常挂在 SceneTree 里，常用 `queue_free()` 延迟释放。"
      },
      {
        type: "flow",
        title: "Node 的基本生命周期",
        steps: [
          { title: "创建", text: "脚本 `Node.new()`、C++ `memnew`，或 PackedScene.instantiate()。" },
          { title: "建立父子关系", text: "调用 `add_child()`，设置 parent 和 children。" },
          { title: "进入 SceneTree", text: "父节点已在树内时，子节点会跟着 `_set_tree()`。" },
          { title: "enter_tree", text: "节点获得 tree、viewport、group 注册，并收到 ENTER_TREE。" },
          { title: "ready", text: "子节点通常先 ready，父节点后 ready。" },
          { title: "process", text: "SceneTree 根据 process 开关和优先级调度。" },
          { title: "queue_free", text: "释放请求进入删除队列，安全点再真正删除。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Node` 声明在 `scene/main/node.h:55`，继承 `Object`。真正关键的状态集中在内部 `Data` 结构：`parent`、`owner`、`children`、`tree`、`viewport`、`grouped`、`process_owner`、process flags、scene instance/inherited state、`scene_file_path`、`ready_notified`、`unique_scene_id` 等。"
      },
      {
        type: "paragraph",
        text: "这说明 Node 的职责不是“执行一切游戏逻辑”，而是维护运行时树结构和节点级状态。每帧谁被调用、何时删除、组调用怎么遍历、场景切换怎么进行，主要由 SceneTree 协调；Node 提供可被 SceneTree 调度的状态和入口。"
      },
      {
        type: "table",
        title: "Node::Data 里的关键字段",
        headers: ["字段", "含义", "典型相关源码"],
        rows: [
          ["`parent`", "当前父节点。", "`add_child()`、`remove_child()`。"],
          ["`children` / `children_cache`", "按名称保存子节点，并缓存遍历顺序。", "`_update_children_cache_impl()`。"],
          ["`owner` / `owned`", "保存场景归属，用于 PackedScene.pack/editor。", "`set_owner()`、`SceneState::_parse_node()`。"],
          ["`tree`", "当前所属 SceneTree；为空表示不在树内。", "`_set_tree()`、`is_inside_tree()`。"],
          ["`grouped`", "节点所在组和 persistent 标记。", "`add_to_group()`、`remove_from_group()`。"],
          ["`process` / `physics_process`", "是否启用每帧/物理帧处理。", "`set_process()`、`set_physics_process()`。"],
          ["`instance_state` / `inherited_state`", "场景实例/继承状态。", "PackedScene.instantiate 编辑器路径。"],
          ["`scene_file_path`", "实例化来源场景路径。", "`PackedScene::instantiate()`、`set_scene_file_path()`。"],
          ["`unique_scene_id`", "场景内唯一 id，辅助 SceneState 路径恢复。", "SceneState pack/instantiate。"]
        ]
      },
      {
        type: "heading",
        title: "父子关系：add_child 不等于保存"
      },
      {
        type: "paragraph",
        text: "`Node::add_child()` 在 `scene/main/node.cpp:1711`。它会检查不能把节点加到自己下面、不能重复 parent、不能形成环、父节点是否正在 blocked 遍历；然后验证子节点名称，最后调用 `_add_child_nocheck()` 真正设置 `p_child->data.parent = this`、写入 children，并在父节点已在树内时让子节点 `_set_tree(data.tree)`。"
      },
      {
        type: "paragraph",
        text: "`remove_child()` 在 `node.cpp:1747`。它会先让子节点 `_set_tree(nullptr)`，触发退出树，再从 children 里移除并清空 parent。注意：从父节点移除并不等于释放对象；它只是变成不在树内的 Node，仍需要被持有者释放或重新加入。"
      },
      {
        type: "flow",
        title: "`parent.add_child(child)` 的内部路径",
        steps: [
          { title: "检查线程和状态", text: "树内节点只能主线程改树；blocked 时提示用 deferred。" },
          { title: "验证 parent", text: "child 不能已有 parent，也不能形成祖先环。" },
          { title: "确认名称", text: "`_validate_child_name()` 保证同一 parent 下名称可用。" },
          { title: "写入 children", text: "`_add_child_nocheck()` 设置 parent、children、index/cache。" },
          { title: "进入树", text: "如果 parent 已在 SceneTree 内，child 也 `_set_tree()`。" },
          { title: "发通知", text: "父子顺序变化和 parented/child_order_changed 信号触发。" }
        ]
      },
      {
        type: "table",
        title: "parent 和 owner 的差别",
        headers: ["关系", "决定什么", "源码入口", "常见问题"],
        rows: [
          ["`parent`", "运行时树结构、路径、enter/exit 传播、子节点遍历。", "`add_child()`、`remove_child()`。", "remove_child 后对象没释放。"],
          ["`owner`", "保存场景时节点属于哪个场景根。", "`set_owner()` 在 `node.cpp:2271`。", "动态节点不设置 owner，pack 不进 .tscn。"],
          ["`scene_file_path`", "节点实例来自哪个 PackedScene 文件。", "`PackedScene::instantiate()` 设置。", "它不是 owner，也不代表节点当前在树里。"],
          ["`unique_scene_id`", "SceneState 在继承/恢复路径时使用。", "pack/instantiate 维护。", "不是 ObjectID，也不是全局 id。"]
        ]
      },
      {
        type: "heading",
        title: "进入树、ready 和退出树"
      },
      {
        type: "paragraph",
        text: "`Node::_set_tree()` 在 `node.cpp:3354`。当 tree 从空变成非空，它调用 `_propagate_enter_tree()`，然后在根或父节点已经 ready 的情况下调用 `_propagate_ready()`；当 tree 从非空变空，它调用 `_propagate_exit_tree()`。"
      },
      {
        type: "paragraph",
        text: "`_propagate_enter_tree()` 在 `node.cpp:341`：设置 tree、depth、viewport，把已有 group 注册进 SceneTree，发 `NOTIFICATION_ENTER_TREE`、调用脚本 `_enter_tree`、发 `tree_entered`，然后递归子节点。`_propagate_ready()` 在 `node.cpp:323`：先递归子节点，再给当前节点发 `NOTIFICATION_POST_ENTER_TREE` 和首次 `NOTIFICATION_READY`。"
      },
      {
        type: "flow",
        title: "进入 SceneTree 的通知顺序",
        steps: [
          { title: "_set_tree(tree)", text: "节点获得 tree 指针。" },
          { title: "_propagate_enter_tree", text: "当前节点先 enter_tree，再递归子节点 enter_tree。" },
          { title: "注册组和 viewport", text: "grouped 进入 SceneTree 的 group map。" },
          { title: "_propagate_ready", text: "ready 阶段对子节点递归在前。" },
          { title: "子节点 ready", text: "子节点先收到 READY。" },
          { title: "父节点 ready", text: "父节点后收到 READY，所以父节点通常能访问 ready 后的子节点。" }
        ]
      },
      {
        type: "table",
        title: "几个生命周期入口",
        headers: ["入口/通知", "发生时机", "用途"],
        rows: [
          ["`NOTIFICATION_ENTER_TREE` / `_enter_tree()`", "节点刚进入 SceneTree 时。", "获取 tree、注册运行时关系、早期初始化。"],
          ["`NOTIFICATION_READY` / `_ready()`", "子树进入后，首次 ready。", "访问子节点、初始化场景内依赖。"],
          ["`_process(delta)`", "普通帧，由 SceneTree process 调度。", "帧率相关逻辑。"],
          ["`_physics_process(delta)`", "固定物理帧，由 SceneTree physics_process 调度。", "物理相关逻辑。"],
          ["`NOTIFICATION_EXIT_TREE` / `_exit_tree()`", "节点离开 SceneTree。", "断开运行时关系、释放树相关状态。"],
          ["`queue_free()`", "请求延迟释放。", "避免遍历时立即删除节点。"]
        ]
      },
      {
        type: "heading",
        title: "组和 process：Node 提供状态，SceneTree 执行调度"
      },
      {
        type: "paragraph",
        text: "`add_to_group()` 在 `node.cpp:2458`。如果节点已经在 tree 内，它会立即调用 `data.tree->add_to_group()`；如果不在树内，只先记在 `data.grouped`，等 `_propagate_enter_tree()` 时再注册。`persistent` 为 true 的 group 才会被 SceneState 保存。"
      },
      {
        type: "paragraph",
        text: "`set_process()`、`set_physics_process()` 等方法只是在 Node 上更新处理开关，并让 SceneTree 的 process group 知道节点需要被调度。真正每帧遍历、排序、暂停处理和线程组处理都在 SceneTree。"
      },
      {
        type: "table",
        title: "Node 和 SceneTree 的职责边界",
        headers: ["能力", "Node 负责", "SceneTree 负责"],
        rows: [
          ["父子树", "保存 parent/children/name/index。", "根节点、当前场景、全树遍历入口。"],
          ["生命周期", "传播 enter/ready/exit 通知。", "在主循环阶段调用 process/physics。"],
          ["组", "保存 grouped 和 persistent 标记。", "维护 group map、执行 group call/set/notify。"],
          ["process", "保存开关、优先级、线程组。", "按帧调度节点处理。"],
          ["删除", "`queue_free()` 发起请求。", "`queue_delete()` 和 `_flush_delete_queue()` 安全删除。"],
          ["保存场景", "保存 owner、scene state 相关信息。", "不直接保存 .tscn；PackedScene/SceneState 执行 pack。"]
        ]
      },
      {
        type: "heading",
        title: "queue_free：为什么不是立即删除"
      },
      {
        type: "paragraph",
        text: "`Node::queue_free()` 在 `node.cpp:3461`。它不会马上 `memdelete(this)`，而是把对象交给当前 SceneTree 的 `queue_delete()`。真正删除发生在 SceneTree 的删除队列 flush 阶段。这能避免在遍历子节点、发信号、process 或物理同步时直接释放当前对象造成不一致。"
      },
      {
        type: "paragraph",
        text: "如果节点不在 tree 内，`queue_free()` 会尝试使用 `SceneTree::get_singleton()`。如果没有可用 SceneTree，就会报错。普通 Object 没有 queue_free 语义；这是 Node/SceneTree 组合提供的延迟删除模型。"
      },
      {
        type: "flow",
        title: "`node.queue_free()` 的心智路径",
        steps: [
          { title: "脚本请求释放", text: "调用 `node.queue_free()`。" },
          { title: "找到 SceneTree", text: "优先使用 node 自己的 `data.tree`。" },
          { title: "排入删除队列", text: "SceneTree 记录要删除的 ObjectID/对象。" },
          { title: "当前流程继续", text: "信号、遍历或 process 不被立即打断。" },
          { title: "安全点 flush", text: "SceneTree 在固定阶段统一释放 queued delete。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：实例化场景后进入树"
      },
      {
        type: "code",
        code: [
          "var packed := load(\"res://enemy.tscn\") as PackedScene",
          "var enemy := packed.instantiate()",
          "",
          "# 这里 enemy 只是 Node 子树，还没有进入当前 SceneTree。",
          "add_child(enemy)",
          "# add_child 后才会触发 enter_tree/ready。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这是 PackedScene、Node、SceneTree 三者最常见的连接点：资源蓝图创建 Node，add_child 把 Node 交给树。"
      },
      {
        type: "subheading",
        title: "案例二：动态节点想保存必须设置 owner"
      },
      {
        type: "code",
        code: [
          "@tool",
          "func add_marker() -> void:",
          "    var marker := Node2D.new()",
          "    marker.name = \"Marker\"",
          "    add_child(marker)",
          "    marker.owner = owner if owner != null else self"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "parent 决定它在树里能不能运行，owner 决定 PackedScene.pack 或编辑器保存时它属于哪个场景。"
      },
      {
        type: "subheading",
        title: "案例三：遍历时修改树要延迟"
      },
      {
        type: "code",
        code: [
          "func _ready() -> void:",
          "    for child in get_children():",
          "        child.queue_free()",
          "",
          "    # 如果要在敏感回调里 add_child/remove_child，",
          "    # 常用 call_deferred(\"add_child\", node)。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Node 源码里有 `data.blocked` 防护。父节点正忙着遍历或设置子节点时直接改树，会提示使用 deferred。"
      },
      {
        type: "subheading",
        title: "案例四：组只在进树后进入 SceneTree 的组表"
      },
      {
        type: "code",
        code: [
          "func _init() -> void:",
          "    add_to_group(\"enemies\") # 先记录在 Node.data.grouped。",
          "",
          "func _enter_tree() -> void:",
          "    # 进入树后，SceneTree 才能通过 group 找到它。",
          "    pass"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这解释了为什么同一个 group API 涉及 Node 和 SceneTree 两层：Node 记录自己属于哪些组，SceneTree 负责全局组索引和组调用。"
      },
      {
        type: "heading",
        title: "Node 和相邻概念的边界"
      },
      {
        type: "table",
        title: "不要把这些职责混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "反射、属性、方法、信号、ObjectID。", "不提供父子树和生命周期调度。"],
          ["Node", "父子关系、owner、组、路径、生命周期通知、process 开关。", "不直接执行全局每帧调度。"],
          ["SceneTree", "根树、当前场景、process/physics、组调用、删除队列。", "不保存每个节点业务属性。"],
          ["PackedScene", "保存/实例化 Node 树蓝图。", "不是运行中的 Node。"],
          ["SceneState", "保存 Node 树的结构化数据。", "不接收 enter_tree/ready。"],
          ["Resource", "共享数据资源，通常 RefCounted。", "不是场景树节点。"],
          ["MessageQueue", "延迟 call/set/notification。", "不管理 Node 的 parent/owner。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Node 是 Resource。Node 继承 Object；Resource 通常继承 RefCounted。",
          "误区二：创建 Node 就会 ready。只有进入 SceneTree 后才会 enter_tree/ready。",
          "误区三：add_child 会保存节点。保存需要 owner 和 PackedScene/SceneState 的 pack 规则。",
          "误区四：remove_child 会释放对象。它只断开父子关系，节点仍然存在。",
          "误区五：queue_free 立即删除。它排入 SceneTree 删除队列，安全点才释放。",
          "误区六：父节点 ready 一定早于子节点。通常是子节点先 ready，父节点后 ready。",
          "误区七：group 是全局常驻表。Node 不在树内时只记录本地 grouped，进入树后才注册到 SceneTree。",
          "误区八：可以从任何线程改树。树内 add/remove child 要主线程，源码会报错并建议 deferred。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/main/node.h:55` 和内部 `Data` 结构，知道 Node 保存哪些状态。",
          "读 `add_child()`、`remove_child()` 和 `_add_child_nocheck()`，理解父子关系如何改变。",
          "读 `_set_tree()`、`_propagate_enter_tree()`、`_propagate_ready()`、`_propagate_exit_tree()`，建立生命周期顺序。",
          "读 `set_owner()`，把 parent 和 owner 区分清楚。",
          "读 `add_to_group()` 和 SceneTree 的 group 调用，理解组是 Node/SceneTree 协作。",
          "最后读 `queue_free()` 和 SceneTree 删除队列，理解延迟删除。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Node 是 Godot 场景树里的运行时积木：它维护自己在树中的关系和状态，SceneTree 按这些状态驱动生命周期、每帧处理、组调用和延迟删除。"
      }
    ]
  },
  {
    id: "scenetree",
    title: "SceneTree",
    aliases: ["SceneTree", "MainLoop", "current_scene", "process_frame", "physics_frame", "call_group", "GROUP_CALL_DEFERRED", "queue_delete", "SceneTreeTimer", "create_tween"],
    summary: "Godot 的默认 MainLoop：拥有 root Window 和 current_scene，负责每帧 process/physics 调度、组调用、场景切换、timer/tween、MessageQueue 安全点和删除队列。",
    article: [
      {
        type: "lead",
        text: "SceneTree 是 Godot 项目运行时的总调度器。它继承 MainLoop，不是普通游戏节点；它持有 root Window、current_scene、组索引、process 分组、timer/tween、多玩家轮询、场景切换状态和删除队列。Node 负责保存自己在树里的状态，SceneTree 负责在主循环里按固定节奏驱动这些 Node。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 Node 是舞台上的演员，SceneTree 就像舞台经理和排班表。它知道舞台根在哪里、当前主场景是谁、哪些演员属于 enemies 组、谁每帧要更新、谁物理帧要更新、谁说“稍后删除”。每一帧到来时，SceneTree 按顺序通知大家行动。"
      },
      {
        type: "paragraph",
        text: "所以 `get_tree()` 拿到的不是某个节点，而是这棵运行中场景树的管理者。你用它切换场景、创建一次性 timer、调用某个组、暂停游戏、排队删除对象。真正的节点仍然挂在 root Window 下面，current_scene 只是其中一个被 SceneTree 特别记住的主场景引用。"
      },
      {
        type: "flow",
        title: "一帧里 SceneTree 大概做什么",
        steps: [
          { title: "准备", text: "处理固定时间步插值和 transform 通知。" },
          { title: "物理帧", text: "发 `physics_frame`，调度 physics process 节点。" },
          { title: "安全点", text: "flush MessageQueue、timer、tween 和待删除对象。" },
          { title: "普通帧", text: "发 `process_frame`，调度普通 process 节点。" },
          { title: "场景切换", text: "如果有 pending_new_scene，安全地移除旧场景并挂上新场景。" },
          { title: "渲染前同步", text: "SceneTree 不直接画图，节点状态随后同步到 Server 和渲染阶段。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`SceneTree` 声明在 `scene/main/scene_tree.h:89`，继承 `MainLoop`。这意味着平台层和 `Main::iteration()` 不直接逐个调用 Node；它们调用 MainLoop 的阶段函数，而默认 MainLoop 是 SceneTree。SceneTree 再在 `physics_process()` 和 `process()` 内部调用节点、处理队列、推进 timer/tween。"
      },
      {
        type: "paragraph",
        text: "源码里最能说明职责的是字段：`root` 在 `scene_tree.h:133`，`group_map` 在 `:148`，`delete_queue` 在 `:188`，`current_scene`、`prev_scene_id`、`pending_new_scene_id` 在 `:203` 附近，`timers` 和 `tweens` 在 `:218`，`multiplayer` 在 `:223`。这些字段覆盖了运行中场景树的调度面，而不是某个节点自己的业务数据。"
      },
      {
        type: "table",
        title: "SceneTree 的核心职责",
        headers: ["职责", "关键字段/入口", "说明"],
        rows: [
          ["根窗口", "`Window *root`", "真正的树根是 root Window；游戏场景是它的子树。"],
          ["当前场景", "`current_scene`、`pending_new_scene_id`", "记录主场景和待切换场景，不代表整棵树只有它一个子节点。"],
          ["每帧调度", "`physics_process()`、`process()`、`_process()`", "发帧信号，按 process group 调用节点通知。"],
          ["组索引", "`group_map`、`SceneTreeGroup`", "让 `call_group()`、`notify_group()`、`set_group()` 能找到组内节点。"],
          ["延迟与安全点", "`MessageQueue::flush()`、`_flush_delete_queue()`", "延迟调用和延迟删除在固定位置处理。"],
          ["运行时工具", "`create_timer()`、`create_tween()`", "创建 SceneTreeTimer 和 Tween，并由 SceneTree 推进。"],
          ["多玩家轮询", "`multiplayer`、`multiplayer_poll`", "普通 process 阶段可自动 poll MultiplayerAPI。"]
        ]
      },
      {
        type: "heading",
        title: "SceneTree 是 MainLoop"
      },
      {
        type: "paragraph",
        text: "`iteration_prepare()` 在 `scene/main/scene_tree.cpp:629`，主要服务固定时间步插值和 transform 刷新。`physics_process()` 在 `:639`，`process()` 在 `:688`。这三个函数共同构成 SceneTree 在主循环中的基本节奏。"
      },
      {
        type: "table",
        title: "主要 MainLoop 阶段",
        headers: ["阶段", "源码入口", "做什么"],
        rows: [
          ["prepare", "`SceneTree::iteration_prepare()`", "刷新 transform 通知，更新 fixed timestep interpolation 需要的 tick 状态。"],
          ["physics", "`SceneTree::physics_process()`", "递增 frame，发 `physics_frame`，处理 picking viewports、physics process group、MessageQueue、timer/tween、删除队列。"],
          ["iteration_end", "`SceneTree::iteration_end()`", "物理插值开启时，把 transform 刷到 RenderingServer，并维护客户端插值状态。"],
          ["idle/process", "`SceneTree::process()`", "poll multiplayer，发 `process_frame`，flush MessageQueue，处理普通 process group，flush 场景切换、timer/tween、删除队列。"]
        ]
      },
      {
        type: "flow",
        title: "`SceneTree::process()` 的关键顺序",
        steps: [
          { title: "MainLoop::process", text: "先执行基类 process，可能请求退出。" },
          { title: "multiplayer poll", text: "`multiplayer_poll` 打开时轮询默认和自定义 MultiplayerAPI。" },
          { title: "process_frame", text: "发出全局 `process_frame` 信号。" },
          { title: "MessageQueue flush", text: "先清一批 deferred call/set/notification。" },
          { title: "_process(false)", text: "按普通 process group 调用 Node 通知。" },
          { title: "再次 flush", text: "处理节点 process 期间产生的 deferred 调用。" },
          { title: "flush scene change", text: "如果有 pending_new_scene，挂上新场景并发 `scene_changed`。" },
          { title: "timer/tween/delete", text: "推进 timer/tween，最后 `_flush_delete_queue()`。" }
        ]
      },
      {
        type: "heading",
        title: "process group：为什么不是简单遍历整棵树"
      },
      {
        type: "paragraph",
        text: "SceneTree 内部有 `ProcessGroup`，字段在 `scene_tree.h:100`：一个 group 有自己的 `CallQueue`、普通 process 节点列表、physics process 节点列表、排序脏标记和 owner。`SceneTree::_process_group()` 在 `scene_tree.cpp:1176`，会先 flush 这个组的 call_queue，再排序节点副本，然后检查 `can_process()`、`is_inside_tree()`，最后发 `NOTIFICATION_PROCESS` 或 `NOTIFICATION_PHYSICS_PROCESS`。"
      },
      {
        type: "paragraph",
        text: "这比“每帧递归遍历整棵树”更可控：节点可以设置 process priority、暂停模式、process thread group；SceneTree 可以在节点增删时仍安全地继续当前遍历。`_process()` 在 `scene_tree.cpp:1243` 还会按 process group order 和是否子线程分批处理。"
      },
      {
        type: "table",
        title: "Node 与 SceneTree 在 process 上的分工",
        headers: ["问题", "Node 保存", "SceneTree 执行"],
        rows: [
          ["是否处理", "`set_process()`、`set_physics_process()` 的开关。", "在 `_process_group()` 中检查后发通知。"],
          ["处理顺序", "process priority、physics priority。", "排序 dirty list，然后遍历副本。"],
          ["暂停规则", "process mode 和节点状态。", "`can_process()` 决定当前帧是否跳过。"],
          ["线程组", "process thread group 和 order。", "`_process_groups_thread()` 交给 WorkerThreadPool。"],
          ["延迟调用", "节点可发起 deferred call。", "process group 的 CallQueue 和全局 MessageQueue 在安全点 flush。"]
        ]
      },
      {
        type: "heading",
        title: "组调用：group_map 不是装饰功能"
      },
      {
        type: "paragraph",
        text: "Node 的 `add_to_group()` 只是声明“我属于这个组”；节点进入树后，SceneTree 才把它加入 `group_map`。`SceneTree::call_group_flagsp()` 在 `scene_tree.cpp:361`：它找到组，更新顺序，复制节点数组，然后按 flags 逐个调用，或把调用推入 MessageQueue。"
      },
      {
        type: "paragraph",
        text: "复制节点数组和 `nodes_removed_on_group_call` 是安全设计：组调用过程中节点可能被删除或移出组，SceneTree 不能让当前遍历直接失效。`GROUP_CALL_UNIQUE + GROUP_CALL_DEFERRED` 还会进入 `unique_group_calls`，等 `_flush_ugc()` 在 `scene_tree.cpp:320` 统一去重执行。"
      },
      {
        type: "table",
        title: "GroupCallFlags 怎么影响调用",
        headers: ["flag", "效果", "适合场景"],
        rows: [
          ["`GROUP_CALL_DEFAULT`", "按树顺序立即调用。", "简单通知一批节点。"],
          ["`GROUP_CALL_REVERSE`", "反向遍历组内节点。", "希望子/后加入节点优先处理时。"],
          ["`GROUP_CALL_DEFERRED`", "不立刻调用，推入 MessageQueue。", "避免当前遍历或信号回调中马上改树。"],
          ["`GROUP_CALL_UNIQUE`", "和 deferred 搭配时，同一组同一方法只保留一次。", "频繁请求刷新 UI、AI 目标或缓存时去抖。"]
        ]
      },
      {
        type: "flow",
        title: "`call_group_flags()` 的内部路径",
        steps: [
          { title: "查 group_map", text: "没有这个组或组为空，直接返回。" },
          { title: "处理 unique", text: "`UNIQUE + DEFERRED` 时写入 `unique_group_calls` 去重。" },
          { title: "更新顺序", text: "组被改过时按 Node::Comparator 排序。" },
          { title: "复制节点数组", text: "遍历副本，避免组内增删破坏当前循环。" },
          { title: "按 flags 遍历", text: "正序或反序，跳过已删除节点。" },
          { title: "立即或延迟", text: "立即 `node->callp()`，或 `MessageQueue::push_callp()`。" }
        ]
      },
      {
        type: "heading",
        title: "场景切换：不是 load 后立刻进树"
      },
      {
        type: "paragraph",
        text: "`change_scene_to_file()` 在 `scene_tree.cpp:1702`：先用 `ResourceLoader::load()` 得到 PackedScene。`change_scene_to_packed()` 在 `:1712`：调用 `PackedScene.instantiate()` 得到 Node。`change_scene_to_node()` 在 `:1721`：要求新节点还不能在树内，然后把旧 current_scene 从 root 移除，把新场景的 ObjectID 存进 `pending_new_scene_id`。"
      },
      {
        type: "paragraph",
        text: "真正把新场景挂到 root 下发生在 `_flush_scene_change()`，源码在 `scene_tree.cpp:1672`。它先释放上一场景，再用 ObjectDB 从 `pending_new_scene_id` 找回新节点，设置 `current_scene`，调用 `root->add_child(pending_new_scene)`，最后发 `scene_changed`。这说明场景切换是“请求 + 安全点提交”的模型。"
      },
      {
        type: "flow",
        title: "`change_scene_to_file(\"res://main.tscn\")` 路径",
        steps: [
          { title: "加载资源", text: "ResourceLoader 读取路径，返回 PackedScene。" },
          { title: "实例化节点", text: "PackedScene.instantiate() 创建未进树的 Node 子树。" },
          { title: "移除旧场景", text: "current_scene 从 root 下 remove_child，副作用先发生或排队。" },
          { title: "记录待切换", text: "保存新节点 ObjectID 到 `pending_new_scene_id`。" },
          { title: "process 安全点", text: "`SceneTree::process()` 检查 pending 并 `_flush_scene_change()`。" },
          { title: "挂入 root", text: "root.add_child(new_scene)，发 `scene_changed`。" }
        ]
      },
      {
        type: "table",
        title: "场景切换 API 边界",
        headers: ["API", "输入", "内部动作", "容易误解的点"],
        rows: [
          ["`change_scene_to_file(path)`", "资源路径。", "ResourceLoader.load 后转给 packed 版本。", "加载出来不是 Node，而是 PackedScene。"],
          ["`change_scene_to_packed(packed)`", "PackedScene 资源。", "instantiate 后转给 node 版本。", "实例化后仍未进入 SceneTree。"],
          ["`change_scene_to_node(node)`", "未进树 Node。", "移除旧 current_scene，记录 pending_new_scene_id。", "不是马上 add_child，新场景稍后 flush。"],
          ["`set_current_scene(node)`", "root 的直接子节点或 null。", "只设置 current_scene 指针。", "不等于自动加载/保存/切换场景。"],
          ["`unload_current_scene()`", "无。", "直接 memdelete 当前场景并清空指针。", "不是保存，也不是 Resource 卸载。"]
        ]
      },
      {
        type: "heading",
        title: "删除队列：queue_free 的落点"
      },
      {
        type: "paragraph",
        text: "`SceneTree::queue_delete()` 在 `scene_tree.cpp:1637`。它把对象标记 `_is_queued_for_deletion = true`，再把对象的 `ObjectID` 放入 `delete_queue`。真正删除在 `_flush_delete_queue()` 的 `scene_tree.cpp:1625`：循环取出 ObjectID，用 ObjectDB 查当前对象仍是否存在，存在才 `memdelete(obj)`。"
      },
      {
        type: "paragraph",
        text: "这个设计和 MessageQueue 相邻但不是同一件事。MessageQueue 管延迟方法调用、属性设置、通知；delete_queue 管对象释放。它们常在 process/physics 的安全点附近 flush，是因为两者都要避免“遍历到一半立刻改变世界”。"
      },
      {
        type: "flow",
        title: "`node.queue_free()` 到真正释放",
        steps: [
          { title: "Node.queue_free", text: "节点请求稍后释放。" },
          { title: "SceneTree.queue_delete", text: "设置 queued 标记，保存 ObjectID。" },
          { title: "当前逻辑继续", text: "信号回调、组调用或 process 不被立即打断。" },
          { title: "flush delete queue", text: "SceneTree 在帧阶段靠后统一处理。" },
          { title: "ObjectDB 回查", text: "对象还存在才 memdelete，避免重复释放。" }
        ]
      },
      {
        type: "heading",
        title: "timer、tween 和帧信号"
      },
      {
        type: "paragraph",
        text: "`create_timer()` 在 `scene_tree.cpp:1767` 创建 `SceneTreeTimer`，设置 process_always、process_in_physics、ignore_time_scale 等参数，然后放进 `timers`。`create_tween()` 在 `:1779` 创建 Tween 并放进 `tweens`。SceneTree 在 physics 和 process 阶段分别调用 `process_timers()`、`process_tweens()`。"
      },
      {
        type: "paragraph",
        text: "`process_frame` 和 `physics_frame` 是 SceneTree 的全局信号，绑定在 `_bind_methods()`，源码在 `scene_tree.cpp:1994` 附近。它们表示 SceneTree 进入了对应帧阶段；它们不是某个节点的 `_process(delta)` 或 `_physics_process(delta)` 本身。"
      },
      {
        type: "table",
        title: "常用 SceneTree 能力速查",
        headers: ["脚本入口", "背后机制", "何时使用"],
        rows: [
          ["`get_tree().create_timer(1.0)`", "创建 SceneTreeTimer，之后由 SceneTree 推进。", "等待一次性延迟，不想专门放 Timer 节点。"],
          ["`get_tree().create_tween()`", "创建由 SceneTree 管理的 Tween。", "运行时做临时插值动画。"],
          ["`get_tree().call_group(...)`", "从 group_map 找节点并调用方法。", "向一组节点广播简单命令。"],
          ["`get_tree().queue_delete(obj)`", "把 ObjectID 放入删除队列。", "底层延迟删除入口；通常脚本用 `node.queue_free()`。"],
          ["`get_tree().paused = true`", "影响节点 can_process 和 Server active 状态。", "暂停游戏逻辑，但仍允许特定节点按 process mode 运行。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：切换主场景"
      },
      {
        type: "code",
        code: [
          "func go_to_title() -> void:",
          "    var err := get_tree().change_scene_to_file(\"res://ui/title_screen.tscn\")",
          "    if err != OK:",
          "        push_error(\"Failed to load title screen\")"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这条调用内部先走 ResourceLoader，再实例化 PackedScene，最后等 SceneTree 在 process 安全点提交场景切换。"
      },
      {
        type: "subheading",
        title: "案例二：延迟组调用，避免当前遍历里改树"
      },
      {
        type: "code",
        code: [
          "func retarget_enemies(player: Node) -> void:",
          "    get_tree().call_group_flags(",
          "        SceneTree.GROUP_CALL_DEFERRED | SceneTree.GROUP_CALL_UNIQUE,",
          "        \"enemies\",",
          "        \"refresh_target\",",
          "        player",
          "    )"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`DEFERRED` 把调用推入 MessageQueue，`UNIQUE` 避免同一组同一方法在同一轮里重复排队。"
      },
      {
        type: "subheading",
        title: "案例三：一次性等待"
      },
      {
        type: "code",
        code: [
          "func flash_after_hit() -> void:",
          "    modulate = Color.RED",
          "    await get_tree().create_timer(0.12).timeout",
          "    modulate = Color.WHITE"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这里没有添加 Timer 节点；SceneTreeTimer 由 SceneTree 的 timers 列表推进，时间到后发 timeout。"
      },
      {
        type: "subheading",
        title: "案例四：安全删除当前节点"
      },
      {
        type: "code",
        code: [
          "func die() -> void:",
          "    hide()",
          "    set_process(false)",
          "    queue_free()"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`queue_free()` 不会立刻释放当前对象；它进入 SceneTree 的 delete_queue，等帧阶段靠后位置统一处理。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "SceneTree 不等于这些东西",
        headers: ["概念", "负责什么", "SceneTree 与它的关系"],
        rows: [
          ["Node", "父子关系、owner、组记录、生命周期入口。", "SceneTree 调度 Node，但不是 Node 自身。"],
          ["root Window", "场景树根窗口和视口能力。", "SceneTree 持有 root；root 才是实际树根节点。"],
          ["current_scene", "当前主场景指针。", "只是 SceneTree 记住的一个 Node，不代表全部树内容。"],
          ["PackedScene", "场景蓝图资源。", "SceneTree 切换场景时会加载/实例化它，但不解析 .tscn 文本。"],
          ["MessageQueue", "延迟 call/set/notification。", "SceneTree 决定多个 flush 安全点。"],
          ["ObjectDB", "ObjectID 到 Object 的弱索引。", "delete_queue 和 pending scene 都用 ObjectID 回查对象。"],
          ["RenderingServer", "渲染后端资源和提交。", "SceneTree 不直接画图，最终由 Main/Server 阶段同步和绘制。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：SceneTree 是根节点。实际 root 是 Window，SceneTree 持有它并调度它下面的树。",
          "误区二：current_scene 就是整棵树。它只是主场景引用，autoload、UI、调试节点等也可能挂在 root 下。",
          "误区三：change_scene_to_file 立刻把新场景加进树。它先记录 pending，新场景在 process 安全点 flush。",
          "误区四：SceneTree 负责解析 .tscn。解析和加载属于 ResourceLoader/PackedScene/SceneState。",
          "误区五：call_group 一定安全改树。立即组调用仍在当前遍历中执行；改树敏感操作通常用 DEFERRED。",
          "误区六：MessageQueue 和 delete_queue 是同一个队列。前者处理延迟调用，后者处理对象释放。",
          "误区七：process_frame 等同于每个节点的 _process。它是 SceneTree 的帧信号，节点处理还要经过 process group。",
          "误区八：queue_free 会马上释放内存。它标记并排队，稍后用 ObjectID 回查后再 memdelete。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/main/scene_tree.h:89`，把 SceneTree 当作 MainLoop，而不是 Node。",
          "读 `root`、`group_map`、`delete_queue`、`current_scene`、`timers`、`tweens` 等字段，建立职责地图。",
          "读 `physics_process()` 和 `process()`，把 MessageQueue、timer/tween、场景切换、删除队列的位置画出来。",
          "读 `_process_group()` 和 `_process()`，理解 process group、排序、暂停和线程组。",
          "读 `call_group_flagsp()`、`notify_group_flags()`、`set_group_flags()`，理解组调用为什么要复制节点列表。",
          "读 `change_scene_to_file()` 到 `_flush_scene_change()`，把 ResourceLoader、PackedScene、Node、SceneTree 串起来。",
          "最后读 `queue_delete()` 和 `_flush_delete_queue()`，确认 queue_free 的真实落点。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "SceneTree 是 Godot 默认运行主循环的场景调度器：它不保存游戏数据本身，而是在每帧安全地驱动 Node、组调用、场景切换、timer/tween、延迟调用和延迟删除。"
      }
    ]
  },
  {
    id: "owner",
    title: "owner",
    aliases: ["owner", "Node.owner", "set_owner", "get_owner", "owned", "unique_name_in_owner", "PackedScene.pack", "SceneState::_parse_node", "editable instance", "保存场景"],
    summary: "Node 的保存归属关系：parent 决定运行时树结构，owner 决定 PackedScene.pack 时这个节点是否属于当前场景。",
    article: [
      {
        type: "lead",
        text: "`owner` 是 Godot 场景保存系统里最容易混淆的 Node 概念。`parent` 说明节点运行时挂在哪个父节点下面；`owner` 说明保存场景时，这个节点属于哪个场景根。一个节点可以在树里正常运行，但如果没有正确 owner，PackedScene.pack 或编辑器保存时可能完全不保存它。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 parent 想成“现在住在哪个房间”，把 owner 想成“户口属于哪本房产证”。你把一个椅子搬进房间，它马上能被看见、能使用，这就是 parent 对了；但如果没有把它登记到这套房子的清单里，保存房子蓝图时它不会被写进去，这就是 owner 没设置。"
      },
      {
        type: "paragraph",
        text: "所以编辑器工具脚本里动态创建节点时，常常要同时做两件事：`add_child(child)` 让节点进入树，`child.owner = scene_root` 让它属于要保存的场景。只做 add_child，节点能运行；不设置 owner，重新打开场景可能就没了。"
      },
      {
        type: "flow",
        title: "动态节点为什么会“能运行但不保存”",
        steps: [
          { title: "创建节点", text: "`Node2D.new()` 或 C++ `memnew(Node2D)`。" },
          { title: "add_child", text: "设置 parent，节点进入运行时树。" },
          { title: "没有 owner", text: "节点不属于当前保存场景。" },
          { title: "保存场景", text: "PackedScene.pack 解析节点树。" },
          { title: "过滤掉", text: "SceneState::_parse_node 跳过不属于 owner 的节点。" },
          { title: "重新打开", text: "动态节点没有出现在 .tscn 里。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`owner` 字段在 `scene/main/node.h:203` 附近的 `Node::Data` 里：`parent`、`owner`、`children`、`owned_unique_nodes`、`owned` 是一组相互配合的数据。`parent` 和 `children` 管运行时树；`owner`、`owned`、`owned_unique_nodes` 管保存归属和 `%Name` 这种 owner 内唯一名称。"
      },
      {
        type: "paragraph",
        text: "`Node::set_owner()` 在 `scene/main/node.cpp:2271`。它先清掉旧 owner，再拒绝 `p_owner == this`，如果传入 null 就只表示取消 owner；非空 owner 必须是当前节点的祖先，否则报错 “Owner must be an ancestor in the tree.”。通过检查后，它调用 `_set_owner_nocheck()`，把当前节点放进 owner 的 `data.owned` 列表。"
      },
      {
        type: "table",
        title: "parent、owner、instance、current_scene 的差别",
        headers: ["概念", "保存位置/入口", "决定什么", "不决定什么"],
        rows: [
          ["`parent`", "`Node::data.parent`、`add_child()`", "运行时树结构、NodePath、enter/exit 传播、子节点遍历。", "不代表会被保存进 .tscn。"],
          ["`owner`", "`Node::data.owner`、`set_owner()`", "PackedScene.pack 的保存归属、owner 内唯一名称作用域。", "不表示内存所有权，也不等于父节点。"],
          ["`owned`", "`owner->data.owned`", "某个 owner 直接拥有的节点列表。", "不是 C++ 智能指针，不负责释放节点。"],
          ["`instance`", "PackedScene/SceneState 的实例信息。", "子场景、继承场景、editable children 的保存边界。", "不是运行时 parent，也不是 current_scene。"],
          ["`current_scene`", "`SceneTree::current_scene`", "游戏主场景和场景切换引用。", "不控制 pack 的序列化归属。"]
        ]
      },
      {
        type: "heading",
        title: "set_owner 的约束"
      },
      {
        type: "paragraph",
        text: "owner 必须是祖先，不要求一定是直接 parent。比如 `Root/Level/Enemy/Weapon` 里，Weapon 的 parent 可以是 Enemy，owner 可以是 Root，也可以是 Level，只要 owner 是 Weapon 的祖先。这个限制让保存系统能从场景根出发，用路径稳定地描述“谁属于这个场景”。"
      },
      {
        type: "paragraph",
        text: "`_set_owner_nocheck()` 在 `node.cpp:2211`，只做内部写入：设置 `data.owner`，把当前节点 push 到 owner 的 `data.owned`，保存列表元素句柄 `data.OW`，然后通知 owner 变化。它叫 nocheck，是因为加载 PackedScene 恢复数据时已经知道 SceneState 表是合法结构，需要绕过公开 API 的祖先检查。"
      },
      {
        type: "flow",
        title: "`child.set_owner(scene_root)` 的路径",
        steps: [
          { title: "清旧 owner", text: "如果已有 owner，先 `_clean_up_owner()` 并从旧 owner.owned 移除。" },
          { title: "拒绝自己", text: "owner 不能是节点自身。" },
          { title: "允许 null", text: "传 null 表示清除保存归属。" },
          { title: "检查祖先", text: "非空 owner 必须 `is_ancestor_of(this)`。" },
          { title: "写入关系", text: "`_set_owner_nocheck()` 设置 owner，并登记到 owner.owned。" },
          { title: "更新编辑器状态", text: "触发 owner changed、unique name 和配置警告刷新。" }
        ]
      },
      {
        type: "table",
        title: "owner 合法性例子",
        headers: ["节点路径", "给谁设置 owner", "是否合法", "原因"],
        rows: [
          ["`Root/Marker`", "`Root`", "合法", "Root 是 Marker 的祖先。"],
          ["`Root/Level/Enemy`", "`Level`", "合法", "owner 可以是直接 parent。"],
          ["`Root/Level/Enemy/Weapon`", "`Root`", "合法", "owner 不必是直接 parent，只要是祖先。"],
          ["`Root/Level/Enemy`", "`Enemy`", "非法", "owner 不能是节点自己。"],
          ["`Root/A/Node`", "`Root/B`", "非法", "Root/B 不是 Node 的祖先。"],
          ["`DetachedNode`", "`Root`", "通常非法", "节点还没挂到 Root 下，Root 还不是它祖先。"]
        ]
      },
      {
        type: "heading",
        title: "PackedScene.pack 怎么使用 owner"
      },
      {
        type: "paragraph",
        text: "`PackedScene.pack(scene)` 最终调用 `SceneState::_parse_node(scene, scene, -1, ...)`，入口在 `scene/resources/packed_scene.cpp:1366`。真正筛选节点的是 `_parse_node()`，源码在 `packed_scene.cpp:792`。它一开始就过滤：如果当前节点不是保存根、`p_node->get_owner() != p_owner`，并且也不是 editable instance 允许的内容，就直接 `return OK`，也就是不保存这个节点。"
      },
      {
        type: "paragraph",
        text: "后面 `_parse_node()` 还会写 `NodeData.owner`：保存根节点 owner 是 `-1`；属于当前保存场景的节点 owner 是 `0`；复杂实例边界可能用路径或其他索引表达。实例化时，`SceneState::instantiate()` 在 `packed_scene.cpp:567` 附近读回 `n.owner`，找到 owner 后调用 `node->_set_owner_nocheck(owner)` 恢复归属关系。"
      },
      {
        type: "flow",
        title: "保存场景时 owner 的判断路径",
        steps: [
          { title: "PackedScene.pack(scene)", text: "以 scene 作为保存根 p_owner。" },
          { title: "_parse_node(p_owner, node)", text: "递归检查每个子节点。" },
          { title: "归属过滤", text: "不是根、owner 不是 p_owner、也不是 editable instance，则跳过。" },
          { title: "写 NodeData", text: "保存 name、type、parent、owner、属性、组、连接需要的数据。" },
          { title: "递归子节点", text: "只有符合边界的数据进入 SceneState 表。" },
          { title: "保存资源", text: "SceneState 再被 ResourceSaver 写成 .tscn/.scn。" }
        ]
      },
      {
        type: "table",
        title: "SceneState 中 owner 的几种含义",
        headers: ["值/形式", "含义", "出现场景"],
        rows: [
          ["`-1`", "没有保存 owner，或保存根本身。", "场景根、被跳过的外部归属边界。"],
          ["`0`", "owner 是当前保存场景根。", "普通子节点属于当前场景。"],
          ["`> 0`", "owner 指向 SceneState 里另一个节点索引。", "实例化/继承/可编辑实例的复杂恢复路径。"],
          ["`FLAG_ID_IS_PATH | id`", "owner 或 parent 用 NodePath/id path 表示。", "跨实例边界、无法用本地节点下标直接表达。"]
        ]
      },
      {
        type: "heading",
        title: "owner 和唯一名称"
      },
      {
        type: "paragraph",
        text: "`unique_name_in_owner` 也依赖 owner。`_acquire_unique_name_in_owner()` 在 `node.cpp:2235`，会把 `%Name` 形式的唯一名登记到 `data.owner->data.owned_unique_nodes`。如果同一个 owner 作用域里已经有节点占用这个唯一名，Godot 会警告并取消后一个节点的唯一名标记。"
      },
      {
        type: "paragraph",
        text: "这解释了为什么 `%Player` 这种路径不是全局唯一，而是在 owner 代表的场景边界里唯一。子场景、editable instance 和继承场景会让 owner 边界变复杂，所以查唯一名问题时要同时看 parent 树和 owner 链。"
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：工具脚本动态创建并保存节点"
      },
      {
        type: "code",
        code: [
          "@tool",
          "extends Node2D",
          "",
          "func add_spawn_point() -> void:",
          "    var marker := Marker2D.new()",
          "    marker.name = \"SpawnPoint\"",
          "    add_child(marker)",
          "",
          "    # parent 让它进入树；owner 让它属于当前场景。",
          "    marker.owner = get_tree().edited_scene_root if Engine.is_editor_hint() else owner"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "实际工具脚本里要根据上下文选择正确场景根；核心原则是 owner 必须是 marker 的祖先，并且应该是你想保存到的那个场景根。"
      },
      {
        type: "subheading",
        title: "案例二：C++ 插件创建子节点"
      },
      {
        type: "code",
        code: [
          "Node *marker = memnew(Node);",
          "marker->set_name(\"GeneratedMarker\");",
          "scene_root->add_child(marker);",
          "marker->set_owner(scene_root); // 让 PackedScene.pack(scene_root) 保存它。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "顺序很重要：先 add_child，让 scene_root 成为 marker 的祖先；再 set_owner，否则 owner 合法性检查会失败。"
      },
      {
        type: "subheading",
        title: "案例三：find_child 默认只找 owned 节点"
      },
      {
        type: "code",
        code: [
          "var saved_marker := find_child(\"SpawnPoint\")",
          "var runtime_marker := find_child(\"TemporaryEffect\", true, false)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`find_child(pattern, recursive, owned)` 的 `owned` 默认是 true。源码在 `node.cpp:1989` 附近会跳过没有 owner 的子节点。运行时临时节点如果没有 owner，查找时要显式传 false。"
      },
      {
        type: "subheading",
        title: "案例四：错误 owner 会直接报错"
      },
      {
        type: "code",
        code: [
          "var a := Node.new()",
          "var b := Node.new()",
          "add_child(a)",
          "add_child(b)",
          "",
          "# b 不是 a 的祖先，这会失败。",
          "a.owner = b"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这种错误不是保存时才暴露；`set_owner()` 阶段就会检查 owner 必须是祖先。"
      },
      {
        type: "heading",
        title: "调试动态节点没保存"
      },
      {
        type: "table",
        title: "排查清单",
        headers: ["检查点", "怎么判断", "典型修复"],
        rows: [
          ["节点是否在树里", "`node.get_parent()`、场景树面板。", "先 `parent.add_child(node)`。"],
          ["owner 是否为空", "`node.owner == null`。", "设置为要保存的场景根或合法祖先。"],
          ["owner 是否是祖先", "`owner.is_ancestor_of(node)`。", "先调整 parent，再设置 owner。"],
          ["保存根是否正确", "PackedScene.pack 传入的是哪个 root。", "把 owner 指向 pack 的那个场景根。"],
          ["是否在子场景实例里", "节点属于 inherited/instanced scene。", "理解 editable children 和本地覆盖边界。"],
          ["是否只用 find_child 找不到", "`owned` 参数是否默认 true。", "需要运行时临时节点时传 `owned=false`。"]
        ]
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "owner 不负责的事",
        headers: ["概念", "它负责什么", "和 owner 的边界"],
        rows: [
          ["内存生命周期", "Object/Node 的创建、删除、queue_free。", "owner 不释放节点，`owned` 不是所有权智能指针。"],
          ["SceneTree", "运行时调度、current_scene、删除队列。", "SceneTree 不用 owner 决定每帧调度。"],
          ["parent", "运行时父子结构。", "parent 对了只能说明节点在树里，不说明会保存。"],
          ["ResourceLoader", "按路径加载资源。", "owner 不参与资源路径解析。"],
          ["PackedScene", "保存/实例化场景蓝图。", "PackedScene 使用 owner 判断保存边界。"],
          ["NodePath", "运行时或保存数据中的路径引用。", "owner 会影响某些保存路径如何计算，但不等于路径本身。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：owner 是 C++ 内存所有者。不是；Node 释放仍走 remove/free/queue_free 等生命周期。",
          "误区二：parent 是谁，owner 就是谁。不是；owner 可以是任意祖先，常常是场景根。",
          "误区三：add_child 后节点一定会保存。不会；保存还要看 owner 和 PackedScene/SceneState 规则。",
          "误区四：owner 可以指向任意节点。不能；公开 API 要求 owner 是当前节点祖先。",
          "误区五：current_scene 就是 owner。不是；current_scene 是 SceneTree 的主场景引用，owner 是 Node 的保存归属。",
          "误区六：没有 owner 的节点没用。运行时临时特效、调试节点、缓存节点可以故意不设置 owner。",
          "误区七：所有子场景内部节点都能随便改 owner 保存。实例、继承和 editable children 会限制保存边界。",
          "误区八：find_child 找不到就说明节点不存在。默认 `owned=true` 可能跳过未设置 owner 的运行时节点。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/main/node.h` 里的 `Data` 结构，把 `parent`、`owner`、`owned`、`owned_unique_nodes` 放在一起看。",
          "读 `Node::set_owner()`、`_set_owner_nocheck()`、`_clean_up_owner()`，理解公开检查和内部恢复的区别。",
          "读 `Node::add_child()` 里 owner 不一致警告，确认 parent 变化可能让 owner 失效。",
          "读 `SceneState::_parse_node()` 开头的过滤条件，理解为什么没 owner 的节点不会保存。",
          "读 `_parse_node()` 写 `nd.owner` 的段落，理解 SceneState 如何记录归属。",
          "读 instantiate 时恢复 owner 的代码，确认 .tscn 里的 owner 最后会回到 Node::Data。",
          "最后读 `find_child()` 的 `owned` 参数和 unique name 逻辑，理解 owner 还影响编辑器查找与 `%Name` 作用域。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "owner 是 Node 的“保存归属”，不是父子关系也不是内存所有权；动态节点想进入 .tscn，必须既在正确 parent 树下，又把 owner 指到正确的场景祖先。"
      }
    ]
  },
  {
    id: "rid",
    title: "RID",
    aliases: ["RID", "RID_Alloc", "RID_Owner", "RID_PtrOwner", "Resource ID", "Server 句柄", "get_or_null", "make_rid", "free_rid", "RenderingServer"],
    summary: "Server 世界里的轻量句柄：外部代码只拿 64-bit RID，真实 texture/body/mesh 等对象由对应 Server 的 RID_Owner 或 storage 管理。",
    article: [
      {
        type: "lead",
        text: "RID 是 Godot Server 层使用的轻量资源句柄。场景节点、Resource 或脚本不直接拿渲染纹理、物理 body、导航 map 的内部对象指针，而是拿一个 RID。对应 Server 再用这个 RID 找到自己内部真正的对象槽，执行创建、更新、查询和释放。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 RID 想成取餐牌号码。你手里只有一个号码，真正的餐在厨房。你把号码交给工作人员，工作人员根据号码找到对应餐盘。号码本身不是餐，也不会保存餐的内容；号码只是让外部代码能安全地引用 Server 内部对象。"
      },
      {
        type: "paragraph",
        text: "比如 Sprite2D、MeshInstance3D 或 RigidBody3D 这些节点是你在场景树里操作的对象；渲染后端里的 texture、mesh、instance，物理后端里的 body、shape，通常是 Server 内部对象。场景层只保存 RID 或通过 Resource 间接持有 RID，不需要知道 Vulkan、Godot Physics、Jolt 等后端对象长什么样。"
      },
      {
        type: "flow",
        title: "RID 的直觉流程",
        steps: [
          { title: "节点或资源请求对象", text: "例如创建 texture、mesh、body、shape、canvas item。" },
          { title: "调用对应 Server", text: "RenderingServer、PhysicsServer、NavigationServer 等创建内部对象。" },
          { title: "Server 返回 RID", text: "外部只保存一个 64-bit 句柄。" },
          { title: "后续更新传 RID", text: "设置参数、绑定关系、查询状态时把 RID 传回 Server。" },
          { title: "Server 查真实对象", text: "RID_Owner 或 storage 用 `get_or_null()` 找到对象槽。" },
          { title: "释放时传 RID", text: "调用对应 Server 的 `free_rid()` 或专用释放函数。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`RID` 定义在 `core/templates/rid.h:38`。它内部只有一个 `uint64_t _id`，空 RID 的 `_id` 是 0；`is_valid()` 只是判断 `_id != 0`；`get_local_index()` 取低 32 位。RID 本身不存任何 texture、mesh、body 数据，也没有析构真实后端对象的能力。"
      },
      {
        type: "paragraph",
        text: "真正管理对象的是 `RID_Alloc` 和它外面的 `RID_Owner` / `RID_PtrOwner`。`RID_Alloc` 在 `core/templates/rid_owner.h:92`，内部把对象放在 chunk 数组里，并维护 free list、validator、alloc_count。RID 的低 32 位是本地槽位 index，高 32 位是 validator，用来发现旧 RID 指向已释放或复用槽位的问题。"
      },
      {
        type: "table",
        title: "RID 三层结构",
        headers: ["层", "源码锚点", "职责", "不负责"],
        rows: [
          ["`RID`", "`core/templates/rid.h:38`", "保存 64-bit id，支持比较、hash、valid/null 判断。", "不保存真实对象，也不释放后端资源。"],
          ["`RID_Alloc<T>`", "`rid_owner.h:92`", "分配 RID、管理 chunk/free list/validator、初始化和释放对象槽。", "不暴露具体 Server 语义。"],
          ["`RID_Owner<T>`", "`rid_owner.h:517`", "包装 RID_Alloc，提供 `make_rid()`、`get_or_null()`、`free()`。", "不决定渲染或物理命令什么时候执行。"],
          ["`RID_PtrOwner<T>`", "`rid_owner.h:458`", "存指针型对象，`get_or_null()` 返回 T*。", "不拥有外部所有生命周期规则。"],
          ["Server/storage", "例如 RenderingDevice、TextureStorage、PhysicsServer 实现。", "定义对象类型、状态更新、线程边界、真正释放逻辑。", "不让场景节点直接碰后端内部结构。"]
        ]
      },
      {
        type: "heading",
        title: "RID_Alloc 如何防旧句柄误用"
      },
      {
        type: "paragraph",
        text: "`RID_Alloc::_allocate_rid()` 在 `rid_owner.h:109`。它从 free list 拿一个空槽位，生成新的 validator，把 validator 放进 RID 高 32 位，把槽位 index 放进低 32 位，并在槽位 validator 上临时打上 `0x80000000` 未初始化标记。`make_rid()` 会紧接着调用 `initialize_rid()` 构造 T。"
      },
      {
        type: "paragraph",
        text: "`get_or_null()` 在 `rid_owner.h:191`。它先处理空 RID，再取低 32 位 index，确认 index 没越过 max_alloc，然后取高 32 位 validator，与槽位当前 validator 比较。比较失败返回 null；如果槽位还带未初始化标记，会报 “Attempting to use an uninitialized RID”。"
      },
      {
        type: "paragraph",
        text: "`free()` 在 `rid_owner.h:330`。它再次验证 index 和 validator，调用对象析构，把槽位 validator 设成 `0xFFFFFFFF` 表示无效，然后把 index 放回 free list。这样旧 RID 的低 32 位即使以后复用了同一个槽，validator 也大概率不同，旧句柄不会静默指向新对象。"
      },
      {
        type: "flow",
        title: "RID 分配、查询、释放的内部路径",
        steps: [
          { title: "allocate slot", text: "从 free list 取 index，必要时扩展 chunk。" },
          { title: "generate validator", text: "生成高 32 位校验值，组合成 64-bit RID。" },
          { title: "initialize object", text: "`initialize_rid()` 在槽位上 placement-new 构造 T。" },
          { title: "get_or_null", text: "用 RID 的 index 定位槽位，用 validator 校验是否仍是同一个对象。" },
          { title: "free", text: "析构 T，槽位设为无效，index 回到 free list。" },
          { title: "old RID fails", text: "旧 RID 再查时 validator 不匹配，返回 null 或报错。" }
        ]
      },
      {
        type: "table",
        title: "RID 的 64-bit id 怎么读",
        headers: ["部分", "来源", "用途", "风险控制"],
        rows: [
          ["低 32 位", "槽位 index / local index。", "快速定位 chunk 里的对象槽。", "index 越界直接返回 null。"],
          ["高 32 位", "validator。", "确认这个槽当前仍属于同一次分配。", "释放后 validator 变化或变无效，旧 RID 失效。"],
          ["`_id == 0`", "默认构造 RID。", "表示空句柄。", "`get_or_null(RID())` 返回 null。"],
          ["未初始化位", "`0x80000000`。", "allocate 后 initialize 前的保护状态。", "错误使用未初始化 RID 会报错。"]
        ]
      },
      {
        type: "heading",
        title: "RID 和 Server 的关系"
      },
      {
        type: "paragraph",
        text: "Server API 经常返回 RID。`RenderingServer` 在 `servers/rendering/rendering_server.h` 中有大量 `texture_2d_create()`、`mesh_create()`、`canvas_item_create()`、`instance_create()`，都返回 RID；`PhysicsServer2D/3D` 里的 shape、space、area、body、joint 创建函数也返回 RID。释放入口通常是对应 Server 的 `free_rid()`。"
      },
      {
        type: "paragraph",
        text: "以渲染纹理为例，`TextureStorage` 在 `servers/rendering/renderer_rd/storage_rd/texture_storage.h:209` 有 `RID_Owner<Texture, true> texture_owner`。`texture_allocate()` 在 `texture_storage.cpp:936` 返回 `texture_owner.allocate_rid()`，`texture_2d_initialize()` 在 `:1033` 用 `texture_owner.initialize_rid()` 写入真实 Texture，`texture_free()` 在 `:939` 先 `get_or_null()` 拿到 Texture，再 cleanup，最后 `texture_owner.free()`。"
      },
      {
        type: "flow",
        title: "场景层到 Server 后端",
        steps: [
          { title: "高层对象", text: "Sprite2D、MeshInstance3D、CollisionShape3D、Resource 等保存用户语义。" },
          { title: "提交给 Server", text: "调用 RenderingServer/PhysicsServer 创建或更新对象。" },
          { title: "得到 RID", text: "高层对象只保存句柄，不直接保存后端对象指针。" },
          { title: "Server storage", text: "RID_Owner 或专用 storage 保存真实 Texture/Body/Shape/Instance。" },
          { title: "帧阶段执行", text: "渲染/物理可能排队或在线程边界统一执行。" },
          { title: "退出时释放", text: "高层对象释放或退出树时调用对应 Server 释放 RID。" }
        ]
      },
      {
        type: "table",
        title: "常见 RID 生产者",
        headers: ["Server", "典型 RID", "创建入口", "释放入口"],
        rows: [
          ["RenderingServer", "texture、shader、material、mesh、viewport、canvas item、instance。", "`texture_2d_create()`、`mesh_create()`、`canvas_item_create()`、`instance_create()`。", "`RenderingServer::free_rid()`。"],
          ["PhysicsServer2D", "space、area、body、shape、joint。", "`space_create()`、`body_create()`、`circle_shape_create()`、`joint_create()`。", "`PhysicsServer2D::free_rid()`。"],
          ["PhysicsServer3D", "space、area、body、shape、soft body、joint。", "`body_create()`、`sphere_shape_create()`、`joint_create()`。", "`PhysicsServer3D::free_rid()`。"],
          ["RenderingDevice", "buffer、texture、framebuffer、pipeline、uniform set。", "`texture_create()`、`vertex_buffer_create()`、`compute_pipeline_create()` 等。", "`RenderingDevice::free_rid()`。"],
          ["NavigationServer", "map、region、agent、link、obstacle。", "导航 Server 创建 API。", "对应 Server 的 free 或 free_rid。"]
        ]
      },
      {
        type: "heading",
        title: "RID 不是 Resource，也不是 ObjectID"
      },
      {
        type: "paragraph",
        text: "Resource 是用户可见、可序列化、通常 RefCounted 的对象，例如 Texture2D、Mesh、Material。它可能内部持有一个或多个 RID，把数据提交到 RenderingServer。RID 是 Server 层句柄，不负责 ResourceLoader 缓存、路径、引用计数或 .tres/.res 序列化。"
      },
      {
        type: "paragraph",
        text: "ObjectID 是 ObjectDB 用来弱索引 Object 的 ID，适合 Callable、Signal、删除队列这类对象系统场景；RID 是 Server storage 用来索引后端资源的 ID。把 RID 当 ObjectID 查，或把 ObjectID 传给 Server，都是层级混淆。"
      },
      {
        type: "table",
        title: "RID、ObjectID、Resource 的边界",
        headers: ["概念", "指向什么", "谁管理", "典型误用"],
        rows: [
          ["RID", "Server 内部对象槽。", "RID_Owner、RID_PtrOwner、Server/storage。", "以为 RID 本身包含真实数据或会自动释放。"],
          ["ObjectID", "ObjectDB 中的 Object。", "Object 构造/析构登记到 ObjectDB。", "拿它访问 Server 后端资源。"],
          ["Resource", "用户资源对象，通常 RefCounted。", "ResourceLoader、ResourceCache、Ref。", "把 Resource 当 RID，或以为释放 Resource 一定立即释放所有后端 RID。"],
          ["Node", "场景树运行时对象。", "SceneTree 和父子关系。", "以为 delete Node 会自动释放所有手动创建 RID。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：直接使用 RenderingServer 创建 canvas item"
      },
      {
        type: "code",
        code: [
          "var item := RenderingServer.canvas_item_create()",
          "RenderingServer.canvas_item_set_parent(item, get_canvas_item())",
          "RenderingServer.canvas_item_add_rect(item, Rect2(Vector2.ZERO, Vector2(64, 64)), Color.RED)",
          "",
          "# 不用了要交还给 RenderingServer。",
          "RenderingServer.free_rid(item)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`item` 是 RID，不是 Node。它不会出现在 SceneTree，也没有 `_ready()`、`queue_free()` 或 owner。"
      },
      {
        type: "subheading",
        title: "案例二：物理查询里排除某些 RID"
      },
      {
        type: "code",
        code: [
          "var query := PhysicsRayQueryParameters3D.create(from, to)",
          "query.exclude = [player.get_rid()]",
          "var hit := get_world_3d().direct_space_state.intersect_ray(query)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这里排除的是物理 Server 认识的 RID。高层 Node 负责提供友好的 API，底层查询最终还是在 PhysicsServer 的 space/body/shape 数据上工作。"
      },
      {
        type: "subheading",
        title: "案例三：C++ storage 通过 RID_Owner 管对象"
      },
      {
        type: "code",
        code: [
          "RID rid = texture_owner.make_rid(texture);",
          "Texture *texture_ptr = texture_owner.get_or_null(rid);",
          "ERR_FAIL_NULL(texture_ptr);",
          "",
          "texture_ptr->width = 1024;",
          "texture_owner.free(rid);"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "真实代码通常还会做后端驱动释放、依赖处理和线程检查。RID_Owner 解决的是“句柄到对象槽”的基础映射，不替代 Server 的业务释放逻辑。"
      },
      {
        type: "subheading",
        title: "案例四：不要跨 Server 混用 RID"
      },
      {
        type: "code",
        code: [
          "var texture_rid := RenderingServer.texture_2d_create(image)",
          "",
          "# 错误心智：把渲染纹理 RID 当物理 body RID 使用。",
          "# PhysicsServer3D.body_set_state(texture_rid, PhysicsServer3D.BODY_STATE_TRANSFORM, Transform3D.IDENTITY)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "RID 的数字格式统一，但语义属于创建它的 Server/storage。不同 owner 不会因为同样叫 RID 就能互相理解。"
      },
      {
        type: "heading",
        title: "RID 和相邻概念的边界"
      },
      {
        type: "table",
        headers: ["概念", "边界"],
        rows: [
          ["RID", "Server/storage 内部对象句柄，释放走对应 Server。"],
          ["Resource", "用户可见、可序列化、常由 Ref 管生命周期的资源对象。"],
          ["ObjectID", "ObjectDB 的弱查找 ID，只能回到 Object 世界。"],
          ["Node", "场景树运行对象，可能持有或间接创建 RID，但不等于 RID。"],
          ["RenderingDevice RID", "也是 RID 形状，但属于 RD owner，不能和 RenderingServer 高层 RID 随意混用。"]
        ]
      },
      {
        type: "heading",
        title: "调试 RID 问题"
      },
      {
        type: "table",
        title: "常见问题定位",
        headers: ["现象", "优先检查", "原因"],
        rows: [
          ["RID is null/invalid", "创建 API 是否返回空 RID，资源输入是否为空。", "创建失败或还未 initialize。"],
          ["get_or_null 返回 null", "RID 是否属于当前 owner，是否已被释放。", "validator 不匹配、index 越界或跨 Server 混用。"],
          ["释放时报错", "是否重复 free，是否释放未初始化 RID。", "free 会校验 uninitialized bit 和 validator。"],
          ["节点删了后后端对象还在", "是否有手动创建 RID 没调用 Server free。", "RID 不跟 Node 自动绑定，除非封装层负责释放。"],
          ["资源释放不及时", "ResourceCache、Ref、Server 延迟释放队列。", "Resource 和 RID 是不同生命周期层。"],
          ["渲染/物理不是马上变化", "Server 调用是否排队，帧阶段 sync/step/draw。", "很多 Server 命令在固定阶段统一执行。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：RID 就是真实对象。不是；RID 只是 64-bit 句柄。",
          "误区二：RID 会自动释放后端资源。不会；释放要走创建它的 Server 或 owner。",
          "误区三：RID 和 Resource 是一回事。Resource 是用户对象，RID 是 Server 句柄。",
          "误区四：RID 和 ObjectID 可以互换。ObjectID 查 ObjectDB，RID 查 Server storage。",
          "误区五：所有 RID 都能传给任何 Server。不同 Server/owner 管不同对象，不能混用。",
          "误区六：get_local_index 就足够识别对象。不够；还要 validator 防止旧句柄误指新槽。",
          "误区七：释放 Node 一定释放所有 RID。只有封装层写了释放逻辑才会释放对应 Server 对象。",
          "误区八：Server 调用一定立即影响后端。渲染和物理常有线程边界、命令队列和帧同步点。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/templates/rid.h:38`，确认 RID 内部只有 `uint64_t _id`。",
          "读 `RID_Alloc::_allocate_rid()`，理解 index、validator、free list 和未初始化标记。",
          "读 `RID_Alloc::get_or_null()`，看旧 RID 如何被 validator 拦住。",
          "读 `RID_Alloc::free()`，看对象析构、validator 失效和 slot 回收。",
          "读 `RID_Owner` 与 `RID_PtrOwner`，区分值对象 storage 和指针 storage。",
          "找一个具体 Server storage，例如 TextureStorage 的 `texture_owner`，跟 `allocate/initialize/get/free` 完整走一遍。",
          "最后读 RenderingServer/PhysicsServer 的 create/free_rid API，建立高层节点到后端对象的边界。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "RID 是 Godot Server 层的安全号码牌：外部拿它引用后端对象，真实数据和释放规则始终留在创建它的 Server/storage 里。"
      }
    ]
  },
  {
    id: "server",
    title: "Server 架构",
    aliases: ["Server", "Server 架构", "RenderingServer", "PhysicsServer", "DisplayServer", "AudioServer", "TextServer", "NavigationServer", "XRServer", "Server API"],
    summary: "Godot 的低层隔离层：场景节点表达用户语义，Server 保存真实执行状态，后端把状态变成渲染、物理、音频、文本、导航或平台操作。",
    article: [
      {
        type: "lead",
        text: "Server 是 Godot 在高层场景系统和底层后端之间建立的隔离层。Sprite2D、MeshInstance3D、RigidBody3D、AudioStreamPlayer 等节点负责用户可见语义；RenderingServer、PhysicsServer、AudioServer、TextServer、NavigationServer、DisplayServer 等负责维护真实执行状态；Vulkan、Godot Physics、Jolt、音频 driver、平台窗口系统等后端负责把状态变成实际结果。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Godot 想成一家公司。Node 是前台业务单：角色在哪里、贴图是什么、按钮文字是什么；Server 是专业部门：渲染部门管画面、物理部门管碰撞、音频部门管播放、文本部门管排版、窗口部门管平台窗口。前台不直接去操作显卡或物理求解器，而是把请求交给对应部门。"
      },
      {
        type: "paragraph",
        text: "这样做的好处是：场景代码稳定，底层可以换实现。比如物理可以用 Godot Physics、Jolt 或 Extension；渲染可以把调用排队到渲染线程；DisplayServer 可以按 Windows、macOS、Linux、Web、Android 分平台实现。高层节点只需要调用统一的 Server API。"
      },
      {
        type: "flow",
        title: "Server 架构的基本分层",
        steps: [
          { title: "Scene nodes", text: "保存用户语义和场景树关系，例如 Sprite2D、RigidBody3D、Label。" },
          { title: "Resource", text: "保存可复用数据，例如 Texture2D、Mesh、Material、AudioStream。" },
          { title: "Server API", text: "统一入口，例如 RenderingServer、PhysicsServer、TextServer。" },
          { title: "RID / state", text: "Server 用 RID 和内部 storage 保存真实执行对象。" },
          { title: "Backend", text: "Vulkan、物理库、音频 driver、平台窗口系统执行底层工作。" },
          { title: "Frame sync", text: "主循环在固定阶段 sync、step、draw、update。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "Server 不是某一个类，而是一组架构约定：全局单例或管理器、抽象 API、RID 句柄、内部 storage、可替换后端、固定帧同步点。主文档里的 Server 章节强调三件事：节点保存用户可见状态，Server 保存真实执行状态，后端负责实际执行。"
      },
      {
        type: "paragraph",
        text: "这些 Server 大多在 `servers/` 下定义。`RenderingServer` 在 `servers/rendering/rendering_server.h:64`，`DisplayServer` 在 `servers/display/display_server.h:62`，`PhysicsServer2D` 在 `servers/physics_2d/physics_server_2d.h:213`，`PhysicsServer3D` 在 `servers/physics_3d/physics_server_3d.h:236`，`AudioServer` 在 `servers/audio/audio_server.h:180`，`TextServer` 在 `servers/text/text_server.h:47`，导航和 XR 也分别有自己的 Server。"
      },
      {
        type: "table",
        title: "Server 架构里的几个角色",
        headers: ["角色", "例子", "负责什么", "不负责什么"],
        rows: [
          ["场景节点", "`Sprite2D`、`RigidBody3D`、`Label`、`AudioStreamPlayer`。", "用户 API、场景树生命周期、脚本交互。", "不直接拥有 GPU/物理/音频后端对象。"],
          ["Resource", "`Texture2D`、`Mesh`、`Material`、`AudioStream`。", "可复用数据、加载缓存、序列化。", "不是 Server storage 本体。"],
          ["Server API", "`RenderingServer`、`PhysicsServer3D`、`TextServer`。", "创建/更新/查询/释放底层状态。", "不保存高层场景树语义。"],
          ["RID", "texture RID、body RID、canvas item RID。", "外部引用 Server 内部对象的句柄。", "不保存真实对象数据。"],
          ["Backend", "Vulkan、Godot Physics、Jolt、platform display、audio driver。", "实际执行渲染、求解、混音、窗口操作。", "不直接暴露给普通场景节点。"]
        ]
      },
      {
        type: "heading",
        title: "注册和启动顺序"
      },
      {
        type: "paragraph",
        text: "`register_server_types()` 在 `servers/register_server_types.cpp:146`。它注册 TextServer、AudioServer、NavigationServer2D/3DManager、PhysicsServer2D/3DManager、XRServer 等类型，并把若干管理器加到 Engine singleton。`main/main.cpp:3197` 调用它；`DisplayServer::create()` 则在 `main/main.cpp:3343` 附近根据平台和渲染 driver 创建显示服务。"
      },
      {
        type: "paragraph",
        text: "Physics 和 Navigation 的后端选择走 Manager。`main/main.cpp:362` 和 `:382` 附近通过 `PhysicsServer3DManager`、`PhysicsServer2DManager` 创建实际物理 Server；`register_server_types.cpp:313`、`:356` 注册 Dummy 物理 Server 作为兜底；导航也有自己的 Manager 和 Dummy server。"
      },
      {
        type: "flow",
        title: "启动时 Server 大概怎样出现",
        steps: [
          { title: "创建管理器", text: "Main 初始化 TextServerManager、PhysicsServerManager、NavigationServerManager。" },
          { title: "register_server_types", text: "注册 Server 类、Manager、native struct 和 Engine singleton。" },
          { title: "选择平台服务", text: "DisplayServer 根据平台和 driver 创建窗口/显示服务。" },
          { title: "创建具体后端", text: "Physics/Navigation/Text 等 Manager 选择默认、项目设置或 Dummy 实现。" },
          { title: "加入运行循环", text: "主循环每帧在固定位置调用 sync/process/step/draw/update。" },
          { title: "退出清理", text: "finalize/unregister 阶段释放 Server 和管理器。"}
        ]
      },
      {
        type: "heading",
        title: "主循环里的 Server 同步点"
      },
      {
        type: "paragraph",
        text: "Server 调用不一定马上执行到底层。真正的同步点常在 `Main::iteration()`。物理阶段里，`main/main.cpp:4965` 调 `PhysicsServer3D::sync()`，`:4971` 调 `PhysicsServer2D::sync()`，随后 SceneTree 运行 physics_process，导航在 `:4994` 和 `:4998` 跑 physics_process，物理 Server 在 `:5010` 附近 step。"
      },
      {
        type: "paragraph",
        text: "普通 process 后，导航在 `main/main.cpp:5044`、`:5048` 更新，RenderingServer 在 `:5052` sync，在 `:5064` 或 `:5068` draw。帧尾还有 `AudioServer::update()`，源码在 `main/main.cpp:5087`。这解释了为什么你在节点里设置一个渲染或物理状态，后端可能要等到本帧或下一固定阶段才真正消费。"
      },
      {
        type: "table",
        title: "一帧中常见 Server 阶段",
        headers: ["阶段", "源码入口", "意义"],
        rows: [
          ["XR early process", "`XRServer::_process()`", "每帧早期更新 XR interface、tracker、pose 等状态。"],
          ["Physics sync", "`PhysicsServer2D/3D::sync()`、`flush_queries()`", "把待处理状态同步到物理后端，并处理查询回调。"],
          ["SceneTree physics", "`MainLoop::physics_process()`", "节点固定步逻辑运行，可能继续向 Server 提交状态。"],
          ["Navigation physics", "`NavigationServer2D/3D::physics_process()`", "导航地图、避障等固定步更新。"],
          ["Physics step", "`PhysicsServer2D/3D::step()`", "真正推进物理模拟。"],
          ["Navigation process", "`NavigationServer2D/3D::process()`", "普通帧导航处理。"],
          ["Rendering sync/draw", "`RenderingServer::sync()`、`draw()`", "同步渲染命令并提交绘制。"],
          ["Audio update", "`AudioServer::update()`", "更新音频状态、混音相关工作。"]
        ]
      },
      {
        type: "flow",
        title: "从节点属性变化到后端执行",
        steps: [
          { title: "用户改节点", text: "例如改 Sprite2D texture、RigidBody transform、Label text。" },
          { title: "节点/资源转成 Server 请求", text: "调用 RenderingServer、PhysicsServer、TextServer 等。" },
          { title: "Server 保存状态", text: "内部 storage 根据 RID 找对象，写入新参数或排队命令。" },
          { title: "主循环同步", text: "Main::iteration 在固定阶段 sync/step/draw/update。" },
          { title: "后端执行", text: "GPU 提交、物理求解、音频混音、文本 shaping、窗口事件处理。" },
          { title: "结果回到场景", text: "碰撞结果、渲染帧、音频状态、导航路径等被高层 API 消费。" }
        ]
      },
      {
        type: "heading",
        title: "主要 Server 的职责边界"
      },
      {
        type: "table",
        title: "主文档里的核心 Server",
        headers: ["Server", "源码锚点", "管理什么", "典型调用时机"],
        rows: [
          ["RenderingServer", "`servers/rendering/rendering_server.h:64`", "texture、material、mesh、canvas、viewport、camera、light、particles、global shader 参数。", "节点/资源变更时提交状态，帧末 sync/draw。"],
          ["DisplayServer", "`servers/display/display_server.h:62`", "窗口、屏幕、输入法、光标、剪贴板、原生菜单、平台显示能力。", "启动创建窗口，OS/platform 事件循环持续驱动。"],
          ["PhysicsServer2D/3D", "`physics_server_2d.h:213`、`physics_server_3d.h:236`", "space、area、body、shape、joint、direct state、query。", "固定物理步 sync、flush_queries、step。"],
          ["AudioServer", "`servers/audio/audio_server.h:180`", "bus、effect、stream playback、mixing、driver 输出。", "播放节点提交流，帧尾或音频线程更新。"],
          ["TextServer", "`servers/text/text_server.h:47`", "字体、shaping、fallback、断行、双向文本、字形缓存。", "Control/Label/RichTextLabel/TextEdit 布局文字时。"],
          ["NavigationServer2D/3D", "`navigation_server_2d.h:50`、`navigation_server_3d.h:46`", "map、region、agent、obstacle、link、路径查询、烘焙入口。", "physics/process 阶段更新导航状态。"],
          ["XRServer", "`servers/xr/xr_server.h:54`", "XR interface、tracker、pose、viewport 协作。", "每帧早期处理 XR 状态，并和渲染协作。"]
        ]
      },
      {
        type: "heading",
        title: "线程边界和可替换后端"
      },
      {
        type: "paragraph",
        text: "Server 架构允许“同一个 API，不同执行方式”。`RenderingServerDefault` 在 `servers/rendering/rendering_server_default.h:47` 继承 RenderingServer；如果创建渲染线程，`rendering_server_default.cpp:436` 的 `sync()` 会走 command_queue，`draw()` 在 `:443` 附近决定推到渲染线程还是直接 `_draw()`。"
      },
      {
        type: "paragraph",
        text: "物理也类似。`PhysicsServer3DWrapMT` 在 `servers/physics_3d/physics_server_3d_wrap_mt.h:63`，把底层 PhysicsServer3D 包一层多线程命令队列；`sync()`、`step()`、`flush_queries()` 在 wrap_mt cpp 里转发或同步到底层实现。读物理 bug 时要分清抽象接口、wrap_mt 代理和真正后端。"
      },
      {
        type: "table",
        title: "同一个 Server API 后面可能是什么",
        headers: ["层", "例子", "阅读重点"],
        rows: [
          ["抽象接口", "`RenderingServer`、`PhysicsServer3D`。", "用户和节点看到的统一方法签名。"],
          ["默认实现", "`RenderingServerDefault`。", "命令队列、渲染线程、sync/draw。"],
          ["多线程包装", "`PhysicsServer3DWrapMT`、`PhysicsServer2DWrapMT`。", "调用是否排队，何时 push_and_sync。"],
          ["具体后端", "Godot Physics、Jolt、Vulkan/RD、平台 DisplayServer。", "真实数据结构和执行成本。"],
          ["Dummy/Extension", "Dummy server、GDExtension server。", "禁用功能、测试、替换实现。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：直接画一个低层 canvas item"
      },
      {
        type: "code",
        code: [
          "extends Node2D",
          "",
          "var item: RID",
          "",
          "func _ready() -> void:",
          "    item = RenderingServer.canvas_item_create()",
          "    RenderingServer.canvas_item_set_parent(item, get_canvas_item())",
          "    RenderingServer.canvas_item_add_rect(item, Rect2(Vector2.ZERO, Vector2(80, 40)), Color.ORANGE)",
          "",
          "func _exit_tree() -> void:",
          "    if item.is_valid():",
          "        RenderingServer.free_rid(item)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个例子绕开了普通 CanvasItem 的高级封装。它展示了 Server API 的本质：创建 RID，设置状态，退出时交还给对应 Server。"
      },
      {
        type: "subheading",
        title: "案例二：物理查询看起来是场景 API，底下走 Server"
      },
      {
        type: "code",
        code: [
          "func scan(from: Vector3, to: Vector3) -> Dictionary:",
          "    var query := PhysicsRayQueryParameters3D.create(from, to)",
          "    query.exclude = [get_rid()]",
          "    return get_world_3d().direct_space_state.intersect_ray(query)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "脚本入口很友好，但查询最终落在 PhysicsServer 的 space/body/shape 数据上。`exclude` 里放的也是 RID。"
      },
      {
        type: "subheading",
        title: "案例三：Label 文本不是简单字符串绘制"
      },
      {
        type: "code",
        code: [
          "func set_title(text: String) -> void:",
          "    $TitleLabel.text = text",
          "    # Control/Label 后面会使用 TextServer 做 shaping、fallback、断行和字形缓存。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "文本系统是 Server 架构很典型的例子：高层 Label 保存用户语义，TextServer 处理复杂文字排版，RenderingServer 最后把字形绘制出来。"
      },
      {
        type: "subheading",
        title: "案例四：从源码定位一次“节点改了但画面没变”"
      },
      {
        type: "code",
        code: [
          "# 阅读路线，不是要写进项目的代码：",
          "# 1. 节点属性 setter 是否调用了 RenderingServer？",
          "# 2. 对应 RID 是否 valid？",
          "# 3. Server storage 是否保存了新状态？",
          "# 4. RenderingServer::sync()/draw() 是否在本帧执行？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Server bug 不能只看节点类。要顺着节点 setter、Resource、RID、Server storage、主循环同步点一路查。"
      },
      {
        type: "heading",
        title: "读 Server 源码的固定问题"
      },
      {
        type: "table",
        title: "三问法",
        headers: ["问题", "看哪里", "为什么重要"],
        rows: [
          ["谁拥有真实对象？", "RID_Owner、storage 类、后端对象数组/map。", "决定内存生命周期和泄漏位置。"],
          ["调用立即执行还是排队？", "RenderingServerDefault command_queue、PhysicsServerWrapMT、thread guard。", "决定修改何时生效、线程问题在哪里。"],
          ["生命周期在哪里结束？", "`free_rid()`、节点 `_exit_tree()`、Resource 析构、Server finalize。", "节点消失不等于后端对象自动消失。"],
          ["哪个阶段消费状态？", "`Main::iteration()` 的 sync/step/draw/update。", "决定为什么结果可能延迟到固定步或帧末。"],
          ["后端是否可替换？", "Manager、Dummy、Extension、platform 实现。", "决定 bug 是抽象层问题还是某个后端问题。"]
        ]
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "不要把这些层混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Node", "用户语义、场景树、脚本生命周期。", "不直接实现 GPU/物理/音频后端。"],
          ["Resource", "可加载、可复用、可序列化的数据对象。", "不等于 Server 内部对象槽。"],
          ["RID", "引用 Server 内部对象的句柄。", "不保存真实数据，不自动释放。"],
          ["Server", "抽象 API、真实执行状态、同步点。", "不负责场景树结构和 owner 保存归属。"],
          ["Backend/driver", "平台或库的真实执行。", "不直接暴露给大多数高层节点。"],
          ["MainLoop/SceneTree", "帧循环和 Node 调度。", "不替代 Server 的 storage 和后端执行。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Server 是网络服务器。这里的 Server 是引擎内部服务层，不是联网服务。",
          "误区二：节点直接持有后端对象。多数情况下节点持有用户状态或 RID，真实对象在 Server/storage。",
          "误区三：调用 Server 方法一定马上执行到底层。渲染、物理等可能排队、跨线程或等帧同步点。",
          "误区四：RID 释放等于 delete Node。RID 要走对应 Server 的 free，Node 生命周期是另一层。",
          "误区五：所有平台 Display/Rendering 行为都在同一个文件。平台 DisplayServer 和渲染后端会分实现。",
          "误区六：PhysicsServer3D 就是某一个物理库。它是抽象接口，具体后端可替换。",
          "误区七：TextServer 只是字体缓存。它还处理 shaping、fallback、双向文本、断行等复杂排版。",
          "误区八：读 Server bug 只看 API 声明。必须继续追 storage、RID owner、wrap_mt、主循环同步点和具体后端。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `servers/register_server_types.cpp:146`，知道哪些 Server 和 Manager 被注册。",
          "读 `main/main.cpp:3197` 到 DisplayServer 创建、Audio/XR/Physics/Navigation 初始化，建立启动顺序。",
          "选一个 Server 抽象接口，例如 `RenderingServer` 或 `PhysicsServer3D`，看它暴露哪些 RID 创建和更新 API。",
          "找对应默认实现或 storage，例如 RenderingServerDefault、TextureStorage、PhysicsServerWrapMT。",
          "顺着一个创建 API 查 RID_Owner，确认真实对象在哪里。",
          "读 `Main::iteration()` 中 physics/navigation/render/audio 的 sync/step/draw/update 顺序。",
          "最后回到节点类，确认高层属性 setter 如何把用户语义提交给 Server。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Server 架构把 Godot 分成“场景语义、统一服务、底层后端”三层：节点负责好用，Server 负责隔离和调度，后端负责真正执行。"
      }
    ]
  },
  {
    id: "renderingserver",
    title: "RenderingServer",
    aliases: ["RenderingServer", "RenderingServerDefault", "RendererRD", "RenderingDevice", "canvas_item_create", "instance_create", "texture_2d_create", "mesh_create", "sync", "draw"],
    summary: "Godot 渲染系统的统一 Server API：场景层通过 RID 提交 texture、mesh、canvas item、3D instance、viewport 等状态，默认实现再交给 Renderer/RD/DisplayServer。",
    article: [
      {
        type: "lead",
        text: "RenderingServer 是 Godot 场景层和渲染后端之间的核心接口。CanvasItem、Viewport、Mesh、Material、VisualInstance3D 等高层对象不会直接生成 GPU 命令；它们调用 RenderingServer 创建和更新 RID。RenderingServerDefault 再把这些请求交给 texture storage、canvas renderer、scene renderer、RenderingDevice 和 DisplayServer。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "你可以把 RenderingServer 想成“画面订单系统”。Sprite2D 说“我要画这张图”，MeshInstance3D 说“这个 Mesh 放在这个位置”，Viewport 说“这些东西画到这个窗口或纹理里”。它们不会自己和显卡说话，而是把订单交给 RenderingServer。"
      },
      {
        type: "paragraph",
        text: "RenderingServer 收到的订单通常用 RID 标记：texture RID、mesh RID、canvas item RID、instance RID、viewport RID。RID 像订单号，真实纹理、mesh、渲染实例和 GPU 资源都在渲染后端内部。每帧末，Godot 调用 `sync()` 和 `draw()`，把积累的渲染状态推进到真正绘制。"
      },
      {
        type: "flow",
        title: "从节点到画面的一条直觉路径",
        steps: [
          { title: "用户改节点", text: "修改 Sprite2D texture、MeshInstance3D transform、Viewport 设置。" },
          { title: "节点调用 RenderingServer", text: "创建或更新 canvas item、instance、viewport、material、mesh RID。" },
          { title: "Server 保存渲染状态", text: "RenderingServerDefault 分发到 canvas/scene/storage。" },
          { title: "Renderer 组织绘制", text: "2D canvas 或 3D scene 做排序、剔除、光照、材质处理。" },
          { title: "RenderingDevice 提交底层命令", text: "创建纹理、buffer、pipeline、draw list 等。" },
          { title: "DisplayServer present", text: "窗口 surface/swapchain 把最终图像呈现出来。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`RenderingServer` 声明在 `servers/rendering/rendering_server.h:64`，继承 Object，持有 `singleton`。它定义大量纯虚 API：texture、shader、material、mesh、viewport、camera、light、3D instance、canvas、canvas item、occluder、particles、environment 等。声明文件回答“渲染抽象有哪些”，不包含具体 Vulkan/D3D12/Metal/OpenGL 实现。"
      },
      {
        type: "paragraph",
        text: "`RenderingServer::create()` 在 `servers/rendering/rendering_server.cpp:51`，通过 `create_func` 创建具体实现。默认实现是 `RenderingServerDefault`，声明在 `servers/rendering/rendering_server_default.h:47`。它的职责是把统一 API 分发到 `RSG::texture_storage`、`RSG::canvas`、`RSG::scene`、RendererRD 和命令队列。"
      },
      {
        type: "table",
        title: "RenderingServer 在渲染分层中的位置",
        headers: ["层级", "源码入口", "负责什么", "不负责什么"],
        rows: [
          ["场景节点", "`CanvasItem`、`VisualInstance3D`、`Viewport`。", "用户属性、场景树生命周期、何时提交状态。", "不生成 GPU command buffer。"],
          ["RenderingServer API", "`rendering_server.h:64`。", "统一创建/更新/释放渲染对象的抽象方法。", "不写具体后端 pass。"],
          ["RenderingServerDefault", "`rendering_server_default.h:47`。", "默认实现、线程队列、分发到 storage/canvas/scene。", "不等同于完整 RendererRD pass。"],
          ["RendererCanvas / RendererScene", "`renderer_canvas_render.h`、`renderer_scene_render.h`。", "2D/3D 渲染组织、排序、剔除、光照、后处理。", "不处理 Node 生命周期。"],
          ["RenderingDevice", "`rendering_device.h`。", "纹理、buffer、shader、pipeline、draw/compute list 的底层图形 API 抽象。", "不解释 Sprite2D 的用户行为。"],
          ["DisplayServer", "`display_server.h`。", "窗口、surface、swapchain、present。", "不管理材质和场景剔除。"]
        ]
      },
      {
        type: "heading",
        title: "2D 路径：CanvasItem 提交 canvas item"
      },
      {
        type: "paragraph",
        text: "`CanvasItem` 是 2D 和 GUI 的共同渲染基础，源码在 `scene/main/canvas_item.*`。构造函数在 `canvas_item.cpp:1801` 调用 `RenderingServer::get_singleton()->canvas_item_create()` 得到 canvas item RID；析构在 `:1806` 调用 `free_rid(canvas_item)`。"
      },
      {
        type: "paragraph",
        text: "CanvasItem 进入树时，会把自己的 canvas item RID 接到父 CanvasItem 或 Viewport canvas 下。源码里 `canvas_item_set_parent()` 出现在 `canvas_item.cpp:253`、`:278`、`:298`，可见性在 `:102` 调 `canvas_item_set_visible()`，变换在 `:1126`、`:1129` 调 `canvas_item_set_transform()`。这说明 2D 节点不是自己画，而是持续把状态提交给 Server。"
      },
      {
        type: "flow",
        title: "CanvasItem 的 2D 提交流",
        steps: [
          { title: "CanvasItem 构造", text: "创建 canvas item RID。" },
          { title: "进入树", text: "挂到父 CanvasItem 或 Viewport canvas。" },
          { title: "属性变化", text: "visible、transform、material、z_index 等调用 `canvas_item_set_*`。" },
          { title: "draw 回调", text: "自定义绘制命令记录到 canvas item。" },
          { title: "RendererCanvas", text: "后端排序、裁剪、材质、纹理和 draw list。" },
          { title: "draw/present", text: "帧末 RenderingServer draw 后呈现到窗口或 Viewport texture。" }
        ]
      },
      {
        type: "heading",
        title: "3D 路径：VisualInstance3D 提交 instance"
      },
      {
        type: "paragraph",
        text: "`VisualInstance3D` 是 3D 可见对象的基础，声明在 `scene/3d/visual_instance_3d.h:39`。构造函数在 `visual_instance_3d.cpp:208` 调用 `RenderingServer::instance_create()` 得到 instance RID，并在 `:215` 析构时释放。它保存的是“这个世界里有一个渲染实例”，不是 Mesh 顶点数据。"
      },
      {
        type: "paragraph",
        text: "Mesh 和实例是两层：`VisualInstance3D::set_base()` 在 `visual_instance_3d.cpp:197` 调 `instance_set_base(instance, p_base)`，通常把 Mesh Resource 的 mesh RID 设为这个 instance 的 base。进入 World3D 后，`instance_set_scenario()` 在 `:93` 附近把实例放进 scenario；变换变化时，`:61` 附近调用 `instance_set_transform()`。"
      },
      {
        type: "flow",
        title: "MeshInstance3D 到 3D 后端",
        steps: [
          { title: "Mesh Resource", text: "保存 surface arrays 和材质，内部有 mesh RID。" },
          { title: "VisualInstance3D", text: "创建 instance RID。" },
          { title: "set_base", text: "把 mesh RID 作为 instance base。" },
          { title: "set_scenario", text: "进入 World3D 后加入当前 scenario。" },
          { title: "set_transform", text: "节点变换变化时提交 instance transform。" },
          { title: "RendererScene", text: "剔除、光照、阴影、GI、后处理，再交给 RenderingDevice。" }
        ]
      },
      {
        type: "heading",
        title: "sync 和 draw：为什么渲染常在帧末发生"
      },
      {
        type: "paragraph",
        text: "主循环在普通 process 之后调用 RenderingServer。`main/main.cpp:5052` 调 `RenderingServer::sync()`，`:5064` 或 `:5068` 调 `RenderingServer::draw(wants_present, scaled_step)`。这意味着节点在 `_process()` 里改渲染状态，通常会在同一帧后段统一同步和绘制，而不是 setter 里马上调用 GPU draw。"
      },
      {
        type: "paragraph",
        text: "`RenderingServerDefault::sync()` 在 `rendering_server_default.cpp:435`：有渲染线程时 `command_queue.sync()`，否则 `command_queue.flush_all()`。`draw()` 在 `:443`，先发 `frame_pre_draw` 信号，清 changes；有渲染线程时把 `_draw()` push 到 command_queue，否则直接 `_draw()`。"
      },
      {
        type: "table",
        title: "RenderingServer 的几个关键入口",
        headers: ["入口", "源码", "含义"],
        rows: [
          ["`RenderingServer::get_singleton()`", "`rendering_server.cpp:47`", "获取当前渲染 Server 单例。"],
          ["`RenderingServer::create()`", "`rendering_server.cpp:51`", "通过 create_func 创建具体实现。"],
          ["`canvas_item_create()`", "`rendering_server.h:799`", "创建 2D/GUI canvas item RID。"],
          ["`instance_create()`", "`rendering_server.h:726`", "创建 3D instance RID。"],
          ["`mesh_create()`", "`rendering_server.h:194`", "创建 mesh RID。"],
          ["`texture_2d_create()`", "`rendering_server.h:110`", "创建 texture RID。"],
          ["`free_rid()`", "`rendering_server.h:951`", "释放属于 RenderingServer 的 RID。"],
          ["`sync()` / `draw()`", "`rendering_server.h:966`、`:967`", "主循环帧末同步和绘制。"]
        ]
      },
      {
        type: "heading",
        title: "RenderingServerDefault 怎么分发"
      },
      {
        type: "paragraph",
        text: "`RenderingServerDefault` 的宏可以看出分发风格。`rendering_server_default.h:137` 附近的 `FUNCRIDTEX*` 宏先从 `RSG::texture_storage->texture_allocate()` 拿 RID；如果当前在线程允许的上下文，就直接 initialize；否则把 initialize 推进 `command_queue`。这就是很多渲染 API“返回 RID 很快，真实资源初始化可能排队”的原因。"
      },
      {
        type: "paragraph",
        text: "具体 RD 后端再继续往下：TextureStorage 使用 RID_Owner 管 texture，RendererCanvas 管 2D，RendererScene 管 3D，RenderingDevice 管底层 GPU 抽象，DisplayServer 管窗口和 present。读渲染 bug 时要先判断问题在哪一层，而不是一开始跳进 Vulkan。"
      },
      {
        type: "table",
        title: "RenderingServerDefault 后面的主要部件",
        headers: ["部件", "负责什么", "常见问题"],
        rows: [
          ["Texture/Material/Mesh storage", "保存渲染资源 RID 对应的后端状态。", "资源没上传、材质参数没更新、RID 泄漏。"],
          ["RendererCanvas", "2D/GUI canvas items 的排序、裁剪、材质、绘制。", "2D 节点不显示、z_index/clip 错误。"],
          ["RendererScene", "3D scenario、可见性、光照、阴影、后处理。", "Mesh 不显示、光照/阴影/剔除异常。"],
          ["RenderingDevice", "底层 texture、buffer、pipeline、draw/compute list。", "GPU 资源、shader/pipeline、同步和驱动问题。"],
          ["DisplayServer", "窗口 surface、swapchain、present。", "窗口、显示、vsync、平台 present 问题。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：用 RenderingServer 直接创建 2D 绘制对象"
      },
      {
        type: "code",
        code: [
          "extends Node2D",
          "",
          "var item: RID",
          "",
          "func _ready() -> void:",
          "    item = RenderingServer.canvas_item_create()",
          "    RenderingServer.canvas_item_set_parent(item, get_canvas_item())",
          "    RenderingServer.canvas_item_add_rect(item, Rect2(Vector2.ZERO, Vector2(100, 60)), Color.CORNFLOWER_BLUE)",
          "",
          "func _exit_tree() -> void:",
          "    if item.is_valid():",
          "        RenderingServer.free_rid(item)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个 item 不会出现在 SceneTree，不会收到 Node 生命周期通知；它只是 RenderingServer 管的 canvas item RID。"
      },
      {
        type: "subheading",
        title: "案例二：3D instance 的 base 和 transform 是两件事"
      },
      {
        type: "code",
        code: [
          "# 心智模型：MeshInstance3D 大致把 Mesh RID 和 instance RID 接起来。",
          "var mesh_rid := mesh.get_rid()",
          "var instance := RenderingServer.instance_create()",
          "RenderingServer.instance_set_base(instance, mesh_rid)",
          "RenderingServer.instance_set_scenario(instance, get_world_3d().scenario)",
          "RenderingServer.instance_set_transform(instance, global_transform)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "真实 MeshInstance3D 还处理材质覆盖、层、可见性、AABB、通知和释放。这个例子只展示 RenderingServer 的核心分层：资源 RID + 实例 RID + scenario + transform。"
      },
      {
        type: "subheading",
        title: "案例三：调试“改了贴图但没显示”"
      },
      {
        type: "code",
        code: [
          "# 阅读路线：",
          "# 1. Texture Resource 是否有有效 RID？",
          "# 2. CanvasItem/Sprite2D 是否把 texture RID 提交给 RenderingServer？",
          "# 3. canvas item 是否挂到正确 Viewport canvas？",
          "# 4. RenderingServer::sync()/draw() 是否执行？",
          "# 5. 后端 texture storage 是否上传成功？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "渲染问题通常跨节点、资源、Server、Renderer、RD 多层；只看 Sprite2D 文件很容易停在半路。"
      },
      {
        type: "subheading",
        title: "案例四：不要把 RenderingServer 当即时绘图库"
      },
      {
        type: "code",
        code: [
          "func _process(_delta: float) -> void:",
          "    RenderingServer.canvas_item_set_visible(item, visible)",
          "    # 这里只是在提交状态；真正绘制在主循环后段 sync/draw。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "RenderingServer API 更像“修改渲染世界状态”，不是每个调用都马上发 GPU draw call。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "不要把渲染层混在一起",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["CanvasItem", "2D/GUI 节点的可见性、变换、draw 命令入口。", "不管理 GPU 纹理 storage。"],
          ["VisualInstance3D", "3D instance RID、scenario、transform、base。", "不保存 Mesh surface 数据本体。"],
          ["RenderingServer", "统一渲染 API 和 RID 状态提交。", "不直接等于 Vulkan 或 RendererRD。"],
          ["RendererRD", "3D/2D 后端渲染组织、光照、后处理。", "不处理用户 Node 生命周期。"],
          ["RenderingDevice", "底层图形资源和命令抽象。", "不解释场景节点如何组织。"],
          ["DisplayServer", "窗口和 present 平台能力。", "不管理材质、Mesh 或场景剔除。"],
          ["Resource", "Texture/Mesh/Material 等可复用用户资源。", "不是 RenderingServer 内部对象本身。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：RenderingServer 就是渲染后端。它是统一 API；具体后端在 RenderingServerDefault、RendererRD、RenderingDevice 和 driver 后面。",
          "误区二：CanvasItem 自己画到屏幕。它创建 canvas item RID，并把绘制状态交给 RenderingServer。",
          "误区三：MeshInstance3D 直接保存 GPU mesh。Mesh Resource 有 mesh RID，VisualInstance3D 有 instance RID，后端对象在 Server/storage。",
          "误区四：每个 RenderingServer 调用都是即时 GPU 命令。很多调用只是改状态或排入 command_queue。",
          "误区五：看到 RID 就能随便 free。RenderingServer 创建的 RID 要交给 RenderingServer 释放，不能跨 Server 混用。",
          "误区六：渲染问题应该从 Vulkan 开始查。大多数问题先查节点提交、Resource RID、Server storage、sync/draw。",
          "误区七：DisplayServer 负责材质和光照。DisplayServer 主要管窗口、surface 和 present。",
          "误区八：RenderingDevice 能解释所有渲染行为。RenderingDevice 是底层图形 API 抽象，不知道 Sprite2D 或 Label 的用户语义。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `servers/rendering/rendering_server.h:64`，看统一 API 分成 texture、mesh、viewport、instance、canvas 等几大块。",
          "读 `CanvasItem` 构造、析构、enter tree 和 transform/visible 提交，理解 2D 路径。",
          "读 `VisualInstance3D` 构造、`set_base()`、`instance_set_scenario()`、`instance_set_transform()`，理解 3D instance 路径。",
          "读 `RenderingServerDefault` 的 `FUNCRIDTEX*` 宏和 `free_rid()`，看默认实现如何分发到 storage 和命令队列。",
          "读 `RenderingServerDefault::sync()` 和 `draw()`，把渲染状态提交和帧末绘制对上。",
          "再进入 RendererCanvas/RendererScene，区分 2D draw 组织和 3D scene rendering。",
          "最后进入 RenderingDevice 和平台 DisplayServer，定位 GPU 资源、pipeline、swapchain 和 present 问题。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "RenderingServer 是 Godot 渲染世界的统一订单台：节点和资源提交 RID 状态，默认 Server 分发给渲染后端，帧末再 sync/draw 到窗口或 Viewport。"
      }
    ]
  },
  {
    id: "canvasitem",
    title: "CanvasItem",
    aliases: ["CanvasItem", "Node2D", "Control", "_draw", "queue_redraw", "NOTIFICATION_DRAW", "canvas_item RID", "draw_rect", "z_index", "CanvasLayer"],
    summary: "Godot 2D 和 GUI 的共同渲染基类：每个 CanvasItem 持有 canvas item RID，把可见性、变换、材质、排序和 draw 命令提交给 RenderingServer。",
    article: [
      {
        type: "lead",
        text: "CanvasItem 是 Godot 2D 和 GUI 渲染的共同基础。Node2D 和 Control 都继承它，所以 2D 游戏对象、UI 控件、自定义绘制和许多编辑器面板最终都走同一套 canvas item RID、RenderingServer canvas API、Viewport/CanvasLayer 挂接和 RendererCanvas 后端。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 RenderingServer 是画面订单系统，CanvasItem 就是“能下 2D 绘制订单的节点”。你在 Node2D 或 Control 里设置 visible、modulate、z_index，或者在 `_draw()` 里调用 `draw_rect()`、`draw_texture()`，CanvasItem 会把这些信息转换成 RenderingServer 能理解的 canvas item 命令。"
      },
      {
        type: "paragraph",
        text: "CanvasItem 自己不是一张图片，也不是 GPU 资源。它更像一个 2D 绘制容器：它知道自己在场景树哪里、挂到哪个 Viewport 或 CanvasLayer、是否可见、怎么排序、有哪些绘制命令。真正画到屏幕，要等 RenderingServer 在帧末处理 canvas。"
      },
      {
        type: "flow",
        title: "CanvasItem 的工作直觉",
        steps: [
          { title: "创建节点", text: "Node2D、Control 或自定义 CanvasItem 子类构造。" },
          { title: "创建 RID", text: "构造时调用 `canvas_item_create()`。" },
          { title: "进入树", text: "挂到父 CanvasItem、CanvasLayer 或 Viewport canvas。" },
          { title: "提交状态", text: "visible、transform、z_index、material 等提交到 RenderingServer。" },
          { title: "重绘", text: "`queue_redraw()` 延迟触发 `_draw()`，记录 draw 命令。" },
          { title: "帧末绘制", text: "RendererCanvas 在 RenderingServer draw 阶段处理。"}
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`CanvasItem` 声明在 `scene/main/canvas_item.h:47`，继承 Node。核心字段包括 `RID canvas_item`、`CanvasLayer *canvas_layer`、`modulate`、`self_modulate`、`z_index`、`visibility_layer`、`visible`、`pending_update`、`draw_commands_dirty`、`top_level`、`material`、texture filter/repeat 缓存，以及一个只保存 CanvasItem 子节点的 `data.canvas_item_children` 优化列表。"
      },
      {
        type: "paragraph",
        text: "构造函数在 `scene/main/canvas_item.cpp:1801` 调用 `RenderingServer::get_singleton()->canvas_item_create()`，析构函数在 `:1806` 调 `free_rid(canvas_item)`。这说明每个 CanvasItem 都对应 RenderingServer 里的一个 canvas item 对象，高层节点只是持有这个 RID 并提交状态。"
      },
      {
        type: "table",
        title: "CanvasItem 关键字段",
        headers: ["字段", "含义", "影响什么"],
        rows: [
          ["`canvas_item`", "RenderingServer 中的 canvas item RID。", "所有 2D draw 命令、可见性、变换和排序都提交到它。"],
          ["`canvas_layer`", "当前所在 CanvasLayer。", "决定画到哪个 canvas，以及排序基准。"],
          ["`visible` / `parent_visible_in_tree`", "自身可见性和父链可见性。", "决定 `is_visible_in_tree()` 和 Server 可见状态。"],
          ["`pending_update`", "是否已经排队重绘。", "避免重复 `queue_redraw()`。"],
          ["`draw_commands_dirty`", "已有 draw 命令是否需要清掉重录。", "_redraw_callback 中决定是否 `canvas_item_clear()`。"],
          ["`z_index` / `z_relative`", "2D 排序相关状态。", "提交给 Server 影响同层绘制顺序。"],
          ["`material` / `use_parent_material`", "CanvasItem 材质状态。", "影响 2D 绘制使用的材质和 shader 参数。"],
          ["`texture_filter_cache` / `texture_repeat_cache`", "继承后的纹理采样设置。", "影响 texture 绘制的 filter/repeat。"]
        ]
      },
      {
        type: "heading",
        title: "进入树：它画到哪个 canvas"
      },
      {
        type: "paragraph",
        text: "`CanvasItem::_notification()` 处理 `NOTIFICATION_ENTER_TREE` 时，会找到父 CanvasItem、CanvasLayer、Viewport 或 Window，设置 `parent_visible_in_tree`，维护父 CanvasItem 的 `canvas_item_children`，然后调用 `_enter_canvas()`。源码入口在 `canvas_item.cpp:306` 到 `:372`。"
      },
      {
        type: "paragraph",
        text: "`_enter_canvas()` 在 `canvas_item.cpp:243`。如果有父 CanvasItem，它调用 `canvas_item_set_parent(canvas_item, parent_item->get_canvas_item())`；如果没有父 CanvasItem，就查找 CanvasLayer 或 Viewport 的 World2D canvas，再把自己挂到那个 canvas RID 下。退出树时 `_exit_canvas()` 在 `:296`，会把 parent 设置为 `RID()` 并离开 canvas group。"
      },
      {
        type: "flow",
        title: "CanvasItem 进入 canvas 的路径",
        steps: [
          { title: "ENTER_TREE", text: "CanvasItem 收到 Node 生命周期通知。" },
          { title: "找父 CanvasItem", text: "有父 CanvasItem 时继承 canvas_layer 并挂到父 canvas item RID。" },
          { title: "找 CanvasLayer/Viewport", text: "没有父 CanvasItem 时，找到当前 CanvasLayer 或 Viewport 的 World2D canvas。" },
          { title: "set_parent", text: "调用 RenderingServer `canvas_item_set_parent()`。" },
          { title: "设置 visibility layer", text: "提交 visibility_layer 到 Server。" },
          { title: "queue_redraw", text: "进入 canvas 后请求重绘。" }
        ]
      },
      {
        type: "heading",
        title: "重绘：queue_redraw 和 _draw"
      },
      {
        type: "paragraph",
        text: "`queue_redraw()` 在 `canvas_item.cpp:481`。它可以从线程调用，但如果节点不在树内或已经 `pending_update`，就直接返回。否则设置 `pending_update = true`，并用 `call_deferred()` 延迟调用 `_redraw_callback()`。这避免一个属性变化连续触发多次立即重绘。"
      },
      {
        type: "paragraph",
        text: "`_redraw_callback()` 在 `canvas_item.cpp:142`。它先确认节点仍在树内；如果 `draw_commands_dirty` 为 true，就调用 `RenderingServer::canvas_item_clear()` 清掉旧命令。随后如果 `is_visible_in_tree()`，设置 `drawing = true`，发 `NOTIFICATION_DRAW`、发 `draw` 信号、调用脚本 `_draw()`，让 `draw_rect()`、`draw_texture()` 等命令写进 canvas item。"
      },
      {
        type: "table",
        title: "draw 相关机制",
        headers: ["机制", "源码入口", "作用"],
        rows: [
          ["`queue_redraw()`", "`canvas_item.cpp:481`", "请求稍后重绘，合并重复请求。"],
          ["`_redraw_callback()`", "`canvas_item.cpp:142`", "安全点清旧 draw 命令并触发绘制回调。"],
          ["`NOTIFICATION_DRAW`", "`canvas_item.h:215`", "通知 C++/脚本当前可以绘制。"],
          ["`_draw()`", "GDVIRTUAL_CALL `_draw`。", "用户重写的绘制入口。"],
          ["`draw_*` 方法", "`draw_rect()`、`draw_line()`、`draw_texture()` 等。", "最终调用 `canvas_item_add_*` 写入 RenderingServer。"],
          ["`drawing` 标记", "`ERR_FAIL_COND_MSG(!drawing, ...)`", "防止在 `_draw()` 之外直接调用绘制命令。"]
        ]
      },
      {
        type: "heading",
        title: "可见性、变换和排序"
      },
      {
        type: "paragraph",
        text: "`set_visible()` 在 `canvas_item.cpp:85`，可见性变化后 `_handle_visibility_change()` 会调用 `canvas_item_set_visible(canvas_item, p_visible)`，发 `NOTIFICATION_VISIBILITY_CHANGED`，可见时 `queue_redraw()`，不可见时发 `hidden` 信号，并把可见性传播给 CanvasItem 子节点。"
      },
      {
        type: "paragraph",
        text: "变换变化会通过 `_notify_transform()` 递归标脏，并把节点加入 SceneTree 的 `xform_change_list`，在合适时机把 `canvas_item_set_transform()` 提交给 RenderingServer。`set_canvas_item_use_identity_transform()` 在 `canvas_item.cpp:1116`，会告诉 Server 不要串联父变换，并在树内立即提交 identity 或当前 transform。"
      },
      {
        type: "paragraph",
        text: "排序方面，`set_z_index()` 在 `canvas_item.cpp:675` 调 `canvas_item_set_z_index()`；`update_draw_order()` 在 `:462` 会根据父子位置或 canvas group 更新 draw index。CanvasLayer、top_level、z_relative、y_sort_enabled 都会影响最终排序。"
      },
      {
        type: "table",
        title: "CanvasItem 提交给 RenderingServer 的状态",
        headers: ["状态", "典型 API", "什么时候改"],
        rows: [
          ["可见性", "`canvas_item_set_visible()`", "`show()`、`hide()`、父链可见性变化、进入树。"],
          ["父子/画布归属", "`canvas_item_set_parent()`", "进入/退出树，切换 CanvasLayer/Viewport。"],
          ["变换", "`canvas_item_set_transform()`", "Node2D/Control 变换变化，identity transform 切换。"],
          ["排序", "`canvas_item_set_z_index()`、`canvas_item_set_draw_index()`", "z_index、子节点顺序、CanvasLayer 排序变化。"],
          ["材质", "`canvas_item_set_material()`、`canvas_item_set_use_parent_material()`", "设置 CanvasItem material 或继承父材质。"],
          ["绘制命令", "`canvas_item_add_rect()`、`add_line()`、`add_polygon()` 等。", "_draw 阶段调用 draw_*。"],
          ["纹理采样", "`canvas_item_set_default_texture_filter/repeat()`", "texture_filter / texture_repeat 或父级继承变化。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：自定义 _draw"
      },
      {
        type: "code",
        code: [
          "extends Node2D",
          "",
          "var radius := 24.0:",
          "    set(value):",
          "        radius = value",
          "        queue_redraw()",
          "",
          "func _draw() -> void:",
          "    draw_circle(Vector2.ZERO, radius, Color.DEEP_SKY_BLUE)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`queue_redraw()` 不会马上画；它排队到 `_redraw_callback()`，再在 `_draw()` 里把 `draw_circle()` 转成 RenderingServer canvas item 命令。"
      },
      {
        type: "subheading",
        title: "案例二：错误地在 _process 里直接 draw"
      },
      {
        type: "code",
        code: [
          "func _process(_delta: float) -> void:",
          "    # 错误：绘制命令只允许在 _draw 或 draw 通知期间调用。",
          "    # draw_rect(Rect2(0, 0, 32, 32), Color.RED)",
          "    queue_redraw()"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "CanvasItem 源码开头的检查会阻止 `_draw()` 之外的绘制命令。要更新绘制内容，改状态后调用 `queue_redraw()`。"
      },
      {
        type: "subheading",
        title: "案例三：UI 和 2D 共用 CanvasItem"
      },
      {
        type: "code",
        code: [
          "# Control 也继承 CanvasItem，所以它也有 visible、modulate、material、",
          "# queue_redraw()、_draw() 和 canvas item RID。",
          "extends Control",
          "",
          "func _draw() -> void:",
          "    draw_rect(get_rect(), Color(0.1, 0.1, 0.1, 0.4))"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "GUI 不是完全独立的绘制体系。Control 的布局和输入是 GUI 层能力，但绘制入口仍来自 CanvasItem。"
      },
      {
        type: "subheading",
        title: "案例四：直接拿 canvas item RID"
      },
      {
        type: "code",
        code: [
          "func flash_low_level() -> void:",
          "    var rid := get_canvas_item()",
          "    RenderingServer.canvas_item_set_modulate(rid, Color(1, 0.5, 0.5))"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这种写法绕过了部分高级封装，适合理解层级。普通项目里通常优先使用节点属性和 `_draw()`。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "CanvasItem 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Node", "场景树、生命周期、父子关系。", "不提供 2D 绘制 RID。"],
          ["CanvasItem", "2D/GUI 渲染状态、canvas item RID、draw 命令。", "不做布局规则，也不做 3D instance。"],
          ["Node2D", "2D transform 和常用 2D 节点行为。", "绘制底层仍来自 CanvasItem。"],
          ["Control", "GUI 布局、主题、输入、焦点。", "绘制底层仍来自 CanvasItem。"],
          ["CanvasLayer", "把 CanvasItem 放到单独 canvas 层。", "不是每个节点的绘制命令存储。"],
          ["Viewport", "提供 canvas/world、渲染目标和输入入口。", "不替代 CanvasItem 的 draw 命令。"],
          ["RenderingServer", "保存 canvas item RID 和绘制命令。", "不处理 Node 业务生命周期。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：CanvasItem 只属于 Node2D。Control 也继承 CanvasItem，GUI 绘制同样走这套体系。",
          "误区二：draw_* 可以随时调用。绘制命令只允许在 `_draw()`、draw 信号或 NOTIFICATION_DRAW 期间调用。",
          "误区三：queue_redraw 会立即重绘。它用 deferred 合并请求，稍后触发 `_redraw_callback()`。",
          "误区四：CanvasItem 就是纹理。它是 2D 绘制节点基类，纹理是 Resource/RenderingServer texture RID。",
          "误区五：visible 只改脚本变量。CanvasItem 会把可见性提交给 RenderingServer 并传播到子 CanvasItem。",
          "误区六：z_index 是全局绝对排序。CanvasLayer、parent、z_relative、y_sort 和 draw index 都会影响结果。",
          "误区七：不在树内也能正常绘制。不在 SceneTree 内时没有有效 Viewport/canvas 挂接，queue_redraw 会直接返回。",
          "误区八：CanvasItem 问题要直接读 RendererRD。先看 CanvasItem 是否正确提交 RID、parent、visible、transform 和 draw 命令。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `canvas_item.h:47` 和字段，确认 CanvasItem 继承 Node 但额外持有 canvas item RID。",
          "读构造和析构，理解 `canvas_item_create()` / `free_rid()` 的生命周期。",
          "读 `_notification(NOTIFICATION_ENTER_TREE)`、`_enter_canvas()`、`_exit_canvas()`，看它如何挂到父 CanvasItem、CanvasLayer 或 Viewport canvas。",
          "读 `queue_redraw()` 和 `_redraw_callback()`，理解 draw 命令的缓存和重录。",
          "读几个 draw_* 方法，例如 `draw_rect()`、`draw_texture()`，确认它们最终调用 `canvas_item_add_*`。",
          "读 `set_visible()`、`set_z_index()`、`set_material()`、transform 提交，建立属性到 RenderingServer 的映射。",
          "最后再进入 RendererCanvas，理解后端如何排序和实际绘制 canvas items。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "CanvasItem 是 Godot 2D/GUI 的渲染入口：它把节点状态和 `_draw()` 命令变成 canvas item RID 上的 RenderingServer 状态，最终由 RendererCanvas 在帧末画出来。"
      }
    ]
  },
  {
    id: "visualinstance3d",
    title: "VisualInstance3D",
    aliases: ["VisualInstance3D", "GeometryInstance3D", "MeshInstance3D", "instance RID", "instance_create", "instance_set_base", "instance_set_scenario", "instance_set_transform", "World3D", "scenario"],
    summary: "Godot 3D 可见对象的基础节点：持有 RenderingServer instance RID，把 Mesh/Light/Decal 等 base、World3D scenario、transform、可见性和渲染层提交给 3D 渲染后端。",
    article: [
      {
        type: "lead",
        text: "VisualInstance3D 是 3D 世界里“可被渲染的实例”的基础类。它继承 Node3D，所以有 3D transform 和树生命周期；它额外持有一个 RenderingServer instance RID。MeshInstance3D、GeometryInstance3D 等类通过这个 instance RID，把 Mesh、材质、可见性、渲染层和 World3D scenario 提交给 3D 渲染后端。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Mesh 想成“模型文件”，把 VisualInstance3D 想成“把这个模型摆到世界里的一个摆放实例”。同一个 Mesh 可以被摆很多次，每一次都有自己的 transform、可见性、材质覆盖和渲染层。VisualInstance3D 管的是“摆在哪里、属不属于这个世界、用哪个 base”，不是 Mesh 顶点数据本身。"
      },
      {
        type: "paragraph",
        text: "MeshInstance3D 就是在 VisualInstance3D 的基础上，把 Mesh Resource 的 RID 设给 instance 的 base。于是 RenderingServer 知道：这里有一个 instance，它在某个 World3D scenario 中，使用这个 Mesh RID，使用这个 transform 和材质覆盖。"
      },
      {
        type: "flow",
        title: "3D 可见对象的直觉路径",
        steps: [
          { title: "Mesh Resource", text: "保存顶点、surface、材质引用，内部有 mesh RID。" },
          { title: "VisualInstance3D", text: "构造时创建 instance RID。" },
          { title: "set_base", text: "把 Mesh/Light/Decal 等 base RID 绑定到 instance。" },
          { title: "进入 World3D", text: "把 instance 放进当前 World3D 的 scenario。" },
          { title: "提交 transform", text: "节点移动时调用 `instance_set_transform()`。" },
          { title: "RendererScene", text: "后端做剔除、光照、阴影、材质和后处理。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`VisualInstance3D` 声明在 `scene/3d/visual_instance_3d.h:39`，继承 Node3D。核心字段只有几类：`RID base`、`RID instance`、`layers`、`sorting_offset`、`sorting_use_aabb_center`。这说明它不是 Mesh 数据容器，而是一个把某个 base 放进 3D 世界的渲染实例。"
      },
      {
        type: "paragraph",
        text: "构造函数在 `scene/3d/visual_instance_3d.cpp:208` 调用 `RenderingServer::instance_create()`，随后 `instance_attach_object_instance_id(instance, get_instance_id())` 把这个渲染实例和 ObjectID 关联起来，并 `set_notify_transform(true)` 让 Node3D transform 变化能通知它。析构函数在 `:215` 调用 `free_rid(instance)`。"
      },
      {
        type: "table",
        title: "VisualInstance3D 的关键状态",
        headers: ["字段/状态", "含义", "提交到哪里"],
        rows: [
          ["`instance`", "RenderingServer 中的 3D instance RID。", "`instance_create()` / `free_rid()`。"],
          ["`base`", "instance 要显示的底层资源 RID，常见是 Mesh RID。", "`instance_set_base()`。"],
          ["`layers`", "3D render layer mask。", "`instance_set_layer_mask()`。"],
          ["World3D scenario", "当前 3D 世界的渲染场景容器。", "`instance_set_scenario()`。"],
          ["global transform", "Node3D 的世界变换。", "`instance_set_transform()`。"],
          ["visibility", "节点树可见性结果。", "`instance_set_visible()`。"],
          ["sorting data", "透明排序相关 pivot 信息。", "`instance_set_pivot_data()`。"]
        ]
      },
      {
        type: "heading",
        title: "进入 World3D：scenario 才是 3D 世界归属"
      },
      {
        type: "paragraph",
        text: "`VisualInstance3D::_notification()` 处理 `NOTIFICATION_ENTER_WORLD` 时，在 `visual_instance_3d.cpp:93` 调 `instance_set_scenario(instance, get_world_3d()->get_scenario())`，然后更新可见性。退出世界时在 `:115` 调 `instance_set_scenario(instance, RID())`，并清掉 skeleton、标记不可见。"
      },
      {
        type: "paragraph",
        text: "这和 2D CanvasItem 挂到 canvas 很像：3D instance 需要进入某个 World3D 的 scenario，RendererScene 才知道它属于哪个 3D 世界、该由哪个 Viewport 渲染。只创建 VisualInstance3D 或只设置 Mesh，还不等于它已经出现在某个 3D 渲染场景中。"
      },
      {
        type: "flow",
        title: "VisualInstance3D 进入 3D 世界",
        steps: [
          { title: "Node3D 进入树", text: "生命周期进入 3D world。" },
          { title: "ENTER_WORLD", text: "VisualInstance3D 收到 world 通知。" },
          { title: "设置 scenario", text: "把 instance RID 放进 `get_world_3d()->get_scenario()`。" },
          { title: "更新可见性", text: "根据 `is_visible_in_tree()` 设置 Server visible。" },
          { title: "提交 transform", text: "可见时确保 instance transform 是最新的。" },
          { title: "EXIT_WORLD", text: "离开时 scenario 设为 RID()，后端不再把它当当前世界实例。"}
        ]
      },
      {
        type: "heading",
        title: "transform、可见性和层"
      },
      {
        type: "paragraph",
        text: "`_update_visibility()` 会比较旧可见状态和 `is_visible_in_tree()`。如果从不可见变可见，并且不用 identity transform，它会先把 `get_global_transform()` 提交给 `instance_set_transform()`，再调用 `instance_set_visible(instance, visible)`。这样重新显示时不会用旧 transform。"
      },
      {
        type: "paragraph",
        text: "`NOTIFICATION_TRANSFORM_CHANGED` 时，如果 instance 可见、没有开启 physics interpolation 的特殊路径，并且没有使用 identity transform，就调用 `instance_set_transform(instance, get_global_transform())`。固定时间步插值路径则走 `fti_update_servers_xform()`，把插值后的 transform 提交给 Server。"
      },
      {
        type: "paragraph",
        text: "`set_layer_mask()` 在 `visual_instance_3d.cpp:130`，保存 `layers` 并调用 `instance_set_layer_mask(instance, p_mask)`。Camera 的 cull mask、Viewport 和 RendererScene 会用这些 layer 判断哪些实例可见。`set_sorting_offset()` 和 `set_sorting_use_aabb_center()` 则通过 `instance_set_pivot_data()` 影响透明排序。"
      },
      {
        type: "table",
        title: "VisualInstance3D 事件到 Server 调用",
        headers: ["事件/方法", "源码入口", "RenderingServer 调用"],
        rows: [
          ["构造", "`visual_instance_3d.cpp:208`", "`instance_create()`、`instance_attach_object_instance_id()`。"],
          ["析构", "`visual_instance_3d.cpp:215`", "`free_rid(instance)`。"],
          ["进入 World3D", "`NOTIFICATION_ENTER_WORLD`", "`instance_set_scenario(instance, world.scenario)`。"],
          ["退出 World3D", "`NOTIFICATION_EXIT_WORLD`", "`instance_set_scenario(instance, RID())`。"],
          ["变换变化", "`NOTIFICATION_TRANSFORM_CHANGED`", "`instance_set_transform(instance, get_global_transform())`。"],
          ["可见性变化", "`NOTIFICATION_VISIBILITY_CHANGED`", "`instance_set_visible(instance, visible)`。"],
          ["设置层", "`set_layer_mask()`", "`instance_set_layer_mask(instance, mask)`。"],
          ["设置 base", "`set_base()`", "`instance_set_base(instance, base)`。"]
        ]
      },
      {
        type: "heading",
        title: "GeometryInstance3D 和 MeshInstance3D"
      },
      {
        type: "paragraph",
        text: "`GeometryInstance3D` 继承 VisualInstance3D，增加材质覆盖、材质叠加、投影设置、GI mode、visibility range、transparency、LOD bias、custom AABB、instance shader 参数等。比如 `set_material_override()` 在 `visual_instance_3d.cpp:218` 后调用 `instance_geometry_set_material_override(get_instance(), material_rid)`。"
      },
      {
        type: "paragraph",
        text: "`MeshInstance3D::set_mesh()` 在 `scene/3d/mesh_instance_3d.cpp:120`。如果 Mesh 有效，它先 `set_base(mesh->get_rid())`，再连接 Mesh changed 信号并刷新；如果 Mesh 为空，就 `set_base(RID())`。表面材质覆盖在 `mesh_instance_3d.cpp:381` 通过 `instance_set_surface_override_material()` 提交给 RenderingServer。"
      },
      {
        type: "table",
        title: "3D 可见对象的继承分工",
        headers: ["类", "新增职责", "典型 Server 调用"],
        rows: [
          ["`Node3D`", "3D transform、父子空间、world 通知。", "本身不创建渲染 instance。"],
          ["`VisualInstance3D`", "instance RID、base RID、scenario、layer、可见性。", "`instance_create()`、`instance_set_base()`、`instance_set_transform()`。"],
          ["`GeometryInstance3D`", "材质覆盖、阴影、GI、透明、LOD、visibility range、custom AABB。", "`instance_geometry_set_*()`。"],
          ["`MeshInstance3D`", "绑定 Mesh Resource、surface override material、blend shape。", "`set_base(mesh->get_rid())`、`instance_set_surface_override_material()`。"],
          ["`Light3D` / `Decal` 等", "也可用 instance/base 模型表达 3D 可见或影响对象。", "设置对应 light/decal base RID。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：MeshInstance3D 设置 Mesh"
      },
      {
        type: "code",
        code: [
          "var mesh_instance := MeshInstance3D.new()",
          "mesh_instance.mesh = preload(\"res://models/crate.tres\")",
          "add_child(mesh_instance)",
          "",
          "# 内部心智模型：",
          "# VisualInstance3D 有 instance RID，Mesh Resource 有 mesh RID。",
          "# set_mesh() 会把 mesh.get_rid() 设置为 instance 的 base。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Mesh 数据和摆放实例分开，是 3D 渲染性能和复用的基础：一个 Mesh RID 可以被多个 instance 使用。"
      },
      {
        type: "subheading",
        title: "案例二：直接操作 instance RID"
      },
      {
        type: "code",
        code: [
          "func set_render_layer(layer: int, enabled: bool) -> void:",
          "    set_layer_mask_value(layer, enabled)",
          "    # 最终调用 RenderingServer.instance_set_layer_mask(instance, mask)。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "layer mask 决定 Camera/Viewport 的可见性过滤。它不是物理 collision layer，也不是 Node group。"
      },
      {
        type: "subheading",
        title: "案例三：材质覆盖不是改 Mesh 资源"
      },
      {
        type: "code",
        code: [
          "func highlight(material: Material) -> void:",
          "    material_override = material",
          "    # 这会给当前 instance 设置 override，",
          "    # 不会修改 Mesh Resource 本身。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "多个 MeshInstance3D 共用同一个 Mesh 时，instance 的 material_override 只影响当前实例；Mesh surface 自带材质仍属于 Mesh Resource 层。"
      },
      {
        type: "subheading",
        title: "案例四：节点在树里但 3D 不显示"
      },
      {
        type: "code",
        code: [
          "# 排查思路：",
          "# 1. MeshInstance3D.mesh 是否为空？",
          "# 2. instance base 是否是有效 mesh RID？",
          "# 3. 节点是否进入 World3D scenario？",
          "# 4. visible/layers/camera cull mask 是否匹配？",
          "# 5. transform/AABB 是否被剔除到相机外？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "3D 不显示常常不是单点问题。VisualInstance3D 帮你把几条关键线索串起来：base、scenario、transform、visible、layers。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "VisualInstance3D 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Node3D", "3D transform、空间层级、world 通知。", "不创建 RenderingServer instance。"],
          ["VisualInstance3D", "instance RID、base、scenario、visible、layers。", "不保存 Mesh 顶点数据。"],
          ["GeometryInstance3D", "几何渲染通用参数，如材质覆盖、阴影、GI、LOD。", "不指定具体 Mesh 数据格式。"],
          ["MeshInstance3D", "把 Mesh Resource 接到 VisualInstance3D instance 上。", "不拥有 RendererScene 后端对象本体。"],
          ["Mesh Resource", "surface arrays、材质引用、mesh RID。", "不表示某个世界里的摆放位置。"],
          ["World3D scenario", "3D 世界里的渲染容器 RID。", "不是 Node parent，也不是 PackedScene owner。"],
          ["RenderingServer", "保存 instance/base/scenario/transform 状态。", "不执行 Node3D 生命周期。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：VisualInstance3D 保存 Mesh 数据。不是；它保存 instance RID，Mesh 数据在 Mesh Resource 和 RenderingServer mesh RID 后面。",
          "误区二：Node3D 进入树就一定可渲染。只有 VisualInstance3D 这类可见实例会创建渲染 instance。",
          "误区三：设置 mesh 就等于进入渲染世界。还要进入 World3D scenario，并满足 visible/layers/camera 条件。",
          "误区四：material_override 会修改 Mesh 资源。它是 instance 级覆盖，不改共享 Mesh 本体。",
          "误区五：render layers 等同 physics layers。两者是不同系统，分别用于相机可见性和物理碰撞过滤。",
          "误区六：instance RID 可以当 Mesh RID 用。instance 是摆放实例，mesh 是几何资源，两者通过 `instance_set_base()` 连接。",
          "误区七：transform 每次都直接提交。physics interpolation 或 identity transform 会改变提交路径。",
          "误区八：3D 不显示就直接看 shader。先查 base、scenario、visible、layers、transform、AABB 和 camera。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `visual_instance_3d.h:39`，确认字段只有 base、instance、layers 和排序相关状态。",
          "读构造/析构，理解 instance RID 的创建、ObjectID 绑定和释放。",
          "读 `_notification()` 的 ENTER_WORLD、TRANSFORM_CHANGED、EXIT_WORLD、VISIBILITY_CHANGED 分支。",
          "读 `set_base()`、`set_layer_mask()`、sorting 方法，建立属性到 RenderingServer 的映射。",
          "读 `GeometryInstance3D` 的材质覆盖、阴影、GI、visibility range、LOD 相关方法。",
          "读 `MeshInstance3D::set_mesh()` 和 surface override material，理解 Mesh Resource 如何接到 instance。",
          "最后进入 RendererScene，理解 scenario、culling、light、shadow 和 draw pass 如何消费这些 instance。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "VisualInstance3D 是 3D 渲染实例的桥：它把 Node3D 的世界位置和 Mesh/材质等 base RID 接到 World3D scenario，让 RenderingServer 能在 3D 后端里渲染它。"
      }
    ]
  },
  {
    id: "rendererrd",
    title: "RendererRD",
    aliases: ["RendererRD", "RendererCompositorRD", "RendererSceneRenderRD", "RendererCanvasRenderRD", "RendererViewport", "RendererSceneCull", "RendererCanvasCull", "RD renderer", "RenderSceneBuffersRD", "renderer_rd"],
    summary: "Godot 的 RD 渲染后端组织层：接收 RenderingServerDefault 分发的状态，按 viewport 组织 2D/3D 渲染，再通过 RenderingDevice 提交底层图形命令。",
    article: [
      {
        type: "lead",
        text: "RendererRD 不是脚本侧可直接调用的 API，而是 Godot 渲染后端的一组内部实现。RenderingServerDefault 接收场景层提交的 RID 和状态后，会把工作交给 renderer_rd 下的 storage、canvas、scene、viewport、compositor 等模块；这些模块再使用 RenderingDevice 把渲染变成底层图形资源和命令。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 RenderingServer 是订单台，RendererRD 就是后厨的生产线。订单台收到“画这个 2D item”“渲染这个 3D instance”“这个 Viewport 要输出到窗口”之后，RendererRD 负责安排先做谁、怎么排序、哪些对象可见、用什么光照和材质、最后把图像贴到屏幕上。"
      },
      {
        type: "paragraph",
        text: "它和 RenderingDevice 的区别是：RendererRD 仍然懂 Godot 的渲染概念，比如 Viewport、canvas、scenario、light、shadow、environment、render target；RenderingDevice 更底层，关心 texture、buffer、shader、pipeline、draw list、swap buffers。"
      },
      {
        type: "flow",
        title: "RendererRD 在渲染链路中的位置",
        steps: [
          { title: "Scene nodes", text: "CanvasItem、VisualInstance3D、Viewport 提交状态。" },
          { title: "RenderingServerDefault", text: "统一 API 分发到 RSG storage/canvas/scene。" },
          { title: "RendererRD storage", text: "TextureStorage、MeshStorage、MaterialStorage 等保存后端状态。" },
          { title: "RendererViewport", text: "按 Viewport 决定渲染 3D、2D、render target 和屏幕输出。" },
          { title: "RendererCanvas/Scene", text: "2D canvas 绘制或 3D scene culling/render。" },
          { title: "RenderingDevice", text: "提交 GPU 资源、pipeline、draw list 和 swap buffers。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "Godot 的 renderer_rd 目录在 `servers/rendering/renderer_rd/`，里面有 `renderer_compositor_rd.*`、`renderer_canvas_render_rd.*`、`renderer_scene_render_rd.*`、`storage_rd/`、`effects/`、`forward_clustered/`、`forward_mobile/`、shader 和 pipeline 缓存等。它是基于 RenderingDevice 的现代渲染后端组织层。"
      },
      {
        type: "paragraph",
        text: "`RendererCompositor` 是渲染后端抽象，声明在 `servers/rendering/renderer_compositor.h`。它暴露 `get_canvas()`、`get_scene()`、`get_texture_storage()`、`get_mesh_storage()`、`begin_frame()`、`blit_render_targets_to_screen()`、`end_frame()` 等接口。`RendererCompositor::create()` 在 `renderer_compositor.cpp:43` 调用 `_create_func()` 创建具体后端。"
      },
      {
        type: "paragraph",
        text: "`RendererCompositorRD` 在 `servers/rendering/renderer_rd/renderer_compositor_rd.h` 中聚合了 `RendererCanvasRenderRD`、`RendererSceneRenderRD`、Fog、LightStorage、MaterialStorage、MeshStorage、ParticlesStorage、TextureStorage、Utilities、UniformSetCacheRD 和 FramebufferCacheRD。它不是一个单独巨型 renderer，而是 RD 后端的总装配器。"
      },
      {
        type: "table",
        title: "RendererRD 主要部件",
        headers: ["部件", "源码入口", "职责"],
        rows: [
          ["`RendererCompositorRD`", "`renderer_compositor_rd.h`", "聚合 RD 后端的 canvas、scene、storage、cache、effects，并负责 begin/end frame。"],
          ["`RendererViewport`", "`servers/rendering/renderer_viewport.cpp`", "按 Viewport 组织 render target、3D scene、2D canvas、blit 到屏幕。"],
          ["`RendererCanvasCull` / `RendererCanvasRenderRD`", "`renderer_canvas_cull.cpp`、`renderer_canvas_render_rd.cpp`", "2D canvas item 的收集、排序、裁剪、实际 RD 绘制。"],
          ["`RendererSceneCull` / `RendererSceneRenderRD`", "`renderer_scene_cull.cpp`、`renderer_scene_render_rd.*`", "3D scenario 的可见性、光照/阴影准备、RD 渲染 pass。"],
          ["`storage_rd`", "`texture_storage`、`mesh_storage`、`material_storage` 等。", "保存 RenderingServer RID 对应的 RD 后端状态。"],
          ["`RenderingDevice`", "`servers/rendering/rendering_device.*`", "底层图形 API 抽象：texture、buffer、shader、pipeline、draw/compute list。"]
        ]
      },
      {
        type: "heading",
        title: "Viewport 是后端渲染调度入口"
      },
      {
        type: "paragraph",
        text: "`RendererViewport` 是理解 RendererRD 的好入口。`renderer_viewport.cpp:332` 调 `RSG::scene->render_camera()` 渲染 3D；`:714` 调 `RSG::canvas->render_canvas()` 渲染 2D canvas；如果要输出到窗口，后面会把 render target blit 到屏幕。Viewport 因此把 3D、2D、render target、window output 串起来。"
      },
      {
        type: "paragraph",
        text: "这也解释了为什么渲染问题常常离不开 Viewport：Viewport 持有 render target、camera、scenario、canvas、shadow atlas、MSAA/HDR/TAA/FSR/VRS 等设置。节点只提交局部状态，Viewport 决定这些状态要画到哪里。"
      },
      {
        type: "flow",
        title: "一个 Viewport 的后端渲染顺序",
        steps: [
          { title: "准备 render target", text: "TextureStorage 管 Viewport 的 render target 和 clear/resolve 状态。" },
          { title: "渲染 3D", text: "有 camera 和 scenario 时调用 `RSG::scene->render_camera()`。" },
          { title: "渲染 2D", text: "遍历 canvas，调用 `RSG::canvas->render_canvas()`。" },
          { title: "处理 render info/effects", text: "更新统计、SDF、MSAA resolve、后处理等。" },
          { title: "blit", text: "需要显示到窗口时，把 render target blit 到 screen。" },
          { title: "present", text: "RendererCompositorRD end_frame 通过 RD swap_buffers。" }
        ]
      },
      {
        type: "heading",
        title: "2D：CanvasCull 到 CanvasRenderRD"
      },
      {
        type: "paragraph",
        text: "2D 后端分两层看。`RendererCanvasCull::render_canvas()` 在 `renderer_canvas_cull.cpp:495`，负责遍历 canvas item、排序、裁剪、灯光和可见性；真正把 item list 交给 RD 绘制的是 `RendererCanvasRenderRD::canvas_render_items()`，源码在 `renderer_canvas_render_rd.cpp:509`。"
      },
      {
        type: "paragraph",
        text: "这对应高层 CanvasItem 的心智模型：CanvasItem 只是把 draw 命令和状态放到 canvas item RID 上；RendererCanvasCull/RenderRD 才在 Viewport 渲染时把这些 item 变成批次、材质、纹理绑定和 draw list。"
      },
      {
        type: "heading",
        title: "3D：SceneCull 到 SceneRenderRD"
      },
      {
        type: "paragraph",
        text: "3D 后端也分两层。`RendererSceneCull::render_camera()` 在 `renderer_scene_cull.cpp:2666`，根据 camera、scenario、visible layers、shadow atlas 等收集可见实例；它在 `:2779` 进入 `_render_scene()`。RD 层的 `RendererSceneRenderRD` 声明在 `renderer_scene_render_rd.h:57`，内部持有 tone mapper、FSR、VRS、resolve、copy/debug effects 等，并定义 `_render_scene()`、render buffers、material/uv2/sdfgi 等底层 pass 接口。"
      },
      {
        type: "paragraph",
        text: "所以 3D 不显示、阴影错误、GI 异常、后处理问题不应直接混在一个文件里查。先在 SceneCull 看实例是否被收集和剔除，再到 SceneRenderRD/forward_clustered/forward_mobile 看 pass 和 shader 变体。"
      },
      {
        type: "table",
        title: "2D 和 3D 后端路径对照",
        headers: ["路径", "前端提交", "中间组织", "RD 执行"],
        rows: [
          ["2D / GUI", "CanvasItem 的 canvas item RID 和 draw 命令。", "RendererCanvasCull 收集、排序、裁剪、灯光。", "RendererCanvasRenderRD 绑定纹理/材质并绘制 item list。"],
          ["3D Mesh", "VisualInstance3D 的 instance/base/scenario/transform。", "RendererSceneCull 做可见性、layer、AABB、light/shadow 准备。", "RendererSceneRenderRD 和 forward renderer 执行 pass。"],
          ["Viewport 输出", "Viewport RID、render target、camera/canvas 设置。", "RendererViewport 决定先画 3D、再画 2D、是否 blit。", "RendererCompositorRD / RenderingDevice swap buffers。"],
          ["资源", "Texture/Mesh/Material RID。", "storage_rd 保存后端对象和依赖。", "RenderingDevice 管 texture、buffer、pipeline、uniform set。"]
        ]
      },
      {
        type: "heading",
        title: "RenderingDevice 的边界"
      },
      {
        type: "paragraph",
        text: "`RendererCompositorRD::blit_render_targets_to_screen()` 展示了 RD 边界：它先 `RD::get_singleton()->screen_prepare_for_drawing()`，再创建 draw list，绑定 render pipeline、index array、uniform set，设置 push constant，draw，最后 `draw_list_end()`。这些是底层图形命令抽象，而不是 Godot 场景节点语义。"
      },
      {
        type: "paragraph",
        text: "`RendererCompositorRD::begin_frame()` 在 `renderer_compositor_rd.cpp:123` 更新 frame、delta、time，并把时间传给 canvas/scene；`end_frame()` 在 `:135` 调 `RD::get_singleton()->swap_buffers(p_present)`。这说明 RendererRD 的帧边界最终落到 RenderingDevice。"
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：2D 节点不显示怎么追"
      },
      {
        type: "code",
        code: [
          "# 阅读路线：",
          "# 1. CanvasItem 是否在树内，并挂到正确 Viewport/CanvasLayer？",
          "# 2. canvas item RID 是否 visible，draw 命令是否被 _draw 记录？",
          "# 3. RendererCanvasCull 是否把 item 放进渲染列表？",
          "# 4. RendererCanvasRenderRD 是否绑定了正确纹理/材质？",
          "# 5. Viewport render target 是否最终 blit 到屏幕？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "RendererRD 不是第一站。先确认 CanvasItem 提交正确，再进入 CanvasCull/CanvasRenderRD。"
      },
      {
        type: "subheading",
        title: "案例二：3D Mesh 不显示怎么追"
      },
      {
        type: "code",
        code: [
          "# 阅读路线：",
          "# 1. MeshInstance3D 是否有有效 Mesh RID？",
          "# 2. VisualInstance3D 是否进入 World3D scenario？",
          "# 3. layer/camera cull mask/AABB 是否导致被剔除？",
          "# 4. RendererSceneCull 是否收集到该 instance？",
          "# 5. RendererSceneRenderRD/forward renderer 的 pass 是否执行？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "大多数 3D 可见性问题先卡在 instance/scenario/culling，而不是 shader 编译。"
      },
      {
        type: "subheading",
        title: "案例三：区分渲染后端和 RD 底层"
      },
      {
        type: "code",
        code: [
          "# RendererRD 仍然知道 Godot 概念：Viewport、Scenario、Canvas、Light。",
          "# RenderingDevice 更底层：Texture、Buffer、Pipeline、UniformSet、DrawList。",
          "# 读代码时先判断变量属于哪一层。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "看到 `RID` 不够，还要看它是谁的 RID：render target RID、RD texture RID、mesh RID、instance RID 的层级不同。"
      },
      {
        type: "subheading",
        title: "案例四：窗口 present 问题不要只看 RendererScene"
      },
      {
        type: "code",
        code: [
          "# 如果离屏 Viewport texture 正常，但窗口不显示：",
          "# 1. RendererViewport 是否生成了 blit 到 screen 的任务？",
          "# 2. RendererCompositorRD::blit_render_targets_to_screen 是否执行？",
          "# 3. RenderingDevice screen/swap_buffers 是否成功？",
          "# 4. DisplayServer 的窗口/surface 是否有效？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "RendererScene 负责场景内容，窗口输出还要经过 RendererViewport、RendererCompositorRD、RenderingDevice 和 DisplayServer。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "RendererRD 和周边层级",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["RenderingServer", "统一渲染 API 和 RID 状态提交。", "不直接写具体 RD pass。"],
          ["RendererRD", "RD 后端组织：storage、viewport、canvas、scene、effects。", "不是脚本 API，也不是单一类。"],
          ["RendererViewport", "按 Viewport 调度 3D、2D、render target 和 blit。", "不保存 Node 生命周期。"],
          ["RendererCanvas", "2D item 的收集、排序、绘制。", "不处理 3D culling/light。"],
          ["RendererScene", "3D scenario 可见性、光照、阴影、后处理。", "不处理 GUI 布局。"],
          ["RenderingDevice", "底层图形资源和命令。", "不理解 Node/Resource 的用户语义。"],
          ["DisplayServer", "窗口 surface 和平台 present。", "不管理渲染场景内容。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：RendererRD 是一个可直接调用的脚本类。不是；它是内部 RD 后端模块集合。",
          "误区二：RenderingServer 和 RendererRD 是同一层。RenderingServer 是 API，RendererRD 是默认后端实现的一部分。",
          "误区三：2D 和 3D 都从同一个函数画出来。2D 走 CanvasCull/CanvasRenderRD，3D 走 SceneCull/SceneRenderRD。",
          "误区四：RenderingDevice 能解释所有渲染 bug。它只知道底层图形资源和命令，不知道场景语义。",
          "误区五：Viewport 只是节点容器。渲染后端里 Viewport 是 render target、camera、canvas、scenario 和屏幕输出的调度单元。",
          "误区六：看见 RD texture RID 就等于 RenderingServer texture RID。不同层 RID 语义不同，要看 owner/storage。",
          "误区七：3D 不显示就先查 shader。先查 instance 是否进入 scenario、是否被 culling/layer/camera 过滤。",
          "误区八：窗口不刷新一定是 draw pass 错。也可能是 render target、blit、swap buffers 或 DisplayServer 窗口问题。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `renderer_compositor.h`，理解 compositor 抽象要提供 canvas、scene、storage 和 frame 边界。",
          "读 `renderer_compositor_rd.h`，把 RD 后端有哪些 storage/effects/cache 画出来。",
          "读 `renderer_viewport.cpp`，看一个 Viewport 如何调用 `render_camera()`、`render_canvas()` 和 blit。",
          "2D 问题读 `renderer_canvas_cull.cpp` 再读 `renderer_canvas_render_rd.cpp`。",
          "3D 问题读 `renderer_scene_cull.cpp` 再读 `renderer_scene_render_rd.*` 和 forward renderer。",
          "资源问题读 `storage_rd` 下的 texture、mesh、material storage。",
          "底层 GPU 问题最后进入 `RenderingDevice`、pipeline/uniform/framebuffer cache 和平台 DisplayServer。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "RendererRD 是 Godot 基于 RenderingDevice 的渲染后端组织层：它把 RenderingServer 的 RID 状态按 Viewport、Canvas、Scene 和 Storage 分工，最终提交到底层图形 API。"
      }
    ]
  },
  {
    id: "renderingdevice",
    title: "RenderingDevice",
    aliases: ["RenderingDevice", "RD", "RenderingDeviceDriver", "RenderingDeviceGraph", "draw_list", "compute_list", "uniform_set", "render_pipeline", "compute_pipeline", "swap_buffers", "create_local_rendering_device", "create_local_device"],
    summary: "Godot 的底层图形 API 抽象：管理 texture、buffer、shader、pipeline、uniform set 和 draw/compute/raytracing list，并通过具体 driver 提交到 GPU 或 swapchain。",
    article: [
      {
        type: "lead",
        text: "RenderingDevice 是 Godot 渲染链路里最接近 GPU、但仍然保持跨平台抽象的一层。RendererRD 不直接写 Vulkan、Metal 或 D3D12 命令，而是调用 RenderingDevice 创建 texture、buffer、shader、pipeline、uniform set、framebuffer，再录制 draw list、compute list 或 raytracing list；RenderingDevice 再把这些抽象命令交给 RenderingDeviceDriver 的具体后端实现。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 RenderingServer 是“引擎的渲染订单台”，RendererRD 是“后厨调度”，RenderingDevice 就是“后厨里真正操作炉灶和工具的标准操作手册”。它不关心 Sprite2D 为什么要画，也不关心 MeshInstance3D 在场景树里叫什么；它只关心：用哪张纹理、哪个 buffer、哪个 shader、哪个 pipeline，往哪个 framebuffer 画，最后要不要把画面提交到窗口。"
      },
      {
        type: "paragraph",
        text: "它仍然不是某个显卡 API 本身。Godot 用 RenderingDevice 把 Vulkan、Metal、D3D12 等差异包起来，让上层后端可以写一套“RD 语言”。具体到平台时，RenderingDeviceDriver 才把这些动作翻译成真实驱动调用。"
      },
      {
        type: "flow",
        title: "RenderingDevice 在渲染链路中的位置",
        steps: [
          { title: "RenderingServer", text: "场景层提交 texture、mesh、viewport、canvas item、instance 等 RID 状态。" },
          { title: "RendererRD", text: "按 viewport、canvas、scene、storage 组织渲染工作。" },
          { title: "RenderingDevice", text: "创建 RD 资源，录制 draw/compute/raytracing list，管理帧和同步。" },
          { title: "RenderingDeviceDriver", text: "把 RD 命令映射到 Vulkan、Metal、D3D12 等后端。" },
          { title: "Display / GPU", text: "执行命令，处理 swapchain、present 和 GPU 资源。"}
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`RenderingDevice` 声明在 `servers/rendering/rendering_device.h:67`，继承 `Object`，内部有 `_THREAD_SAFE_CLASS_`、singleton、`RenderingContextDriver *context`、`RenderingDeviceDriver *driver` 和当前 device。它既暴露给脚本/扩展使用，也被 RendererRD 大量调用。"
      },
      {
        type: "paragraph",
        text: "它的接口可以分成四类：资源创建与释放、shader/uniform/pipeline 绑定、命令列表录制、帧提交与屏幕 swapchain。`RenderingDeviceDriver` 声明在 `servers/rendering/rendering_device_driver.h:90`，定义 BufferID、TextureID、CommandBufferID、SwapChainID、PipelineID 等低层 ID，并要求后端实现 `texture_create()`、`command_render_draw()`、`command_compute_dispatch()`、`command_queue_execute_and_present()` 等操作。"
      },
      {
        type: "table",
        title: "RenderingDevice 管什么",
        headers: ["类别", "典型 API", "含义"],
        rows: [
          ["Texture", "`texture_create()`、`texture_update()`、`texture_get_data_async()`", "GPU 或共享纹理资源，包含格式、尺寸、usage、mipmap、layer 等信息。"],
          ["Buffer", "`vertex_buffer_create()`、`storage_buffer_create()`、`buffer_update()`", "顶点、索引、uniform、storage、间接绘制等 GPU buffer。"],
          ["Shader", "`shader_create_from_bytecode_with_samplers()`", "后端可用的 shader bytecode 和反射信息。"],
          ["UniformSet", "`uniform_set_create()`", "把 texture、sampler、buffer 等资源按 shader set/binding 打包，供 pipeline 使用。"],
          ["Pipeline", "`render_pipeline_create()`、`compute_pipeline_create()`", "shader 加上 raster/depth/blend/vertex format 等固定状态，形成可绑定的执行配置。"],
          ["Draw/Compute List", "`draw_list_begin()`、`draw_list_draw()`、`compute_list_dispatch()`", "录制一段绘制或计算命令。"],
          ["Frame / Screen", "`screen_prepare_for_drawing()`、`draw_list_begin_for_screen()`、`swap_buffers()`", "管理窗口 swapchain、framebuffer 获取、执行命令和 present。"]
        ]
      },
      {
        type: "heading",
        title: "RD RID 和普通渲染 RID 的区别"
      },
      {
        type: "paragraph",
        text: "RenderingServer 也用 RID，RenderingDevice 也用 RID，但它们不是同一层。RenderingServer 的 texture RID、mesh RID、instance RID 是 Godot 渲染语义；RenderingDevice 的 texture RID、buffer RID、pipeline RID 是底层图形资源。看到 RID 时必须先确认它属于哪个 owner/storage。"
      },
      {
        type: "table",
        title: "三层 RID 对照",
        headers: ["层级", "例子", "读源码时问的问题"],
        rows: [
          ["场景/资源层", "`ImageTexture`、`ArrayMesh`、`ShaderMaterial`", "这个对象如何序列化、缓存、被节点引用？"],
          ["RenderingServer 层", "`texture_2d_create()`、`mesh_create()`、`instance_create()`", "这个资源或实例如何进入渲染 Server 的状态表？"],
          ["RenderingDevice 层", "`texture_create()`、`vertex_buffer_create()`、`render_pipeline_create()`", "GPU 资源、pipeline 和命令列表是否有效？"]
        ]
      },
      {
        type: "heading",
        title: "命令列表不是立即执行"
      },
      {
        type: "paragraph",
        text: "RenderingDevice 的 draw/compute 调用更像“录制命令”，不是每行立刻让 GPU 执行。`draw_list_begin()` 会打开一个绘制列表，随后绑定 pipeline、uniform set、vertex/index buffer，最后 `draw_list_draw()` 和 `draw_list_end()`。Compute 也是类似：`compute_list_begin()`、绑定 compute pipeline、绑定 uniform set、设置 push constant、dispatch、end。"
      },
      {
        type: "paragraph",
        text: "源码里有一个关键提醒：`buffer_update()` 的注释说明 Godot 的 render graph 可能重排 copy、draw、compute 命令。Godot 内部通过 `RenderingDeviceGraph` 记录资源 usage、自动插入 barrier、合并和重排命令；所以公共 API 会阻止用户在不安全的时机更新 buffer，避免“以为画的是旧数据，实际因为重排变成新数据”的问题。"
      },
      {
        type: "flow",
        title: "一次 draw list 的生命周期",
        steps: [
          { title: "创建资源", text: "准备 texture、buffer、shader、uniform set、pipeline。" },
          { title: "begin draw list", text: "选择 framebuffer 或 screen，指定 clear、viewport、scissor。" },
          { title: "绑定状态", text: "绑定 render pipeline、uniform set、vertex array、index array、push constant。" },
          { title: "记录 draw", text: "调用 draw、draw indexed 或 indirect draw。" },
          { title: "end list", text: "关闭列表，命令进入 RenderingDeviceGraph。" },
          { title: "execute / present", text: "主设备在 `swap_buffers()` 执行，local device 用 `submit()` 和 `sync()`。" }
        ]
      },
      {
        type: "heading",
        title: "主设备和本地设备"
      },
      {
        type: "paragraph",
        text: "`RenderingServer::get_rendering_device()` 在 `rendering_server.cpp:1884` 返回全局 singleton。`RenderingServer::create_local_rendering_device()` 在 `:1889` 调用 singleton 的 `create_local_device()`，后者在 `rendering_device.cpp:8987` 新建一个 RenderingDevice 并用同一个 context 初始化。"
      },
      {
        type: "paragraph",
        text: "主设备通常绑定窗口和 swapchain，帧尾通过 `swap_buffers()` 走 `_end_frame()`、`_execute_frame()`、frame 轮转和 `_begin_frame()`。本地设备更适合离屏计算、导入烘焙、工具任务；源码里 `submit()` 和 `sync()` 明确要求只能用于 local device，主设备不能这样提交。"
      },
      {
        type: "table",
        title: "主 RenderingDevice 与 local RenderingDevice",
        headers: ["设备", "来源", "典型用途", "提交方式"],
        rows: [
          ["主设备", "`RenderingDevice::get_singleton()` / `RenderingServer.get_rendering_device()`", "引擎主渲染、窗口输出、RendererRD pass。", "`swap_buffers(present)` 结束并执行一帧。"],
          ["local device", "`RenderingServer.create_local_rendering_device()`", "离屏 compute、烘焙、工具任务、可控同步的 GPU 工作。", "`submit()` 后 `sync()` 等待完成。"]
        ]
      },
      {
        type: "heading",
        title: "屏幕和 swapchain"
      },
      {
        type: "paragraph",
        text: "窗口输出也在 RenderingDevice 的边界里。`screen_create()` 从 RenderingContextDriver 拿窗口 surface 并创建 swapchain；`screen_prepare_for_drawing()` 获取当前 framebuffer，必要时 resize swapchain，并把 swapchain 放进本帧待 present 列表；`draw_list_begin_for_screen()` 用这个 screen framebuffer 开始绘制。"
      },
      {
        type: "paragraph",
        text: "`RendererCompositorRD::blit_render_targets_to_screen()` 就直接调用 `RD::get_singleton()->screen_prepare_for_drawing()`，然后 `draw_list_begin_for_screen()`，绑定 blit pipeline，最后由 `RendererCompositorRD::end_frame()` 调 `RD::get_singleton()->swap_buffers(p_present)`。这说明窗口 present 问题要同时看 DisplayServer surface、RenderingDevice swapchain 和 renderer 的 blit。"
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：读懂一个屏幕 blit"
      },
      {
        type: "code",
        code: [
          "# RendererCompositorRD::blit_render_targets_to_screen 的心智模型：",
          "screen_prepare_for_drawing(window)",
          "draw_list = draw_list_begin_for_screen(window)",
          "draw_list_bind_render_pipeline(draw_list, blit_pipeline)",
          "draw_list_bind_index_array(draw_list, index_array)",
          "draw_list_bind_uniform_set(draw_list, texture_uniform_set, 0)",
          "draw_list_set_push_constant(draw_list, blit_params)",
          "draw_list_draw(draw_list, true)",
          "draw_list_end()",
          "swap_buffers(present)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这段代码已经不再谈 Sprite2D、CanvasItem 或 Camera3D，而是在说 framebuffer、pipeline、uniform、push constant 和 draw list。"
      },
      {
        type: "subheading",
        title: "案例二：用 local device 做离屏计算"
      },
      {
        type: "code",
        code: [
          "# 伪代码：工具/烘焙类任务常见结构",
          "var rd = RenderingServer.create_local_rendering_device()",
          "var input = rd.storage_buffer_create(size, data)",
          "var shader = rd.shader_create_from_spirv(bytecode)",
          "var pipeline = rd.compute_pipeline_create(shader)",
          "var uniforms = rd.uniform_set_create([...], shader, 0)",
          "var list = rd.compute_list_begin()",
          "rd.compute_list_bind_compute_pipeline(list, pipeline)",
          "rd.compute_list_bind_uniform_set(list, uniforms, 0)",
          "rd.compute_list_dispatch(list, groups_x, groups_y, groups_z)",
          "rd.compute_list_end()",
          "rd.submit()",
          "rd.sync()",
          "rd.free_rid(input)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这里的重点是 local device：你可以提交一批 compute 工作并同步等待。主渲染设备的帧节奏由引擎控制，不应该用 `submit()` / `sync()` 这套方式。"
      },
      {
        type: "subheading",
        title: "案例三：uniform set 报错怎么追"
      },
      {
        type: "code",
        code: [
          "# 如果看到类似：Uniforms supplied for set are not the same format",
          "# 先查：",
          "# 1. uniform_set_create() 用的是不是同一个 shader 和 set index？",
          "# 2. binding 号、UniformType、RID 数量是否和 shader reflection 匹配？",
          "# 3. pipeline 绑定的 shader variant 是否改变了 set layout？",
          "# 4. uniform set 是否被 free 或 linear pool 生命周期已经结束？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "RenderingDevice 在 debug 路径里保存 pipeline 和 uniform set 的格式信息，就是为了在 Vulkan 这类底层 API 不主动保护你的地方提前报出更可读的错误。"
      },
      {
        type: "subheading",
        title: "案例四：窗口没画出来但 render target 正常"
      },
      {
        type: "code",
        code: [
          "# 排查顺序：",
          "# 1. RendererViewport 是否把 render target 加入 blit 列表？",
          "# 2. screen_prepare_for_drawing() 是否因为窗口最小化或 swapchain resize 返回错误？",
          "# 3. draw_list_begin_for_screen() 是否拿到有效 framebuffer？",
          "# 4. swap_buffers(true) 是否 present 了待提交 swapchain？",
          "# 5. DisplayServer 的窗口 surface 是否仍然有效？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`screen_prepare_for_drawing()` 对最小化、快速 resize 等情况会允许失败而不报噪音错误，所以窗口问题不能只盯着 3D pass。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "RenderingDevice 和周边层",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["RenderingServer", "面向场景和脚本的渲染 API，管理渲染语义 RID。", "不直接录制底层 draw/compute 命令。"],
          ["RendererRD", "把 Godot 的 viewport/canvas/scene/storage 组织成 RD 工作。", "不直接实现 Vulkan/Metal/D3D12。"],
          ["RenderingDevice", "底层资源、pipeline、uniform set、draw/compute list、frame/swapchain。", "不理解 Node、ResourceLoader、场景树生命周期。"],
          ["RenderingDeviceGraph", "记录资源 usage、barrier、命令重排和执行顺序。", "不定义用户可见渲染语义。"],
          ["RenderingDeviceDriver", "具体后端命令接口，映射到 Vulkan/Metal/D3D12。", "不决定 Sprite2D 或 MeshInstance3D 应该怎么组织。"],
          ["DisplayServer", "窗口、surface、事件和平台显示能力。", "不管理 shader、material、pipeline 或 culling。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：RenderingDevice 就是 Vulkan。不是；它是 Godot 的跨平台低层图形抽象，具体 API 在 driver 后面。",
          "误区二：RenderingDevice 能解释 Sprite2D 为什么没显示。它只能解释底层资源和命令，Sprite2D 还要看 CanvasItem、RenderingServer、RendererCanvas。",
          "误区三：所有 RID 都能混用。RenderingServer texture RID 和 RD texture RID 是不同层级的句柄。",
          "误区四：draw_list_draw() 立刻执行。它先记录到命令图，帧提交或 local device submit 时才执行。",
          "误区五：buffer_update 后马上 draw 一定按代码顺序发生。render graph 可能重排 copy 和 draw，公共 API 会阻止危险用法。",
          "误区六：主设备可以随便 submit/sync。源码明确 `submit()` / `sync()` 只给 local device 用。",
          "误区七：窗口不显示就是 shader 错。也可能是 screen framebuffer、swapchain、present 或 DisplayServer surface 问题。",
          "误区八：uniform set 只是资源数组。它必须和 shader set/binding 格式匹配，否则 pipeline 绑定时会报错。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `rendering_device.h:67`，把资源、pipeline、draw list、compute list、screen 和 submit/sync 分区画出来。",
          "读 `rendering_device_driver.h:90`，理解 RD 如何把抽象资源映射到底层 driver ID。",
          "读 texture/buffer/shader/pipeline 创建函数，确认 RD RID 如何包装 driver ID 和验证数据。",
          "读 `draw_list_begin()` 到 `draw_list_end()`，再读 `compute_list_begin()` 到 `compute_list_end()`。",
          "读 `RenderingDeviceGraph` 的 add_draw_list/add_compute_list/add_texture_update，理解命令记录、barrier 和重排。",
          "读 `screen_prepare_for_drawing()`、`draw_list_begin_for_screen()`、`swap_buffers()`，理解窗口 present 边界。",
          "最后回到 RendererRD 的调用点，例如 compositor blit、lightmapper、post effects，观察上层如何使用 RD。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "RenderingDevice 是 Godot 的跨平台底层图形命令层：它把 RendererRD 的渲染/计算工作变成 texture、buffer、pipeline、uniform set 和命令列表，再交给具体 driver 与 GPU 或窗口 swapchain 对接。"
      }
    ]
  },
  {
    id: "displayserver",
    title: "DisplayServer",
    aliases: ["DisplayServer", "DisplayServerWindows", "DisplayServerX11", "DisplayServerWayland", "DisplayServerMacOS", "DisplayServerWeb", "DisplayServerAndroid", "DisplayServerHeadless", "display driver", "process_events", "window_set_input_event_callback", "swap_buffers", "window surface", "clipboard", "IME", "cursor"],
    summary: "Godot 的平台显示与窗口边界：按平台创建窗口和 surface，抽取系统事件，维护屏幕/窗口/鼠标/键盘/剪贴板/IME 等能力，并把输入交给 Window/Viewport。",
    article: [
      {
        type: "lead",
        text: "DisplayServer 是 Godot 和操作系统显示系统之间的边界。它不是渲染场景的后端，也不负责材质、灯光或 shader；它负责把 Windows、X11、Wayland、macOS、Android、Web 等平台的窗口、屏幕、输入、光标、剪贴板、IME、对话框和 surface 能力统一成一套引擎 API。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 DisplayServer 想成 Godot 和电脑桌面系统之间的“窗口管理员”。游戏想开窗口、改标题、全屏、隐藏鼠标、读取剪贴板、收键盘鼠标输入、知道屏幕大小，都要通过它。Windows、macOS、网页浏览器、Android 的窗口系统完全不同，DisplayServer 把这些差异包成统一接口。"
      },
      {
        type: "paragraph",
        text: "它和 RenderingDevice 的区别很重要：DisplayServer 管“窗口和平台事件”，RenderingDevice 管“GPU 资源和图形命令”。窗口能不能创建、鼠标事件有没有进来、surface 是否有效，先看 DisplayServer；shader、pipeline、draw list 是否正确，才看 RenderingDevice。"
      },
      {
        type: "flow",
        title: "DisplayServer 的位置",
        steps: [
          { title: "OS / Browser / Mobile", text: "原生窗口、surface、消息队列、键鼠触摸、剪贴板、IME。" },
          { title: "DisplayServer subclass", text: "Windows/X11/Wayland/macOS/Web/Android 等平台实现。" },
          { title: "DisplayServer API", text: "统一窗口、屏幕、输入、鼠标、键盘、剪贴板、对话框接口。" },
          { title: "Window / Viewport", text: "注册回调，接收 InputEvent 并分发给场景和 GUI。" },
          { title: "RenderingDevice", text: "使用窗口 surface/swapchain 输出画面，但不处理输入语义。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`DisplayServer` 声明在 `servers/display/display_server.h:62`，继承 `Object`，并通过 singleton 暴露给引擎其他系统。它内部有一个最多 64 项的创建函数注册表：平台实现调用 `register_create_function()` 注册自己，`DisplayServer::create()` 再按索引和渲染 driver 创建实际 DisplayServer。"
      },
      {
        type: "paragraph",
        text: "`DisplayServer::register_create_function()` 在 `display_server.cpp:2027`，会把新平台 driver 插到 headless 前面，因为 headless 总是最后一个 fallback。`DisplayServer::create()` 在 `:2051` 调用对应 create function。`Main::setup2()` 在 `main/main.cpp:3343` 创建 DisplayServer，如果失败会尝试其他 display driver，并跳过 headless，除非用户明确请求。"
      },
      {
        type: "table",
        title: "DisplayServer 主要职责",
        headers: ["职责", "典型 API", "说明"],
        rows: [
          ["事件泵", "`process_events()`", "从平台消息队列取出窗口、鼠标、键盘、触摸、文件拖放等事件。"],
          ["窗口管理", "`window_set_title()`、`window_set_size()`、`window_set_mode()`、`window_set_flag()`", "统一窗口大小、位置、全屏、置顶、焦点、最小/最大尺寸等能力。"],
          ["屏幕信息", "`get_screen_count()`、`screen_get_size()`、`screen_get_dpi()`、`screen_get_refresh_rate()`", "查询多屏、DPI、可用区域、刷新率、缩放等平台信息。"],
          ["输入回调", "`window_set_input_event_callback()`、`window_set_input_text_callback()`", "把平台输入交给 Window，再进入 Viewport 和 SceneTree。"],
          ["鼠标键盘", "`mouse_set_mode()`、`warp_mouse()`、`cursor_set_shape()`、`keyboard_get_layout_name()`", "捕获/隐藏鼠标、光标形状、键盘布局、虚拟键盘和 IME。"],
          ["剪贴板/对话框", "`clipboard_get()`、`clipboard_set()`、`file_dialog_show()`", "接入平台剪贴板、原生文件对话框、提示框等用户系统能力。"],
          ["显示 surface", "平台实现里的 rendering context/window create", "给 RenderingDevice 或 GL manager 提供可绘制窗口 surface。"]
        ]
      },
      {
        type: "heading",
        title: "平台实现和注册"
      },
      {
        type: "paragraph",
        text: "DisplayServer 不是一个跨平台万能实现，而是一组平台子类。Windows 在 `platform/windows/display_server_windows.cpp:8279` 注册 `windows`；LinuxBSD 的 X11 在 `platform/linuxbsd/x11/display_server_x11.cpp:7589` 注册 `x11`，Wayland 在 `platform/linuxbsd/wayland/display_server_wayland.cpp:2604` 注册 `wayland`；macOS 在 `display_server_macos.mm:3524` 注册 `macos`；Android 在 `display_server_android.cpp:682` 注册 `android`；Web 在 `display_server_web.cpp:1239` 注册 `web`。"
      },
      {
        type: "table",
        title: "常见平台实现差异",
        headers: ["平台", "源码入口", "特别关注"],
        rows: [
          ["Windows", "`platform/windows/display_server_windows.cpp`", "Win32 消息、窗口样式、RAW input、剪贴板、IME、DPI、HDR。"],
          ["Linux X11", "`platform/linuxbsd/x11/display_server_x11.cpp`", "X11 event、窗口管理器协议、XInput、剪贴板 selection、IME。"],
          ["Linux Wayland", "`platform/linuxbsd/wayland/display_server_wayland.cpp`", "Wayland seat、surface、xdg shell、限制更多的平台能力。"],
          ["macOS", "`platform/macos/display_server_macos.mm`", "Cocoa window/view、NSEvent、菜单栏、Retina scale、原生输入法。"],
          ["Android", "`platform/android/display_server_android.cpp`", "Java/Kotlin Activity、Surface、虚拟键盘、触摸和传感器桥接。"],
          ["Web", "`platform/web/display_server_web.cpp`", "JavaScript 回调、canvas、浏览器输入限制、剪贴板/全屏权限。"],
          ["Headless", "`servers/display/display_server_headless.*`", "无真实窗口，供服务器、导出工具、CI 或 doctool 使用。"]
        ]
      },
      {
        type: "heading",
        title: "输入如何进入场景"
      },
      {
        type: "paragraph",
        text: "平台事件不会直接调用脚本节点。平台 DisplayServer 把原生事件转换成 `InputEvent`，再调用 Window 注册的回调。`Window::_update_window_callbacks()` 在 `scene/main/window.cpp:1452` 把 `_window_input()`、`_window_input_text()`、drop files、window event 等回调注册到 DisplayServer。"
      },
      {
        type: "paragraph",
        text: "`Window::_window_input()` 在 `window.cpp:1996` 接收 InputEvent，先让 Window 子类有机会处理，再发 `window_input` 信号，最后调用 `push_input()`。`Viewport::push_input()` 在 `viewport.cpp:3501` 会做本地坐标转换、mouse over 更新、子窗口转发，然后按顺序调用 `_input`、GUI 输入、`_unhandled_input`。"
      },
      {
        type: "flow",
        title: "平台输入到脚本回调",
        steps: [
          { title: "平台消息", text: "Win32 message、NSEvent、X11/Wayland event、JS callback、Android input。" },
          { title: "DisplayServer subclass", text: "转换成 Godot 的 InputEvent 或文本输入。" },
          { title: "Window callback", text: "`window_set_input_event_callback()` 指向 `Window::_window_input()`。" },
          { title: "Viewport", text: "`push_input()` 处理 GUI、子窗口、坐标和 handled 状态。" },
          { title: "SceneTree / Control", text: "进入 `_input`、`_gui_input`、`_unhandled_input` 等用户层逻辑。" }
        ]
      },
      {
        type: "heading",
        title: "每帧事件泵"
      },
      {
        type: "paragraph",
        text: "DisplayServer 的事件泵在主循环里持续执行。Windows 的 `OS_Windows::run()` 在 `platform/windows/os_windows.cpp:2353` 每轮先调 `DisplayServer::get_singleton()->process_events()`，再进入 `Main::iteration()`；LinuxBSD 的同类路径在 `platform/linuxbsd/os_linuxbsd.cpp:1000`。`Main::iteration()` 内部也会在特定输入刷新策略下再次处理事件。"
      },
      {
        type: "paragraph",
        text: "这解释了很多输入问题：如果平台事件没有被 process_events 抽出来，后面的 Input、Window、Viewport 都不会收到；如果 GUI 控件消费了事件，脚本的 unhandled 阶段也可能看不到。DisplayServer 是入口，不是最终分发规则。"
      },
      {
        type: "heading",
        title: "和渲染输出的关系"
      },
      {
        type: "paragraph",
        text: "DisplayServer 负责窗口和 surface，RenderingDevice 负责 GPU 命令和 swapchain。RD 路径下，DisplayServer 平台实现通常创建窗口和 rendering context 所需的 surface；`RenderingDevice::screen_create()` 再从 context 取该 window surface 创建 swapchain。OpenGL 这类路径则常在平台 DisplayServer/GL manager 中实现 `swap_buffers()`。"
      },
      {
        type: "paragraph",
        text: "因此“窗口黑屏”不能只看一个系统：窗口是否可绘制、surface 是否有效、swapchain 是否 resize、RendererViewport 是否 blit、RenderingDevice 是否 present，都可能参与。DisplayServer 主要回答“窗口和平台层是否正常”。"
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：启动时 display driver 选择失败"
      },
      {
        type: "code",
        code: [
          "# 阅读路线：",
          "# 1. main/main.cpp:3343 DisplayServer::create(...)",
          "# 2. display_server.cpp:2051 调注册表里的 create_function",
          "# 3. 当前平台 register_*_driver 是否执行",
          "# 4. 当前 rendering_driver 是否被该 DisplayServer 支持",
          "# 5. fallback 是否跳过 headless，或用户是否显式 --headless"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "如果启动日志提示 display driver failed，不要先看 RendererRD。先确认平台 DisplayServer 是否注册、命令行 display/rendering driver 是否匹配、窗口或 surface 创建是否失败。"
      },
      {
        type: "subheading",
        title: "案例二：鼠标点击脚本没收到"
      },
      {
        type: "code",
        code: [
          "# 排查顺序：",
          "# 1. 平台 DisplayServer 的 process_events 是否被主循环调用？",
          "# 2. 平台实现是否把原生鼠标事件转成 InputEventMouseButton？",
          "# 3. Window 是否注册了 window_set_input_event_callback？",
          "# 4. Window::_window_input 是否调用 push_input？",
          "# 5. Viewport GUI 或 Control 是否 accept_event 消费了事件？",
          "# 6. 脚本监听的是 _input、_gui_input 还是 _unhandled_input？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "DisplayServer 只负责把事件送进 Godot。事件进来后是否被 GUI 消费，属于 Viewport、Control 和 SceneTree 输入链路。"
      },
      {
        type: "subheading",
        title: "案例三：窗口 resize 后画面异常"
      },
      {
        type: "code",
        code: [
          "# 先分层：",
          "# DisplayServer: window size / scale / DPI / can_draw 是否正确？",
          "# Window / Viewport: viewport size、content scale、render target 是否更新？",
          "# RenderingDevice: screen_prepare_for_drawing 是否触发 swapchain resize？",
          "# RendererRD: render target 是否重新 blit 到 screen？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "resize 是跨层问题：平台窗口尺寸变化从 DisplayServer 进来，但最终影响 Viewport render target、RD framebuffer 和 RendererRD blit。"
      },
      {
        type: "subheading",
        title: "案例四：导出服务器或 CI 不想开窗口"
      },
      {
        type: "code",
        code: [
          "# 常见路径：",
          "godot --headless --script tools/build_report.gd",
          "",
          "# 源码心智：",
          "# --headless 选择 DisplayServerHeadless",
          "# 没有真实窗口和平台 surface",
          "# 适合服务器、测试、doctool、批处理导入导出"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Headless 仍然是 DisplayServer 的一种实现，但很多窗口、输入、屏幕、剪贴板能力没有真实平台效果。读 headless bug 时不要期待它和桌面窗口行为一致。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "DisplayServer 和周边层级",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["OS", "进程、文件、线程、时间、动态库等更底层跨平台服务。", "不直接分发 Godot 场景输入。"],
          ["DisplayServer", "窗口、屏幕、surface、输入回调、鼠标键盘、剪贴板、IME、对话框。", "不管理材质、Mesh、shader、光照或场景剔除。"],
          ["Input", "维护输入状态、action、鼠标位置、按键状态。", "不直接调用原生 Win32/X11/Cocoa API。"],
          ["Window / Viewport", "把 DisplayServer 输入分发给 GUI、SceneTree 和子窗口。", "不创建平台原生窗口底层对象。"],
          ["RenderingDevice", "GPU 资源、命令列表、swapchain/framebuffer。", "不处理键盘、鼠标、剪贴板和窗口事件语义。"],
          ["RendererRD", "渲染场景内容并 blit 到 render target 或 screen。", "不负责系统消息队列和平台窗口生命周期。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：DisplayServer 是渲染后端。不是；它主要是平台显示、窗口和输入边界。",
          "误区二：窗口黑屏一定是 DisplayServer。也可能是 Viewport、RendererRD、RenderingDevice 或 swapchain。",
          "误区三：InputEvent 直接从 OS 到脚本。中间有 DisplayServer、Window、Viewport、GUI 消费和 SceneTree 分发。",
          "误区四：所有平台窗口能力一致。Wayland、Web、Android、macOS、Windows 的限制差异很大。",
          "误区五：headless 只是隐藏窗口。它是独立 DisplayServer 实现，很多平台窗口/输入能力不存在。",
          "误区六：DisplayServer 应该知道 shader 编译失败。shader/RD/pipeline 问题不属于这一层。",
          "误区七：剪贴板、IME、虚拟键盘只是 Input 的事。它们通常从 DisplayServer 的平台实现接进来。",
          "误区八：process_events 只是辅助函数。它是平台消息进入 Godot 的关键入口。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `servers/display/display_server.h:62`，把主接口分成 process、mouse、keyboard、clipboard、screen、window、dialog 几组。",
          "读 `display_server.cpp:2027` 和 `:2051`，理解平台实现如何注册和创建。",
          "读 `main/main.cpp:3343` 的创建和 fallback 逻辑，确认 display driver 与 rendering driver 的关系。",
          "选一个平台实现，例如 Windows/X11/Wayland/macOS/Web/Android，读 register、create_func、构造函数和 `process_events()`。",
          "跟 `Window::_update_window_callbacks()` 到 `_window_input()`，再进 `Viewport::push_input()`，建立输入链路。",
          "遇到窗口输出问题，再把 DisplayServer surface 与 RenderingDevice screen/swapchain、RendererRD blit 串起来看。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "DisplayServer 是 Godot 的平台显示入口：它把不同操作系统的窗口、屏幕、输入、剪贴板、IME 和 surface 能力统一起来，再把事件交给 Window/Viewport，把可绘制窗口交给渲染链路。"
      }
    ]
  },
  {
    id: "textureimage",
    title: "Texture / Image",
    aliases: ["Texture / Image", "Texture/Image", "Texture", "Image", "Texture2D", "ImageTexture", "CompressedTexture2D", "PortableCompressedTexture2D", "TextureStorage", "ImageLoader", "ImageFormatLoader", "texture_2d_create", "texture_2d_update", "texture_get_rd_texture", "RD texture"],
    summary: "Godot 图片和纹理的分层：Image 是 CPU 侧像素数据，Texture2D/ImageTexture 是资源对象，RenderingServer texture RID 是渲染语义句柄，RenderingDevice texture 才是底层 GPU 资源。",
    article: [
      {
        type: "lead",
        text: "Texture / Image 这组概念最容易混，因为“图片文件、Image、Texture2D、RenderingServer texture RID、RD texture RID”经常都被口头叫作贴图。读源码时必须分层：Image 管 CPU 内存里的像素和格式转换；Texture2D/ImageTexture 是用户可见 Resource；RenderingServer 的 texture RID 是渲染 Server 句柄；RendererRD 的 TextureStorage 再把它变成 RenderingDevice 的 GPU texture。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Image 想成“电脑内存里的一张像素表”，你能读写每个像素、改大小、转格式、压缩、保存成 PNG/JPG。Texture 则更像“拿给渲染系统用的贴图资源”：Sprite2D、Material、UI 图标一般拿的是 Texture2D，而不是直接拿 Image。"
      },
      {
        type: "paragraph",
        text: "ImageTexture 就是常见桥梁：你把一个 Image 交给 `ImageTexture.create_from_image()`，它会创建一个 Texture2D 资源，并把像素提交给 RenderingServer。之后真正画到屏幕时，后端已经在使用 GPU 纹理了。"
      },
      {
        type: "flow",
        title: "从图片文件到 GPU 纹理",
        steps: [
          { title: "文件", text: "PNG/JPG/WebP/DDS/KTX/SVG 等磁盘或内存数据。" },
          { title: "ImageLoader", text: "按扩展名选择 ImageFormatLoader，把文件解码成 Image。" },
          { title: "Image", text: "CPU 侧 width、height、format、mipmaps、byte data。" },
          { title: "Texture2D / ImageTexture", text: "用户可见 Resource，保存尺寸格式并暴露 draw/get_rid。" },
          { title: "RenderingServer texture RID", text: "`texture_2d_create()` 进入渲染 Server。" },
          { title: "TextureStorage / RD texture", text: "转换成 RenderingDevice texture，供 RendererRD 采样和绘制。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Image` 声明在 `core/io/image.h:64`，继承 `Resource`。它的核心字段很直接：`format`、`Vector<uint8_t> data`、`width`、`height`、`mipmaps`。它提供 `load()`、`load_from_file()`、`create_from_data()`、`set_data()`、`convert()`、`resize()`、`generate_mipmaps()`、`compress()`、`decompress()`、`get_pixel()`、`set_pixel()` 等 CPU 侧图像操作。"
      },
      {
        type: "paragraph",
        text: "`ImageLoader` 在 `core/io/image_loader.cpp:83` 根据扩展名找 `ImageFormatLoader`。PNG 的 loader 在 `drivers/png/image_loader_png.cpp`，JPG/WebP/BMP/TGA/SVG/HDR/EXR/KTX/DDS 等在 drivers 或 modules 中注册。`Image::load()` 在 `image.cpp:2766` 最终调用 `ImageLoader::load_image()`。"
      },
      {
        type: "table",
        title: "Image、Texture、RD texture 分层",
        headers: ["层级", "源码入口", "保存什么", "典型用途"],
        rows: [
          ["Image", "`core/io/image.h:64`", "CPU 字节数组、尺寸、格式、mipmap 信息。", "解码、像素编辑、保存、生成纹理数据。"],
          ["Texture2D", "`scene/resources/texture.h:41`", "用户可见 2D 纹理抽象，暴露尺寸、格式、draw、get_image。", "Sprite2D、Control、Material、Inspector 属性。"],
          ["ImageTexture", "`scene/resources/image_texture.cpp`", "Texture2D 的具体资源，内部持有 RenderingServer texture RID。", "运行时从 Image 创建/更新贴图。"],
          ["CompressedTexture2D", "`scene/resources/compressed_texture.*`", "导入后的压缩/流式纹理资源，按需加载。", "项目资源、导出纹理、VRAM 压缩格式。"],
          ["RenderingServer texture RID", "`servers/rendering/rendering_server.h:110`", "渲染 Server 层纹理句柄。", "让场景和资源不直接接触后端对象。"],
          ["RenderingDevice texture RID", "`RenderingDevice::texture_create()`", "底层 GPU texture / texture view。", "RendererRD 采样、render target、compute。"]
        ]
      },
      {
        type: "heading",
        title: "ImageTexture 如何提交给 Server"
      },
      {
        type: "paragraph",
        text: "`ImageTexture::create_from_image()` 在 `image_texture.cpp:58` 实例化资源并调用 `set_image()`。`set_image()` 保存宽高、格式和 mipmap 状态；如果内部 texture RID 为空，就调用 `RenderingServer::texture_2d_create(p_image)`；如果已有 RID，则创建新 texture 并 `texture_replace()`。`update()` 则要求新 Image 的尺寸、格式、mipmap 设置都和原纹理一致，然后调用 `texture_2d_update()`。"
      },
      {
        type: "paragraph",
        text: "`Texture2D` 本身是抽象资源，声明在 `texture.h:41`。它定义 `get_format()`、`get_width()`、`get_height()`、`get_image()`、`draw()`、`draw_rect()`、`draw_rect_region()` 等接口。真正持有 RID 的通常是 ImageTexture、CompressedTexture2D、TextureRD、CameraTexture 等派生类。"
      },
      {
        type: "table",
        title: "ImageTexture 的关键行为",
        headers: ["方法", "约束", "后端动作"],
        rows: [
          ["`create_from_image(image)`", "image 不能为 null 或 empty。", "实例化 ImageTexture，再调用 `set_image()`。"],
          ["`set_image(image)`", "允许首次创建或替换纹理。", "调用 `texture_2d_create()`，必要时 `texture_replace()`。"],
          ["`update(image)`", "尺寸、格式、mipmap 必须与已有纹理一致。", "调用 `texture_2d_update()` 上传新像素。"],
          ["`get_image()`", "只有 image_stored 时有效。", "通过 RenderingServer 从后端取回 Image，可能有同步成本。"],
          ["`get_rid()`", "如果还没有 RID，会创建 placeholder。", "返回 RenderingServer texture RID，不是 RD texture RID。"]
        ]
      },
      {
        type: "heading",
        title: "TextureStorage 如何变成 RD texture"
      },
      {
        type: "paragraph",
        text: "RenderingServerDefault 会把 texture API 分发给后端 texture storage。RD 后端的 `TextureStorage` 在 `servers/rendering/renderer_rd/storage_rd/texture_storage.*`。`texture_allocate()` 在 `texture_storage.cpp:935` 分配 RenderingServer 层 texture RID；`texture_2d_initialize()` 在 `:967` 校验 Image 格式，把 Image format 映射到 RD format，设置 `RD::TextureFormat`，再调用 `RD::get_singleton()->texture_create()` 创建底层 RD texture。"
      },
      {
        type: "paragraph",
        text: "如果存在 sRGB 变体，TextureStorage 会用 `texture_create_shared()` 创建 shared view。`texture_2d_update()` 在 `:1612` 先检查尺寸和格式，再验证 Image 格式，最后调用 `RD::texture_update()` 上传数据。`texture_rd_initialize()` 则反过来：把一个已有 RD texture 包成 RenderingServer texture，让高级资源或渲染路径可以把 RD 纹理当 Texture 使用。"
      },
      {
        type: "flow",
        title: "ImageTexture.set_image 后端路径",
        steps: [
          { title: "ImageTexture.set_image", text: "保存 width、height、format、mipmaps。" },
          { title: "RenderingServer.texture_2d_create", text: "创建渲染 Server 层 texture RID。" },
          { title: "TextureStorage.texture_2d_initialize", text: "验证 Image，转换 Image::Format 到 RD::DataFormat。" },
          { title: "RD::texture_create", text: "创建 GPU texture，并可创建 sRGB shared view。" },
          { title: "Texture2D.get_rid", text: "Sprite2D/Material 使用 RenderingServer texture RID。" },
          { title: "RendererRD sampling", text: "绘制时从 TextureStorage 取 RD texture 给 shader 采样。" }
        ]
      },
      {
        type: "heading",
        title: "导入纹理和运行时 Image 的区别"
      },
      {
        type: "paragraph",
        text: "运行时 `Image.load()` 只是把文件解码成 Image；`ImageTexture.create_from_image()` 则立即创建可渲染 Texture。编辑器导入的图片通常会经过 `ResourceImporterTexture`，保存成 CompressedTexture2D 等导入资源，包含压缩格式、mipmap、normal/roughness/3D 检测、平台压缩设置等。这条路更适合项目资源和导出。"
      },
      {
        type: "table",
        title: "运行时加载与导入资源",
        headers: ["路径", "产物", "适合", "注意点"],
        rows: [
          ["`Image.load()`", "Image CPU 数据。", "像素处理、截图保存、程序化生成。", "不能直接给 Sprite2D 显示，需转 Texture。"],
          ["`ImageTexture.create_from_image()`", "ImageTexture / Texture2D。", "运行时生成贴图、编辑器预览、动态图标。", "上传到渲染后端，有 GPU 资源成本。"],
          ["编辑器导入", "CompressedTexture2D / layered / 3D texture。", "正式项目资产、平台压缩、mipmap 和导出。", "不要绕过导入设置直接用原图判断运行时格式。"],
          ["`TextureRD` / `texture_rd_initialize()`", "把 RD texture 包成 Texture Resource。", "后处理、渲染插件、GPU 生成纹理。", "必须确认 RD texture usage、格式和共享 view。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：运行时生成一张贴图"
      },
      {
        type: "code",
        code: [
          "var image := Image.create_empty(64, 64, false, Image.FORMAT_RGBA8)",
          "image.fill(Color.RED)",
          "",
          "var texture := ImageTexture.create_from_image(image)",
          "$Sprite2D.texture = texture"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这里 Image 是 CPU 像素表，ImageTexture 才是可给 Sprite2D 使用的 Texture2D 资源。"
      },
      {
        type: "subheading",
        title: "案例二：更新已有动态纹理"
      },
      {
        type: "code",
        code: [
          "var image := Image.create_empty(128, 128, false, Image.FORMAT_RGBA8)",
          "var texture := ImageTexture.create_from_image(image)",
          "",
          "# 后续每帧或按需更新：尺寸、格式、mipmap 必须一致。",
          "image.fill(Color(randf(), randf(), randf()))",
          "texture.update(image)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`ImageTexture::update()` 会检查 width、height、format、mipmaps；不一致时不是自动重建，而是报错。需要换尺寸或格式时用 `set_image()` 或新建 Texture。"
      },
      {
        type: "subheading",
        title: "案例三：Image 改了但画面没变"
      },
      {
        type: "code",
        code: [
          "var image := texture.get_image()",
          "image.set_pixel(0, 0, Color.GREEN)",
          "",
          "# 只改 CPU Image 不会自动改 GPU texture。",
          "texture.update(image)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Image 和 Texture 是两份不同层级的数据。修改 Image 后必须重新提交给 Texture/RenderingServer，渲染后端才会看到变化。"
      },
      {
        type: "subheading",
        title: "案例四：排查贴图显示成黑色或透明"
      },
      {
        type: "code",
        code: [
          "# 阅读路线：",
          "# 1. Image 是否 empty？width/height/format/data 是否正确？",
          "# 2. ImageLoader 是否识别扩展名并成功解码？",
          "# 3. ImageTexture 是否调用 set_image 或 create_from_image？",
          "# 4. Texture2D.get_rid 是否返回 placeholder 还是有效 texture RID？",
          "# 5. TextureStorage 是否创建 RD texture，sRGB view 是否正确？",
          "# 6. 材质或 CanvasItem 是否真的绑定了这个 texture？"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "不要一上来查 shader。先确认 CPU 数据、资源对象、Server RID、RD texture 和材质绑定每一层是否存在。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "Texture / Image 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Image", "CPU 侧像素、格式、mipmap、图像处理和文件解码结果。", "不表示已经在 GPU 上可采样。"],
          ["Texture2D", "用户可见 2D 纹理资源接口。", "不一定保存 CPU 像素副本。"],
          ["ImageTexture", "从 Image 创建/更新 RenderingServer texture RID。", "不负责导入压缩策略。"],
          ["CompressedTexture2D", "导入后的压缩纹理资源。", "不等同于原始 PNG/JPG 文件。"],
          ["TextureStorage", "Server texture RID 到 RD texture 的后端状态。", "不处理 ResourceLoader 缓存和导入选项 UI。"],
          ["RenderingDevice texture", "GPU texture、view、usage、format。", "不理解 Sprite2D 或 Resource 的高层语义。"],
          ["ImageLoader", "把图片文件/内存解码成 Image。", "不创建 Texture2D 或 GPU 资源。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Image 就是 Texture。Image 是 CPU 数据，Texture 才是可渲染资源接口。",
          "误区二：修改 Image 后画面会自动变化。不会；需要 `texture.update(image)` 或重新 set image。",
          "误区三：Texture2D 一定能便宜地 `get_image()`。从 GPU 取回可能导致同步和拷贝成本。",
          "误区四：所有图片文件运行时都按原格式使用。导入器可能生成压缩纹理、mipmap 和平台特定格式。",
          "误区五：RenderingServer texture RID 可以当 RD texture RID 用。两者属于不同层级。",
          "误区六：`ImageTexture.update()` 可以随便换尺寸。它要求尺寸、格式、mipmap 完全一致。",
          "误区七：sRGB 只是 shader 问题。TextureStorage 会创建普通和 sRGB view，材质采样路径会受影响。",
          "误区八：贴图黑了就直接查 shader。先查 Image 数据、Texture resource、RID、后端 RD texture 和绑定关系。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/io/image.h:64`，确认 Image 字段和 CPU 图像操作能力。",
          "读 `Image::load()` 到 `ImageLoader::load_image()`，理解图片格式 loader 如何接入。",
          "读 `scene/resources/texture.h:41`，理解 Texture2D 是抽象 Resource 接口。",
          "读 `ImageTexture::set_image()`、`update()`、`get_rid()`，理解 Image 如何变成 RenderingServer texture RID。",
          "读 `RenderingServer::texture_2d_create()` 相关 API，再进入 `RendererTextureStorage` 抽象。",
          "读 `TextureStorage::texture_2d_initialize()` 和 `_texture_2d_update()`，看 Image::Format 如何映射到 RD texture。",
          "最后看 `ResourceImporterTexture` 和 `CompressedTexture2D`，理解编辑器导入纹理与运行时 Image 的差异。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Image 是 CPU 像素数据，Texture2D/ImageTexture 是用户可见资源，RenderingServer texture RID 是渲染句柄，RenderingDevice texture 是 GPU 资源；读贴图源码时先判断自己站在哪一层。"
      }
    ]
  },
  {
    id: "mesh",
    title: "Mesh",
    aliases: ["Mesh", "ArrayMesh", "MeshInstance3D", "PrimitiveMesh", "ImmediateMesh", "SurfaceTool", "surface arrays", "Mesh surface", "ARRAY_VERTEX", "ARRAY_INDEX", "PRIMITIVE_TRIANGLES", "mesh RID", "mesh_create", "mesh_add_surface_from_arrays", "mesh_create_surface_data_from_arrays", "mesh_surface_update_vertex_region", "RendererMeshStorage", "MeshStorage", "vertex buffer", "index buffer"],
    summary: "Godot 3D 几何的分层：Mesh 是资源接口，ArrayMesh 保存 surface arrays，MeshInstance3D 把 mesh RID 放进场景，RendererRD 再把 surface data 变成 GPU buffer。",
    article: [
      {
        type: "lead",
        text: "Mesh 是 Godot 里“几何长什么样”的资源概念。它不等于场景里的物体，也不等于 GPU 命令。更准确地说：Mesh/ArrayMesh 描述顶点、索引、法线、UV、材质和包围盒；MeshInstance3D 描述“这个 Mesh 在场景里放在哪里”；RenderingServer 的 mesh RID 是渲染 Server 看到的几何资源句柄；RendererRD 的 MeshStorage 才把 surface data 变成 RD vertex/index buffer。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Mesh 想成“模型图纸”。图纸上有点的位置、三角形怎么连、每个面用什么材质。MeshInstance3D 像是把这张图纸摆到房间里的一个家具实例：同一张椅子图纸可以摆十把椅子，每把椅子位置不同，但几何数据可以共享。"
      },
      {
        type: "paragraph",
        text: "所以你在编辑器里看到的一个 3D 物体通常至少有两层：`MeshInstance3D` 是节点，负责位置、可见性、材质覆盖、骨骼等场景语义；`Mesh` 是资源，负责顶点和 surface。换材质可能只改某个 surface 或实例覆盖；移动物体只改 instance transform，不会复制顶点。"
      },
      {
        type: "flow",
        title: "一张 Mesh 画出来的路径",
        steps: [
          { title: "ArrayMesh / PrimitiveMesh", text: "保存或生成 surface arrays：顶点、索引、法线、UV、颜色、骨骼权重。" },
          { title: "Mesh.get_rid()", text: "资源返回 RenderingServer 层 mesh RID；ArrayMesh 首次需要时会创建 RID。" },
          { title: "MeshInstance3D.set_mesh", text: "把 mesh RID 设为 VisualInstance3D 的 base。" },
          { title: "World3D scenario", text: "实例进入当前 3D 世界，带着 transform、可见性、材质覆盖。" },
          { title: "RendererRD MeshStorage", text: "把每个 surface 转成 RD vertex buffer、attribute buffer、index buffer。" },
          { title: "RendererSceneRenderRD", text: "剔除、排序、选择材质和 pipeline，最终提交 draw。"}
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Mesh` 基类声明在 `scene/resources/mesh.h:50`，继承 `Resource`。它定义的是资源接口：`get_surface_count()`、`surface_get_arrays()`、`surface_get_format()`、`surface_get_primitive_type()`、`surface_set_material()`、`get_aabb()`、`get_rid()` 等。`Mesh` 本身不要求一定用 ArrayMesh 存数据，PrimitiveMesh、PlaceholderMesh、脚本派生类也可以实现这些虚接口。"
      },
      {
        type: "paragraph",
        text: "`ArrayMesh` 是最常见的可序列化 Mesh。它在 `mesh.h:300` 继承 Mesh，内部有 `Vector<Surface> surfaces`、`mutable RID mesh`、整体 `AABB aabb`、blend shape 信息和 custom AABB。`_create_if_empty()` 在 `mesh.cpp:1580` 左右首次创建 `RenderingServer::mesh_create()`，并设置 blend shape mode/count/path。"
      },
      {
        type: "paragraph",
        text: "`ArrayMesh::add_surface_from_arrays()` 在 `mesh.cpp:1820` 调用 `RenderingServer::mesh_create_surface_data_from_arrays()`，把脚本层的 Array 槽位校验并打包成 `RenderingServerTypes::SurfaceData`，再通过 `ArrayMesh::add_surface()` 调用 `RenderingServer::mesh_add_surface()`。这一步会更新 ArrayMesh 自己的 surface 元信息、AABB、属性列表和 changed 信号。"
      },
      {
        type: "table",
        title: "Mesh 分层边界",
        headers: ["层级", "典型类型/函数", "负责什么", "不负责什么"],
        rows: [
          ["资源接口", "`Mesh`", "定义 surface、AABB、材质、RID、碰撞形状生成等通用能力。", "不规定数据一定如何存储。"],
          ["资源实现", "`ArrayMesh`、`PrimitiveMesh`", "保存或生成 surface arrays，并持有 RenderingServer mesh RID。", "不保存每个场景实例的位置。"],
          ["场景实例", "`MeshInstance3D`", "把 Mesh RID 作为 instance base，管理 transform、骨骼、blend shape 权重、surface override material。", "不拥有 Mesh 的顶点数据。"],
          ["Server 句柄", "`RenderingServer::mesh_create()`", "提供渲染 Server 层 mesh RID 和 mesh API。", "不直接暴露 Vulkan/D3D12 buffer 细节。"],
          ["RD 后端", "`RendererRD::MeshStorage`", "创建 RD vertex/index/storage buffer，维护 material cache、LOD、AABB、依赖通知。", "不理解节点 owner、场景树生命周期。"],
          ["材质", "`Material` / surface override", "决定每个 surface 用什么 shader 参数和渲染状态。", "不改变顶点数组本身。"]
        ]
      },
      {
        type: "heading",
        title: "Surface arrays 是核心数据结构"
      },
      {
        type: "paragraph",
        text: "Godot 的 Mesh surface 通常用一个长度为 `Mesh.ARRAY_MAX` 的 Array 表示。每个槽位固定含义：顶点、法线、切线、颜色、UV、骨骼、权重、索引等。`RenderingServer::mesh_create_surface_data_from_arrays()` 在 `rendering_server.cpp:1179` 校验这些槽位：顶点必须是 `PackedVector2Array` 或 `PackedVector3Array`；索引来自 `PackedInt32Array`；有 normal 时会补 tangent format；blend shape 的 vertex/normal/tangent 格式必须和主 surface 匹配。"
      },
      {
        type: "table",
        title: "常见 Mesh Array 槽位",
        headers: ["槽位", "常见类型", "用途", "读源码时注意"],
        rows: [
          ["`ARRAY_VERTEX`", "`PackedVector3Array` 或 `PackedVector2Array`", "顶点位置，是 Mesh 可见的基础。", "2D 顶点会设置 `ARRAY_FLAG_USE_2D_VERTICES`。"],
          ["`ARRAY_NORMAL`", "`PackedVector3Array`", "光照需要的法线方向。", "有 normal 时 format 里会考虑 tangent 相关约束。"],
          ["`ARRAY_TANGENT`", "`PackedFloat32Array`", "法线贴图等切线空间效果。", "不是普通 Vector3 数组，通常按 4 float 一组。"],
          ["`ARRAY_COLOR`", "`PackedColorArray`", "顶点色。", "进入 attribute buffer，不等同材质颜色。"],
          ["`ARRAY_TEX_UV` / `ARRAY_TEX_UV2`", "`PackedVector2Array`", "贴图坐标和光照贴图坐标。", "UV 错误通常表现为贴图拉伸或错位。"],
          ["`ARRAY_BONES` / `ARRAY_WEIGHTS`", "`PackedInt32Array` / `PackedFloat32Array`", "骨骼蒙皮。", "4/8 权重会影响 format 和 skin buffer。"],
          ["`ARRAY_INDEX`", "`PackedInt32Array`", "顶点复用和三角形连接顺序。", "没有 index 时按顶点顺序绘制。"]
        ]
      },
      {
        type: "flow",
        title: "ArrayMesh.add_surface_from_arrays 内部流程",
        steps: [
          { title: "脚本传入 arrays", text: "Array 长度必须等于 `ARRAY_MAX`，每个槽位放对应 PackedArray。" },
          { title: "创建 SurfaceData", text: "`mesh_create_surface_data_from_arrays()` 校验类型、计算 format、打包二进制数据。" },
          { title: "ArrayMesh.add_surface", text: "记录 surface 元信息、AABB、primitive、array length、index length。" },
          { title: "RenderingServer.mesh_add_surface", text: "把 SurfaceData 交给 Server 层 mesh RID。" },
          { title: "RendererRD.MeshStorage", text: "创建 vertex/attribute/skin/index/storage buffer，并通知依赖更新。" }
        ]
      },
      {
        type: "heading",
        title: "MeshInstance3D 只连接资源和实例"
      },
      {
        type: "paragraph",
        text: "`MeshInstance3D::set_mesh()` 在 `scene/3d/mesh_instance_3d.cpp:120`。设置新 mesh 时，它会断开旧 mesh 的 changed 连接，保存 `Ref<Mesh>`；如果 mesh 有效，先调用 `set_base(mesh->get_rid())`，再连接 `_mesh_changed()`。`set_base()` 来自 VisualInstance3D，把这个 instance RID 的 base 指向 mesh RID。"
      },
      {
        type: "paragraph",
        text: "`_mesh_changed()` 会按 surface 数量调整 `surface_override_materials`，按 blend shape 数量调整权重和动态属性，然后把已有 surface override material 重新提交给 `RenderingServer::instance_set_surface_override_material()`。因此 Mesh 资源变化会影响使用它的实例，但实例材质覆盖、blend shape 权重和 transform 仍然属于 MeshInstance3D。"
      },
      {
        type: "table",
        title: "MeshInstance3D 相关行为",
        headers: ["行为", "源码入口", "关键点"],
        rows: [
          ["设置 Mesh", "`set_mesh()` `mesh_instance_3d.cpp:120`", "调用 `mesh->get_rid()`，再 `set_base()`。"],
          ["换 surface 材质覆盖", "`set_surface_override_material()` `:375`", "提交到 instance，不改 Mesh resource 的 surface material。"],
          ["取最终材质", "`get_active_material()` `:393`", "优先 instance material_override，再 surface override，最后 Mesh surface material。"],
          ["blend shape 权重", "`set_blend_shape_value()` `:177`", "调用 `instance_set_blend_shape_weight()`，权重是实例状态。"],
          ["生成碰撞", "`create_trimesh_collision()` / `create_convex_collision()`", "读取 Mesh 几何生成物理 Shape，不代表渲染 Mesh 自动参与碰撞。"]
        ]
      },
      {
        type: "heading",
        title: "RD 后端如何保存 Mesh"
      },
      {
        type: "paragraph",
        text: "RenderingServer 的 mesh API 在 `rendering_server.h:194` 附近，真正的后端抽象是 `RendererMeshStorage`，声明在 `servers/rendering/storage/mesh_storage.h:37`。RD 实现在 `servers/rendering/renderer_rd/storage_rd/mesh_storage.*`。`MeshStorage::mesh_add_surface()` 在 `mesh_storage.cpp:263` 开始做后端工作：校验 surface data 大小，创建 vertex buffer、attribute buffer、skin buffer、index buffer、LOD index buffer，必要时创建 blend shape storage buffer 和 uniform set。"
      },
      {
        type: "paragraph",
        text: "动态更新不是重新走脚本 Array 槽位，而是直接更新后端 buffer 的某个 byte 区间。`mesh_surface_update_vertex_region()`、`mesh_surface_update_attribute_region()`、`mesh_surface_update_skin_region()`、`mesh_surface_update_index_region()` 分别在 `mesh_storage.cpp:571`、`:584`、`:597`、`:610` 调用 `RD::buffer_update()`。这要求调用方知道 surface 的二进制布局，适合引擎内部或很谨慎的高级代码。"
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：运行时创建一个三角形 Mesh"
      },
      {
        type: "code",
        code: [
          "var arrays := []",
          "arrays.resize(Mesh.ARRAY_MAX)",
          "arrays[Mesh.ARRAY_VERTEX] = PackedVector3Array([",
          "    Vector3(-0.5, 0.0, 0.0),",
          "    Vector3(0.5, 0.0, 0.0),",
          "    Vector3(0.0, 1.0, 0.0),",
          "])",
          "arrays[Mesh.ARRAY_INDEX] = PackedInt32Array([0, 1, 2])",
          "",
          "var mesh := ArrayMesh.new()",
          "mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)",
          "",
          "var instance := MeshInstance3D.new()",
          "instance.mesh = mesh",
          "add_child(instance)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个例子里，`ArrayMesh` 负责几何数据，`MeshInstance3D` 负责把它放进场景。`add_surface_from_arrays()` 会触发 RenderingServer 的 surface data 打包；`instance.mesh = mesh` 会让 instance base 指向 mesh RID。"
      },
      {
        type: "subheading",
        title: "案例二：为什么一个 Mesh 可以被多个实例共享"
      },
      {
        type: "code",
        code: [
          "var shared_mesh := preload(\"res://models/chair.res\")",
          "",
          "for i in 10:",
          "    var chair := MeshInstance3D.new()",
          "    chair.mesh = shared_mesh",
          "    chair.position = Vector3(i * 2.0, 0, 0)",
          "    add_child(chair)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "十个节点共享同一个 Mesh resource 和 mesh RID，但每个 MeshInstance3D 有自己的 instance RID、transform、可见性和可能的材质覆盖。这就是 Resource 和 Node 分离的实际价值。"
      },
      {
        type: "subheading",
        title: "案例三：材质到底从哪里来"
      },
      {
        type: "code",
        code: [
          "# 优先级从高到低：",
          "mesh_instance.material_override",
          "mesh_instance.get_surface_override_material(surface_index)",
          "mesh.surface_get_material(surface_index)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`MeshInstance3D::get_active_material()` 就按这个顺序查。排查“我给 Mesh 设置了材质但场景里没变”时，先看实例有没有 `material_override` 或 surface override。"
      },
      {
        type: "subheading",
        title: "案例四：动态更新顶点为什么更难"
      },
      {
        type: "code",
        code: [
          "# 简单做法：修改 arrays 后重建 surface，适合低频更新。虽然成本高，但不容易错。",
          "mesh.clear_surfaces()",
          "mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)",
          "",
          "# 高级做法：surface_update_vertex_region() 直接更新二进制 buffer。",
          "# 这要求你按当前 surface format 正确打包 PackedByteArray。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "Godot 的底层更新函数操作的是已经打包好的 byte buffer，不是 `PackedVector3Array`。如果只是偶尔变化，重建 surface 更直观；如果要高频更新，必须理解 `mesh_surface_make_offsets_from_format()` 计算出的布局。"
      },
      {
        type: "subheading",
        title: "案例五：导入模型和 PrimitiveMesh 的区别"
      },
      {
        type: "paragraph",
        text: "glTF、FBX、OBJ 等导入路径最终会生成 Mesh/ArrayMesh 资源和材质；`BoxMesh`、`SphereMesh`、`CapsuleMesh` 等 `PrimitiveMesh` 则在运行时按参数生成 surface arrays。两者进入 RenderingServer 后都表现为 mesh RID 和 surfaces，只是资源数据的来源不同。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "Mesh 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Mesh", "几何资源接口、surface、AABB、材质引用、RID。", "不保存节点 transform。"],
          ["ArrayMesh", "可序列化 surface arrays 和 mesh RID。", "不自动管理场景实例。"],
          ["MeshInstance3D", "场景实例、base、材质覆盖、骨骼、blend shape 权重。", "不拥有底层 vertex buffer。"],
          ["VisualInstance3D", "通用 3D 可见实例和 instance RID。", "不关心 Mesh surface 槽位。"],
          ["Material", "surface 或实例的渲染状态和 shader 参数。", "不定义三角形拓扑。"],
          ["RenderingServer mesh RID", "Server 层几何资源句柄。", "不是 ObjectID，也不是 Node。"],
          ["RendererRD MeshStorage", "RD buffer、LOD、AABB、依赖通知和后端资源。", "不处理编辑器导入 UI。"],
          ["TriangleMesh / Shape3D", "碰撞、导航或几何查询使用的派生数据。", "不是渲染 Mesh 本身。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：MeshInstance3D 就是 Mesh。MeshInstance3D 是节点实例，Mesh 是资源。",
          "误区二：移动 Mesh 会改顶点。移动节点只改 instance transform，不改 Mesh surface arrays。",
          "误区三：Mesh 的 material 一定最终生效。实例的 material_override 和 surface override 优先级更高。",
          "误区四：ArrayMesh 的 arrays 是随便摆的普通数组。它必须按 `Mesh.ARRAY_*` 槽位和 PackedArray 类型组织。",
          "误区五：没有索引数组就不能画。可以无索引绘制，只是按顶点顺序解释 primitive。",
          "误区六：改 Mesh 会自动生成碰撞。渲染 Mesh 和 Physics Shape 是两套数据，需要显式生成或配置。",
          "误区七：动态更新顶点等同于改 PackedVector3Array。底层 update region 要写入打包后的 byte buffer。",
          "误区八：mesh RID 是 GPU buffer。mesh RID 是 RenderingServer 句柄，RD buffer 是后端内部资源。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/resources/mesh.h:50`，确认 Mesh 基类定义的 surface、AABB、材质和 RID 接口。",
          "读 `ArrayMesh` 在 `mesh.h:300` 的字段，理解 `surfaces`、`mesh RID`、`aabb`、blend shapes 的分工。",
          "读 `ArrayMesh::_create_if_empty()`、`add_surface_from_arrays()`、`add_surface()`，看资源如何创建 Server mesh RID。",
          "读 `RenderingServer::mesh_create_surface_data_from_arrays()`，理解 surface arrays 如何被校验和打包。",
          "读 `MeshInstance3D::set_mesh()` 和 `_mesh_changed()`，理解 Mesh resource 如何挂到 3D instance 上。",
          "读 `RendererMeshStorage` 抽象，再进入 `RendererRD::MeshStorage::mesh_add_surface()`，看 RD buffer 的创建和更新。",
          "最后按问题选择扩展：材质看 Material，骨骼看 Skeleton/Skin，导入看 glTF/ImporterMesh，碰撞看 `create_trimesh_shape()`。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Mesh 是几何资源，ArrayMesh 把 surface arrays 变成 Server mesh RID，MeshInstance3D 把这个 RID 放进场景，RendererRD 再把每个 surface 变成 GPU 可绘制的 buffer。"
      }
    ]
  },
  {
    id: "shader",
    title: "Shader",
    aliases: ["Shader", "ShaderMaterial", "BaseMaterial3D", "shader code", "gdshader", "shader_type", "render_mode", "uniform", "ShaderInclude", "ShaderPreprocessor", "ShaderLanguage", "ShaderTypes", "ShaderCompiler", "ShaderRD", "MaterialStorage", "shader_set_code", "material_set_shader", "material_set_param", "shader_compile_spirv_from_source", "compile_glslang_shader", "SPIR-V", "glslang"],
    summary: "Godot shader 的分层：Shader 资源保存代码并处理 include，ShaderMaterial 保存参数，MaterialStorage 按 shader_type 创建后端数据，ShaderCompiler/ShaderRD/RenderingDevice 最终生成 GPU shader。",
    article: [
      {
        type: "lead",
        text: "Shader 是 Godot 里描述“材质如何计算颜色、深度、顶点变形、粒子或天空”的代码资源。它不是 Material 本身，也不是 GPU shader 对象本身。`Shader` 资源保存 Godot shader language 源码，`ShaderMaterial` 把 Shader 和 uniform 参数绑定成材质，RenderingServer/MaterialStorage 负责把代码交给对应渲染后端，ShaderCompiler/ShaderRD/RenderingDevice 再生成具体的 GLSL/SPIR-V/驱动 shader。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Shader 想成“上色说明书”。Mesh 决定物体有哪些面，Texture 提供图片，Shader 决定每个像素怎么混合颜色、光照、透明度和特殊效果。Material 则像是把说明书拿来实际使用，并填入参数：颜色是多少、贴图是哪张、粗糙度是多少。"
      },
      {
        type: "paragraph",
        text: "所以一句话区分：`Shader` 是代码，`ShaderMaterial` 是带参数的材质实例，`RenderingDevice shader` 是后端真正能给 GPU 使用的编译结果。你在脚本里改 `shader.code` 是改源码；改 `material.set_shader_parameter()` 是改参数；两者走的引擎路径不一样。"
      },
      {
        type: "flow",
        title: "从 .gdshader 到 GPU shader",
        steps: [
          { title: "ResourceLoader", text: "`ResourceFormatLoaderShader` 读取 `.gdshader` 文本并创建 Shader 资源。" },
          { title: "Shader.set_code", text: "预处理 include，判断 `shader_type`，保存源码并触发 changed。" },
          { title: "Shader.get_rid", text: "需要 RID 时创建 RenderingServer shader RID，并提交预处理后的代码。" },
          { title: "ShaderMaterial", text: "绑定 shader RID 和 uniform 参数，得到 material RID。" },
          { title: "MaterialStorage", text: "按 shader type 创建 2D/3D/sky/fog/particles 后端 ShaderData/MaterialData。" },
          { title: "ShaderCompiler / ShaderRD", text: "把 Godot shader language 翻译成后端 GLSL stage，再编译成 SPIR-V 和 RD shader。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Shader` 类声明在 `scene/resources/shader.h:39`，继承 `Resource`。核心字段包括 `shader_rid`、`preprocessed_code`、`mode`、`include_dependencies`、源码 `code`、`include_path` 和默认贴图参数。`Mode` 包括 spatial、canvas_item、particles、sky、fog、texture_blit，对应不同渲染路径和内建变量集合。"
      },
      {
        type: "paragraph",
        text: "`Shader::set_code()` 在 `shader.cpp:87`。它先断开旧 include 依赖，保存原始源码，再用 `ShaderPreprocessor` 处理 `#include`，并记录 `ShaderInclude` 依赖。随后调用 `ShaderLanguage::get_shader_type(preprocessed_code)` 判断模式：`canvas_item`、`particles`、`sky`、`fog`、`texture_blit` 或默认 `spatial`。如果 shader RID 已经存在，会调用 `RenderingServer::shader_set_code()` 更新后端。"
      },
      {
        type: "paragraph",
        text: "Shader RID 是懒创建的。`Shader::_check_shader_rid()` 在 `shader.cpp:50`：只有当 `get_rid()`、uniform 列表或材质真正需要后端资源时，才调用 `RenderingServer::shader_create_from_code(preprocessed_code, get_path())`。这避免刚加载资源时立刻编译所有 shader。"
      },
      {
        type: "table",
        title: "Shader 分层边界",
        headers: ["层级", "典型类型/函数", "负责什么", "不负责什么"],
        rows: [
          ["文本资源", "`Shader`、`.gdshader`", "保存 Godot shader language 源码、include 依赖、shader mode、默认贴图。", "不保存每个材质实例的参数值。"],
          ["材质实例", "`ShaderMaterial`", "绑定 Shader、保存 uniform 参数缓存、创建 material RID。", "不解析 shader 语法本身。"],
          ["Server API", "`shader_create`、`shader_set_code`、`material_set_shader`", "给场景层和资源层提供 shader/material RID 接口。", "不实现具体编译器。"],
          ["后端存储", "`RendererRD::MaterialStorage`", "按 shader type 创建 ShaderData/MaterialData，维护参数、默认贴图、material update queue。", "不关心资源文件如何加载。"],
          ["语言编译", "`ShaderLanguage`、`ShaderCompiler`", "解析 Godot shader language，校验内建变量、render modes、uniform，并生成后端 stage 代码。", "不直接创建 GPU pipeline。"],
          ["RD 编译", "`ShaderRD`、`RenderingDevice`、`glslang`", "生成 shader variants，把 GLSL stage 编译为 SPIR-V/bytecode/RD shader。", "不管理编辑器 Inspector 参数 UI。"]
        ]
      },
      {
        type: "heading",
        title: "ShaderMaterial 如何绑定参数"
      },
      {
        type: "paragraph",
        text: "`ShaderMaterial::set_shader()` 在 `scene/resources/material.cpp:387`。它保存 `Ref<Shader>`，取 `shader->get_rid()`，再把 shader RID 提交给 `RenderingServer::material_set_shader()`。在编辑器中还会连接 shader 的 changed 信号，用于更新 Inspector 的 shader_parameter 属性列表。"
      },
      {
        type: "paragraph",
        text: "`ShaderMaterial::set_shader_parameter()` 在 `material.cpp:419`。普通 Variant 参数会直接通过 `material_set_param()` 下发；对象参数如果是 Texture，会先转成 texture RID 再下发。`_get_property_list()` 会调用 `shader->get_shader_uniform_list()`，把 shader 里的 uniform 映射成 `shader_parameter/<name>` 属性，这就是 Inspector 能自动显示 shader 参数的原因。"
      },
      {
        type: "table",
        title: "Shader / Material 常用操作",
        headers: ["操作", "源码入口", "实际影响"],
        rows: [
          ["设置 shader 源码", "`Shader::set_code()`", "预处理 include，判断 mode，必要时更新 shader RID。"],
          ["取 shader RID", "`Shader::get_rid()`", "懒创建 RenderingServer shader RID。"],
          ["绑定到材质", "`ShaderMaterial::set_shader()`", "把 shader RID 设置到 material RID。"],
          ["设置 uniform", "`ShaderMaterial::set_shader_parameter()`", "写入参数缓存并调用 `material_set_param()`。"],
          ["生成参数列表", "`Shader::get_shader_uniform_list()`", "向 RenderingServer 询问 uniform 列表，并过滤默认贴图。"],
          ["默认贴图", "`Shader::set_default_texture_parameter()`", "给 shader uniform 配默认 texture RID。"]
        ]
      },
      {
        type: "heading",
        title: "后端如何编译"
      },
      {
        type: "paragraph",
        text: "`RenderingServerDefault::shader_create_from_code()` 在 `rendering_server_default.h:288`：分配 shader RID，初始化 shader，设置 path hint，再 `shader_set_code()`。RD 后端的 `MaterialStorage::shader_set_code()` 在 `renderer_rd/storage_rd/material_storage.cpp:2201`：先用 `ShaderLanguage::get_shader_type()` 得到 shader type，再按 type 创建对应 ShaderData，例如 3D shader 会由 forward clustered/mobile renderer 注册的数据工厂创建。"
      },
      {
        type: "paragraph",
        text: "`ShaderCompiler::compile()` 在 `servers/rendering/shader_compiler.cpp:1526`。它从 `ShaderTypes` 取得当前 shader mode 的函数、render modes、stencil modes 和合法 shader types，然后调用 `ShaderLanguage::compile()` 解析源码。成功后，ShaderCompiler 会把 AST 转成后端可用的 stage 代码和 uniform 信息；失败时会重建 include 文件上下文并报告具体文件和行号。"
      },
      {
        type: "paragraph",
        text: "`ShaderRD::version_set_code()` 在 `renderer_rd/shader_rd.cpp:815`，它保存每个 shader version 的 uniforms、vertex globals、fragment globals 和变体 defines。真正编译 stage 时，`ShaderRD::compile_stages()` 在 `:1166` 调用 `RenderingDevice::shader_compile_spirv_from_source()`；后者在 `rendering_device.cpp:228` 通过 glslang 模块的 `compile_glslang_shader()` 生成 SPIR-V。glslang 模块初始化在 `modules/glslang/register_types.cpp:145`。"
      },
      {
        type: "flow",
        title: "一次 Shader 编译的内部路径",
        steps: [
          { title: "ShaderLanguage::get_shader_type", text: "从源码中读 `shader_type`，决定 2D/3D/sky/fog/particles 路径。" },
          { title: "MaterialStorage::shader_set_code", text: "按 type 创建或替换 ShaderData，并通知拥有它的 Material 更新。" },
          { title: "ShaderCompiler::compile", text: "解析 Godot shader language，校验内建变量和 render_mode，生成 stage 代码。" },
          { title: "ShaderRD::version_set_code", text: "把 stage code、uniforms、defines 记录到一个可变体化的 shader version。" },
          { title: "RenderingDevice::shader_compile_spirv_from_source", text: "调用 glslang 编译 GLSL 到 SPIR-V。" },
          { title: "RD shader / pipeline", text: "后续 render pipeline 用编译后的 RD shader、vertex format、framebuffer format 等创建 pipeline。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：创建一个最小 3D ShaderMaterial"
      },
      {
        type: "code",
        code: [
          "var shader := Shader.new()",
          "shader.code = \"\"\"",
          "shader_type spatial;",
          "uniform vec4 tint : source_color = vec4(1.0, 0.2, 0.1, 1.0);",
          "",
          "void fragment() {",
          "    ALBEDO = tint.rgb;",
          "}",
          "\"\"\"",
          "",
          "var material := ShaderMaterial.new()",
          "material.shader = shader",
          "material.set_shader_parameter(\"tint\", Color.CORNFLOWER_BLUE)",
          "$MeshInstance3D.material_override = material"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`shader.code` 会走 `Shader::set_code()`；`material.shader = shader` 会让 ShaderMaterial 创建/绑定 material RID；`set_shader_parameter()` 下发的是 uniform 参数，不会重新解析整个 shader。"
      },
      {
        type: "subheading",
        title: "案例二：改源码和改参数的成本不同"
      },
      {
        type: "code",
        code: [
          "# 改参数：通常只更新 material uniform，适合频繁变化。",
          "material.set_shader_parameter(\"tint\", Color.RED)",
          "",
          "# 改源码：会触发 shader 重新解析/编译，别每帧做。",
          "shader.code = shader.code.replace(\"ALBEDO\", \"EMISSION\")"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "排查性能问题时先看自己改的是 uniform 还是 shader code。uniform 是材质数据更新；shader code 可能导致 ShaderData 重建、变体重新编译和 pipeline 重新准备。"
      },
      {
        type: "subheading",
        title: "案例三：include 改了为什么主 Shader 会变"
      },
      {
        type: "code",
        code: [
          "// res://common_color.gdshaderinc",
          "vec3 apply_tint(vec3 color, vec3 tint) {",
          "    return color * tint;",
          "}",
          "",
          "// res://main.gdshader",
          "shader_type spatial;",
          "#include \"res://common_color.gdshaderinc\"",
          "uniform vec3 tint = vec3(1.0);",
          "void fragment() { ALBEDO = apply_tint(ALBEDO, tint); }"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`Shader::set_code()` 会记录 `ShaderInclude` 依赖并连接 changed 信号。include 资源变化时 `_dependency_changed()` 会重新 `set_code(get_code())`，因此主 shader 会重新预处理和编译。"
      },
      {
        type: "subheading",
        title: "案例四：uniform 在 Inspector 里怎么出现"
      },
      {
        type: "paragraph",
        text: "ShaderMaterial 的 `_get_property_list()` 会从 shader RID 获取 uniform 列表，并把每个 uniform 变成 `shader_parameter/<name>`。如果 uniform 类型改了，ShaderMaterial 会检查缓存类型是否兼容；不兼容时用 `shader_get_parameter_default()` 重置默认值。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "Shader 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Shader", "Godot shader language 源码、mode、include 依赖和默认贴图。", "不保存每个材质实例的参数值。"],
          ["ShaderMaterial", "绑定 Shader 和 uniform 参数，生成 material RID。", "不实现 shader parser。"],
          ["BaseMaterial3D", "用属性生成内置 3D shader 和材质状态。", "不是用户手写 shader 源码的唯一入口。"],
          ["ShaderInclude", "可复用 include 代码和依赖通知。", "不能独立成为可绘制材质。"],
          ["ShaderLanguage", "词法、语法、类型、内建变量、uniform 和 render_mode 校验。", "不直接生成 RD pipeline。"],
          ["ShaderCompiler", "把 Godot shader AST 转成后端 stage 代码和 uniform 信息。", "不负责资源加载和 Inspector。"],
          ["ShaderRD", "管理 shader variants、version、stage 编译和缓存。", "不理解 Godot 资源路径语义。"],
          ["RenderingDevice shader", "后端 GPU shader/bytecode 资源。", "不保存 `shader_parameter/*` 这样的材质属性。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Shader 就是 Material。Shader 是代码，Material 是带参数和渲染状态的资源。",
          "误区二：改 uniform 会重编译 shader。通常不会；重编译主要来自改 shader code、include 或后端变体。",
          "误区三：`shader_type` 只是注释。它决定使用 2D、3D、particles、sky、fog 等不同内建变量和后端路径。",
          "误区四：Shader RID 就是 RD shader。Shader RID 是 RenderingServer 层句柄，RD shader 是后端内部编译结果。",
          "误区五：所有 shader 都一开始加载时编译。Shader RID 懒创建，后端也可能按 variant 延迟编译。",
          "误区六：贴图 uniform 可以直接传 Object 到后端。ShaderMaterial 会把 Texture 对象转为 texture RID。",
          "误区七：include 是简单文本替换。资源层会跟踪 ShaderInclude 依赖，变化后触发重新预处理。",
          "误区八：shader 报错只看主文件行号。ShaderCompiler 会重建 include 上下文，真正错误可能在 include 文件。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/resources/shader.h:39`，确认 Shader 资源字段和 Mode 枚举。",
          "读 `ResourceFormatLoaderShader::load()`，理解 `.gdshader` 如何加载为 Shader 资源。",
          "读 `Shader::set_code()`，看 include 预处理、mode 判断和 changed 信号。",
          "读 `ShaderMaterial::set_shader()`、`set_shader_parameter()`，理解代码资源和材质参数的边界。",
          "读 `RenderingServerDefault::shader_create_from_code()` 和 `MaterialStorage::shader_set_code()`，看 RID 如何进入后端。",
          "读 `ShaderTypes` 和 `ShaderLanguage::compile()`，理解内建变量、函数和 render_mode 的来源。",
          "读 `ShaderCompiler::compile()`、`ShaderRD::version_set_code()`、`RenderingDevice::shader_compile_spirv_from_source()`，追到 SPIR-V 编译。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Shader 是源码资源，ShaderMaterial 是参数化材质，MaterialStorage 按 shader_type 创建后端数据，ShaderCompiler/ShaderRD/RenderingDevice 才把它变成 GPU 可执行的 shader。"
      }
    ]
  },
  {
    id: "renderthread",
    title: "Render Thread",
    aliases: ["Render Thread", "render thread", "RenderingServer::sync", "RenderingServer::draw", "RenderingServerDefault::sync", "RenderingServerDefault::draw", "RenderingServerDefault::_draw", "CommandQueueMT", "command_queue", "server_thread", "create_thread", "separate_thread_render", "rendering/driver/threads/thread_model", "--render-thread separate", "flush_all", "push_and_sync", "call_on_render_thread"],
    summary: "Godot 渲染线程模型：主线程提交 RenderingServer API，安全/单线程模式直接 flush，Separate 模式把命令排入 CommandQueueMT，由渲染线程执行 `_draw()` 和资源操作。",
    article: [
      {
        type: "lead",
        text: "Render Thread 不是一个单独的渲染系统，而是 RenderingServerDefault 的执行模型。主线程或其他线程调用 RenderingServer API；如果启用了 separate render thread，很多调用不会立刻执行后端操作，而是写入 `CommandQueueMT`，由渲染线程 flush。每帧 `Main::iteration()` 会先 `RenderingServer::sync()`，再按需要 `RenderingServer::draw()`，这两个点决定“命令何时真的落到渲染后端”。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把主线程想成“写任务单的人”，渲染线程想成“真正搬东西和画画的人”。脚本说“把这个 Mesh 的材质改了”，主线程可能只是把这件事写进队列；渲染线程稍后统一拿队列里的任务执行。这样主线程不用在每次改属性时都直接碰底层图形 API。"
      },
      {
        type: "paragraph",
        text: "但这也意味着不能随便假设“我刚调用 RenderingServer，后端立刻完成了”。如果代码需要结果，必须用同步调用或等到 `sync()`；如果某个操作必须在渲染线程上做，Godot 提供 `call_on_render_thread()` 这样的入口。"
      },
      {
        type: "flow",
        title: "一帧里的渲染同步点",
        steps: [
          { title: "脚本/节点改状态", text: "CanvasItem、VisualInstance3D、Resource 等调用 RenderingServer API。" },
          { title: "命令入队或直接执行", text: "同线程时直接执行；separate render thread 时写入 `CommandQueueMT`。" },
          { title: "Main::iteration sync", text: "`RenderingServer::sync()` 等待前面命令和上一帧绘制关键点完成。" },
          { title: "是否需要 draw", text: "根据窗口可绘制、render loop、pending resources、low processor usage 和 `has_changed()` 判断。" },
          { title: "RenderingServer::draw", text: "发出 `frame_pre_draw`，清零 changes，然后执行或排队 `_draw()`。" },
          { title: "RenderingServerDefault::_draw", text: "更新 scene/canvas/particles/probes/viewports，提交后端并 present。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`RenderingServerDefault` 的线程相关字段在 `servers/rendering/rendering_server_default.h:80` 附近：`CommandQueueMT command_queue`、`Thread::ID server_thread`、`WorkerThreadPool::TaskID server_task_id`、`bool create_thread`。`create_thread` 为 true 时，RenderingServerDefault 会把渲染 Server 的执行放到 WorkerThreadPool 的 pump task 上。"
      },
      {
        type: "paragraph",
        text: "启动配置来自命令行和项目设置。`main.cpp:1562` 处理 `--render-thread safe/separate`；`main.cpp:2749` 读取 `rendering/driver/threads/thread_model`，并把结果写入 `OS::_separate_thread_render`。编辑器和项目管理器会强制关闭 separate render thread；无线程构建也会关闭。创建 RenderingServer 的位置在 `main.cpp:3540`，参数是 `OS::get_singleton()->is_separate_thread_rendering_enabled()`。"
      },
      {
        type: "paragraph",
        text: "`RenderingServerDefault::init()` 在 `rendering_server_default.cpp:276`。如果 `create_thread` 为 true，它会释放当前显示线程上下文，启动名为 `Rendering Server pump task` 的 WorkerThreadPool 任务，设置 command queue 的 pump task id，然后用 `push_and_sync()` 调用 `_init()`。否则 `server_thread` 就是主线程，直接 `_init()`。"
      },
      {
        type: "table",
        title: "Safe 和 Separate 的区别",
        headers: ["模式", "执行方式", "关键源码", "读源码时注意"],
        rows: [
          ["Safe / 非 separate", "RenderingServer 和后端大多在主线程直接执行，其他线程来的命令由 `sync()` flush。", "`create_thread == false`，`sync()` 调 `command_queue.flush_all()`。", "容易读，但主线程和渲染工作更耦合。"],
          ["Separate render thread", "主线程调用 API 多数入队，渲染线程循环 `command_queue.flush_all()`。", "`init()` 启动 WorkerThreadPool task，`_thread_loop()` flush 队列。", "要区分调用发生在哪个线程、真正执行发生在哪个线程。"],
          ["必须返回值的 API", "使用 `push_and_ret()` 或同步路径。", "`server_wrap_mt_common.h` 中一系列宏。", "会阻塞调用方，滥用会破坏并行收益。"],
          ["必须同步完成的 API", "使用 `push_and_sync()`。", "`CommandQueueMT::push_and_sync()`。", "常见于初始化、资源创建顺序和跨线程安全边界。"],
          ["普通写操作", "使用 `push()`，稍后执行。", "`ASYNC_COND_PUSH` 相关宏。", "不能马上假设后端状态已更新。"]
        ]
      },
      {
        type: "heading",
        title: "CommandQueueMT 做了什么"
      },
      {
        type: "paragraph",
        text: "`CommandQueueMT` 在 `core/templates/command_queue_mt.h`。它把“对象指针 + 成员函数 + 参数”放进一段 byte buffer。`push()` 只入队；`push_and_sync()` 入队后等待命令执行；`push_and_ret()` 还会把返回值写回调用方。`flush_all()` 会逐条取出命令并调用真实方法；遇到 sync 命令时推进 `sync_head` 并唤醒等待者。"
      },
      {
        type: "paragraph",
        text: "这个设计解释了 RenderingServerDefault 里大量宏的意义：在非渲染线程调用时，API 可以被封装为队列命令；在渲染线程本身调用时，则可以直接执行或先 `flush_if_pending()` 保证顺序。例如 `shader_create_from_code()`、`material_create_from_shader()`、`mesh_create_from_surfaces()` 都会根据 `Thread::get_caller_id() == server_thread` 和 `can_create_resources_async()` 决定直接执行还是入队。"
      },
      {
        type: "flow",
        title: "CommandQueueMT 同步命令流程",
        steps: [
          { title: "push_and_sync", text: "调用方把命令写入 `command_mem`，递增 `sync_tail`。" },
          { title: "通知 pump task", text: "如果有渲染线程任务，调用 `notify_yield_over()` 唤醒它。" },
          { title: "调用方等待", text: "`_wait_for_sync()` 等到 `sync_head` 追上目标值。" },
          { title: "渲染线程 flush", text: "`_thread_loop()` 或 `sync()` 调 `flush_all()`，逐条执行命令。" },
          { title: "命令完成", text: "sync 命令执行后递增 `sync_head` 并唤醒等待者。" }
        ]
      },
      {
        type: "heading",
        title: "每帧 draw 具体做什么"
      },
      {
        type: "paragraph",
        text: "`Main::iteration()` 在 `main.cpp:5051` 先调用 `RenderingServer::sync()`，注释说明这是为了同步仍在绘制的上一帧。随后根据窗口是否可绘制、是否有 pending RD resources、render loop 是否启用、low processor usage 和 `has_changed()` 判断是否调用 `draw()`。"
      },
      {
        type: "paragraph",
        text: "`RenderingServerDefault::draw()` 在 `rendering_server_default.cpp:443`。它要求从主线程触发，先发 `frame_pre_draw` 信号，再把静态 `changes` 清零；separate 模式下把 `_draw(p_present, frame_step)` 入队，否则直接调用 `_draw()`。`_draw()` 在 `:76`，依次调用 `rasterizer->begin_frame()`、`scene->update()`、`canvas->update()`、`particles_storage->update_particles()`、`scene->render_probes()`、`viewport->draw_viewports()`、`canvas_render->update()`、`rasterizer->end_frame()`，最后处理可见性通知、post draw、GPU profile 和内存信息。"
      },
      {
        type: "table",
        title: "关键函数速查",
        headers: ["函数/字段", "位置", "作用"],
        rows: [
          ["`create_thread`", "`rendering_server_default.h:85`", "是否使用 separate render thread。"],
          ["`command_queue`", "`rendering_server_default.h:80`", "跨线程传递渲染命令。"],
          ["`RenderingServerDefault::init()`", "`rendering_server_default.cpp:276`", "启动渲染线程或直接初始化后端。"],
          ["`_thread_loop()`", "`rendering_server_default.cpp:415`", "渲染线程循环 yield 并 `flush_all()`。"],
          ["`sync()`", "`rendering_server_default.cpp:435`", "等待或 flush 所有已提交命令。"],
          ["`draw()`", "`rendering_server_default.cpp:443`", "主线程触发一帧渲染，必要时把 `_draw()` 入队。"],
          ["`_draw()`", "`rendering_server_default.cpp:76`", "真正更新 scene/canvas/viewport 并提交/present。"],
          ["`redraw_request()`", "`rendering_server_default.h:103`", "增加 `changes`，让低功耗模式知道需要重绘。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：为什么刚设置材质不一定立刻进后端"
      },
      {
        type: "code",
        code: [
          "mesh_instance.set_surface_override_material(0, material)",
          "",
          "# 场景层调用 RenderingServer API。",
          "# Separate render thread 模式下，后端 material/instance 更新可能先进入 command queue。",
          "# 真正执行通常发生在 sync/draw 或渲染线程 flush 时。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "读 bug 时要问两个问题：调用点有没有执行？如果执行了，它是直接执行后端，还是只是入队？很多“刚设置后读不到”的问题，本质是跨线程队列还没 flush。"
      },
      {
        type: "subheading",
        title: "案例二：需要在渲染线程执行的操作"
      },
      {
        type: "code",
        code: [
          "RenderingServer::get_singleton()->call_on_render_thread(my_callable);"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`call_on_render_thread()` 在 `rendering_server_default.h:1217`：如果当前已经在 `server_thread`，先 flush pending 再直接调用；否则把 `_call_on_render_thread()` 入队。XRServer 中不少世界缩放、参考帧更新就通过这个边界提交到渲染线程。"
      },
      {
        type: "subheading",
        title: "案例三：低功耗模式为什么有时不 draw"
      },
      {
        type: "paragraph",
        text: "在 `Main::iteration()` 中，如果开启 low processor usage 且没有强制重绘，Godot 只有在 `RenderingServer::has_changed()` 为 true 时才 draw。`RenderingServerDefault::redraw_request()` 会增加 `changes`；Canvas/Scene/RD 某些动态状态会调用它。排查编辑器或运行时“没刷新”时，`changes` 是否被触发是一个入口。"
      },
      {
        type: "subheading",
        title: "案例四：资源创建顺序为什么要同步"
      },
      {
        type: "paragraph",
        text: "创建 shader、material、mesh 等 RID 时，如果当前在渲染线程，代码常先 `command_queue.flush_if_pending()` 再直接执行；如果在其他线程，则把初始化、设置代码、设置 path 等步骤按顺序 push 到队列。这样可以避免后端看到“material 先绑定了还没初始化的 shader”这类顺序问题。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "Render Thread 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Render Thread", "决定 RenderingServer 命令在哪个线程执行。", "不改变渲染算法本身。"],
          ["RenderingServer", "对场景层公开渲染 API。", "不保证每个 API 都同步完成。"],
          ["CommandQueueMT", "跨线程保存命令、同步等待、flush 执行。", "不理解 Mesh、Shader 的业务含义。"],
          ["RendererRD", "实际渲染后端和资源存储。", "不决定主线程何时提交命令。"],
          ["RenderingDevice", "底层 GPU 抽象和命令提交。", "不是 Godot 的跨线程 API 队列。"],
          ["DisplayServer", "窗口和平台 present/surface。", "不管理 scene/canvas 命令队列。"],
          ["MessageQueue", "Object/SceneTree 层的延迟调用。", "不是 RenderingServer 的渲染命令队列。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Render Thread 是另一个 Renderer。不是，它是 RenderingServerDefault 的线程执行模式。",
          "误区二：调用 RenderingServer API 就等于后端立即完成。Separate 模式下很多调用只是入队。",
          "误区三：`sync()` 就是画一帧。`sync()` 是等待/flush 命令；`draw()` 才触发 `_draw()`。",
          "误区四：`draw()` 可以任意线程调用。源码明确要求从主线程触发。",
          "误区五：CommandQueueMT 和 MessageQueue 一样。一个是渲染 Server 命令队列，一个是对象/场景层延迟调用队列。",
          "误区六：Separate 一定更快。它降低主线程阻塞的可能，但同步点、驱动限制和资源创建仍可能阻塞。",
          "误区七：Editor 也会默认 separate。源码里编辑器和项目管理器会强制关闭 separate render thread。",
          "误区八：读渲染 bug 只看 `_draw()`。很多状态在 `_draw()` 前已经通过队列提交，必须回看调用点和 flush 点。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `main.cpp:1562` 和 `main.cpp:2749`，理解命令行和项目设置如何决定 separate render thread。",
          "读 `RenderingServerDefault` 的字段：`command_queue`、`server_thread`、`create_thread`。",
          "读 `init()`、`_assign_mt_ids()`、`_thread_loop()`，看渲染线程如何启动和 flush。",
          "读 `CommandQueueMT::push()`、`push_and_sync()`、`push_and_ret()`、`flush_all()`，理解命令队列语义。",
          "读 `Main::iteration()` 中 `RenderingServer::sync()` 和 `draw()` 的调用条件。",
          "读 `RenderingServerDefault::_draw()`，把 scene/canvas/viewport/rasterizer 的每帧顺序串起来。",
          "最后回到具体资源 API，比如 shader/material/mesh 创建，看它们何时直接执行、何时入队。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Render Thread 解决的是渲染命令在哪个线程执行：主线程提交状态，CommandQueueMT 维护顺序和同步，`sync()` 等待命令完成，`draw()` 才真正推动一帧渲染。"
      }
    ]
  },
  {
    id: "physicsserver",
    title: "PhysicsServer2D / PhysicsServer3D",
    aliases: ["PhysicsServer2D", "PhysicsServer3D", "PhysicsServer", "Godot Physics", "Jolt Physics", "PhysicsServer3DManager", "PhysicsServer2DManager", "PhysicsServer3DWrapMT", "PhysicsServer2DWrapMT", "Space", "Body", "Area", "Shape", "Joint", "DirectSpaceState", "PhysicsDirectSpaceState3D", "PhysicsDirectBodyState3D", "body_test_motion", "move_and_slide", "move_and_collide", "CollisionObject3D", "PhysicsBody3D", "RigidBody3D", "CharacterBody3D", "Area3D", "CollisionShape3D", "sync", "end_sync", "flush_queries", "step", "fixed physics step"],
    summary: "PhysicsServer2D/3D 是场景物理节点和真实物理后端之间的 Server/RID 边界：节点提交 Space、Body、Area、Shape、Joint，Godot Physics 或 Jolt 在固定物理步里同步、查询和求解。",
    article: [
      {
        type: "lead",
        text: "PhysicsServer2D / PhysicsServer3D 是 Godot 物理系统的统一接口。场景层的 RigidBody、StaticBody、Area、CollisionShape、CharacterBody 等节点负责表达“这是玩家、墙、触发区、碰撞形状”；PhysicsServer 把这些信息变成 RID 管理的 Space、Body、Area、Shape、Joint；真正的 broadphase、窄相碰撞、约束求解和运动查询交给 Godot Physics 或 Jolt 后端。读物理源码时，先把这条边界看清楚，比一开始钻碰撞算法更有效。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把场景里的物理节点想成“物理世界的标签和控制面板”：RigidBody3D 是会被力推动的物体，Area3D 是检测进入/离开的区域，CollisionShape3D 是形状。PhysicsServer 就像后台账本，它不关心节点在编辑器里怎么摆，只记录一个个可计算的 body、area、shape 和 space。"
      },
      {
        type: "paragraph",
        text: "Godot Physics 或 Jolt 则像真正负责计算的人。每个固定物理步开始时，Godot 把节点改过的状态同步给 Server；脚本的 `_physics_process()` 和查询读取这个物理世界；随后 Server 结束同步并调用后端 `step()`，后端才真正算出碰撞、速度、接触和刚体位置。"
      },
      {
        type: "flow",
        title: "从节点到物理后端",
        steps: [
          { title: "用户摆节点", text: "RigidBody3D、Area3D、CollisionShape3D、CharacterBody3D 保存场景语义。" },
          { title: "节点拿 RID", text: "PhysicsBody3D 构造时 `body_create()`；Area3D 构造时 `area_create()`。" },
          { title: "Shape 接进 Body/Area", text: "CollisionShape3D 的 Shape3D Resource 通过 `body_add_shape()` 或 `area_add_shape()` 进入 Server。" },
          { title: "加入 Space", text: "CollisionObject3D 进入树时设置 space、transform、collision layer/mask。" },
          { title: "固定物理步同步", text: "`Main::iteration()` 调 PhysicsServer 的 `sync()`，脚本和查询看到同步点状态。" },
          { title: "后端求解", text: "`end_sync()` 后调用 `step()`，Godot Physics 或 Jolt 计算碰撞和积分。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`PhysicsServer2D` 和 `PhysicsServer3D` 都继承 `Object`，但它们不是普通节点。它们定义一套抽象物理 API：space、area、body、shape、joint、query、direct state、固定步生命周期。3D 声明入口在 `servers/physics_3d/physics_server_3d.h:236`，2D 声明入口在 `servers/physics_2d/physics_server_2d.h:213`。这些头文件是契约，具体执行在模块后端。"
      },
      {
        type: "paragraph",
        text: "PhysicsServer 的核心设计仍然是 RID。场景节点不会直接持有后端 body 对象指针，而是保存一个 RID；后端用 RID_Owner 或自己的映射表找到真实对象。这个模型让场景层、脚本 API、Godot Physics、Jolt、GDExtension 后端可以共享同一套 Server 接口。"
      },
      {
        type: "table",
        title: "PhysicsServer 分层边界",
        headers: ["层级", "代表代码", "负责什么", "不负责什么"],
        rows: [
          ["场景节点层", "`scene/2d/physics`、`scene/3d/physics`", "暴露 RigidBody、Area、CollisionShape、CharacterBody 等用户语义，维护 Node 生命周期和属性。", "不直接做 broadphase、窄相碰撞和刚体求解。"],
          ["Shape Resource", "`scene/resources/2d`、`scene/resources/3d`", "保存 BoxShape、SphereShape、ConcavePolygonShape 等资源数据，并提供 shape RID。", "不是场景节点，也不决定 body 是否启用。"],
          ["PhysicsServer API", "`servers/physics_2d`、`servers/physics_3d`", "定义 space/body/area/shape/joint/query 的统一接口和脚本绑定。", "不固定使用 Godot Physics 还是 Jolt。"],
          ["后端实现", "`modules/godot_physics_*`、`modules/jolt_physics`", "创建真实 Space/Body/Shape，执行查询、接触生成、约束求解和 step。", "不应该改场景节点 API。"],
          ["帧调度", "`main/main.cpp:4965` 到 `5016`", "决定 `sync()`、`physics_process()`、Navigation、`end_sync()`、`step()` 的顺序。", "不实现具体碰撞算法。"]
        ]
      },
      {
        type: "heading",
        title: "核心 RID 类型"
      },
      {
        type: "table",
        title: "PhysicsServer 管的对象",
        headers: ["类型", "直观理解", "源码入口", "读源码时的关键点"],
        rows: [
          ["Space", "一个独立的物理世界，通常对应 World2D/World3D 的 physics space。", "`space_create()`：3D 在 `physics_server_3d.h:289`，2D 在 `physics_server_2d.h:264`。", "body、area、query 都必须属于某个 space；不同 space 之间不会互相碰撞。"],
          ["Shape", "可被碰撞检测使用的几何体，比如 box、sphere、capsule。", "3D shape create 声明在 `physics_server_3d.h:265` 附近；Godot Physics 3D 实现在 `godot_physics_server_3d.cpp:47` 起。", "Shape3D Resource 和 Server shape RID 要分开看。"],
          ["Body", "参与物理的实体：static、kinematic、rigid、character。", "`body_create()`：3D 在 `physics_server_3d.h:401`，2D 在 `physics_server_2d.h:370`。", "节点属性最终会变成 body mode、state、space、layer/mask、shape 列表。"],
          ["Area", "用于检测进入/离开、覆盖重力/阻尼、做监控的区域。", "`area_create()`：3D 在 `physics_server_3d.h:337`，2D 在 `physics_server_2d.h:309`。", "Area 有 shape 和 space，但语义不同于刚体；它偏向检测和影响。"],
          ["Joint", "约束两个 body 的关系，比如 pin、hinge、slider。", "PhysicsServer3D/2D 的 joint API。", "Joint 是求解器约束，不是 Node 层父子关系。"],
          ["DirectSpaceState", "脚本和节点做 ray/point/shape 查询的视图。", "3D 抽象在 `physics_server_3d.h:125`，Godot Physics 3D 实现在 `godot_space_3d.h:41`。", "它查询当前同步点的物理空间；不是另一个世界，也不自动移动 body。"],
          ["body_test_motion", "给一个 body 和运动向量，测试运动路径会不会撞到东西。", "3D 声明在 `physics_server_3d.h:573`，Godot Physics 3D 实现在 `godot_physics_server_3d.cpp:940`。", "CharacterBody 的 `move_and_slide()` 依赖它；这是运动测试，不是刚体积分。"]
        ]
      },
      {
        type: "heading",
        title: "固定物理步"
      },
      {
        type: "paragraph",
        text: "物理不是在每个 setter 里即时完整求解。Godot 在 `Main::iteration()` 里按固定 tick 推动物理。当前源码里，3D `sync()` 在 `main/main.cpp:4965`，2D `sync()` 在 `4971`；脚本物理回调之后，3D `end_sync()` 和 `step()` 在 `5009`、`5010`，2D 在 `5015`、`5016`。中间还有 NavigationServer 的 physics_process。"
      },
      {
        type: "flow",
        title: "一次固定物理步的顺序",
        steps: [
          { title: "准备固定 tick", text: "Main 根据 `physics_ticks_per_second` 计算固定步长，可能一帧跑多次。" },
          { title: "PhysicsServer sync", text: "3D/2D Server 同步节点提交的状态，WrapMT 模式会处理命令队列边界。" },
          { title: "SceneTree physics_process", text: "用户 `_physics_process()`、CharacterBody 移动、RayCast 更新、查询代码运行。" },
          { title: "Navigation physics_process", text: "导航避障等系统在物理节奏中推进。" },
          { title: "PhysicsServer end_sync", text: "关闭脚本可安全读写的同步窗口，准备让后端求解。" },
          { title: "PhysicsServer step", text: "Godot Physics 或 Jolt 真正执行碰撞、约束、积分、接触回调准备。" }
        ]
      },
      {
        type: "table",
        title: "同步函数怎么读",
        headers: ["函数", "位置", "作用", "容易误解的点"],
        rows: [
          ["`sync()`", "`PhysicsServer3DWrapMT::sync()` 在 `physics_server_3d_wrap_mt.cpp:68`；Godot Physics 3D 在 `godot_physics_server_3d.cpp:1692`。", "进入物理同步阶段，处理跨线程队列，让节点和 Server 状态对齐。", "它不是求解本身。"],
          ["`flush_queries()`", "抽象声明在 `physics_server_3d.h:817`、`physics_server_2d.h:614`。", "刷新查询/回调相关数据，后端可在这里把查询结果和监控信号准备好。", "不要把它当成固定步入口。"],
          ["`end_sync()`", "3D 抽象在 `physics_server_3d.h:818`；Jolt 3D 实现在 `jolt_physics_server_3d.cpp:1641`。", "结束同步窗口，阻止脚本在后端 step 时同时改结构。", "看到晚一帧，先查 end_sync 和 step 的相对顺序。"],
          ["`step()`", "3D 抽象在 `physics_server_3d.h:815`；Godot Physics 3D 入口在 `godot_physics_server_3d.cpp:1674`；Jolt 3D 在 `jolt_physics_server_3d.cpp:1623`。", "真正推进一个固定物理步。", "属性 setter 常只是改状态；求解通常在 step。"]
        ]
      },
      {
        type: "heading",
        title: "后端选择"
      },
      {
        type: "paragraph",
        text: "PhysicsServer 的抽象允许 Godot 换后端而不改场景节点 API。Godot Physics 3D 在 `modules/godot_physics_3d/register_types.cpp:56` 注册 `GodotPhysics3D` 并设为默认；Jolt 在 `modules/jolt_physics/register_types.cpp:59` 注册 `Jolt Physics`。`PhysicsServer3DManager` 的声明在 `physics_server_3d.h:1005`，注册和创建默认后端在 `physics_server_3d.cpp:1184`、`1218`。2D 也有对应 manager。"
      },
      {
        type: "table",
        title: "后端和包装层",
        headers: ["组件", "代表源码", "用途"],
        rows: [
          ["Godot Physics 2D/3D", "`modules/godot_physics_2d`、`modules/godot_physics_3d`", "Godot 内置物理后端，2D/3D 都有；3D 的 `body_create()` 在 `godot_physics_server_3d.cpp:442`。"],
          ["Jolt Physics", "`modules/jolt_physics`", "3D 物理替代后端，实现同一套 `PhysicsServer3D` 接口；`body_create()` 在 `jolt_physics_server_3d.cpp:530`。"],
          ["PhysicsServer Manager", "`PhysicsServer3DManager`、`PhysicsServer2DManager`", "记录可用后端、项目设置、默认后端和创建回调。"],
          ["Extension", "`physics_server_2d_extension.h`、`physics_server_3d_extension.h`", "让 GDExtension 后端实现物理 Server 虚函数。"],
          ["WrapMT", "`physics_server_3d_wrap_mt.cpp`、`physics_server_2d_wrap_mt.cpp`", "在线程模式下把调用排入 `CommandQueueMT`，并在 `sync()`/`step()` 附近处理队列。"],
          ["Dummy", "`physics_server_3d_dummy.h`、`physics_server_2d_dummy.h`", "无真实物理后端时提供空实现，保证引擎结构可运行。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：RigidBody3D 是怎样进入 Server 的"
      },
      {
        type: "code",
        code: [
          "PhysicsBody3D::PhysicsBody3D(PhysicsServer3D::BodyMode p_mode) :",
          "    CollisionObject3D(PhysicsServer3D::get_singleton()->body_create(), false) {",
          "    PhysicsServer3D::get_singleton()->body_set_mode(get_rid(), p_mode);",
          "}",
          "",
          "RigidBody3D::RigidBody3D() :",
          "    PhysicsBody3D(PhysicsServer3D::BODY_MODE_RIGID) {",
          "    PhysicsServer3D::get_singleton()->body_set_state_sync_callback(get_rid(), callable_mp(this, &RigidBody3D::_body_state_changed));",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`scene/3d/physics/physics_body_3d.cpp:57` 显示 PhysicsBody3D 构造时先创建 body RID；`rigid_body_3d.cpp:829` 显示 RigidBody3D 再设置状态同步回调。也就是说，RigidBody3D 的 Node 身份和后端刚体不是同一个对象，中间靠 RID 和回调连接。"
      },
      {
        type: "subheading",
        title: "案例二：CollisionShape3D 不是直接参与求解"
      },
      {
        type: "paragraph",
        text: "`CollisionShape3D` 是 Node，用来挂在场景树里；真正的几何体来自 Shape3D Resource 的 RID。`collision_shape_3d.cpp:90` 和 `:223` 会调用 `shape_owner_add_shape()`；`collision_object_3d.cpp:616` 之后把 shape RID 通过 `area_add_shape()` 或 `body_add_shape()` 提交给 Server。排查“形状没生效”时，要同时看 CollisionShape 节点、Shape 资源、shape owner、body/area 是否在 space 里。"
      },
      {
        type: "subheading",
        title: "案例三：CharacterBody3D 的 move_and_slide 不是刚体模拟"
      },
      {
        type: "code",
        code: [
          "bool PhysicsBody3D::move_and_collide(...) {",
          "    bool colliding = PhysicsServer3D::get_singleton()->body_test_motion(get_rid(), p_parameters, &r_result);",
          "    ...",
          "}",
          "",
          "bool CharacterBody3D::move_and_slide() {",
          "    // 多次 move_and_collide，结合 floor/wall/ceiling 判断裁剪运动。",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`PhysicsBody3D::move_and_collide()` 在 `physics_body_3d.cpp:112` 调 `body_test_motion()`；`CharacterBody3D::move_and_slide()` 在 `character_body_3d.cpp:43` 开始，内部多次调用 move/collide 流程并判断地板、墙、天花板。它更像“按查询结果裁剪玩家位移”，不是让刚体求解器自由积分。"
      },
      {
        type: "subheading",
        title: "案例四：Ray 查询只是读当前同步点"
      },
      {
        type: "code",
        code: [
          "var hit = get_world_3d().direct_space_state.intersect_ray(query)",
          "",
          "# 这会查询当前 physics space。",
          "# 它不会移动 body，也不会立刻强制后端 step。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "脚本里的 `direct_space_state.intersect_ray()` 最终落到 `PhysicsDirectSpaceState3D` 的绑定，3D 绑定在 `physics_server_3d.cpp:490`，Godot Physics 3D 查询实现在 `godot_space_3d.cpp:110`，Jolt 3D 在 `jolt_physics_direct_space_state_3d.cpp:464`。如果查询结果看起来晚一帧，通常要回到固定物理步同步点，而不是先怀疑 ray 算法。"
      },
      {
        type: "subheading",
        title: "案例五：Jolt 和 Godot Physics 切换时 API 不变"
      },
      {
        type: "paragraph",
        text: "项目把 3D 物理后端从 Godot Physics 切到 Jolt 时，RigidBody3D、Area3D、CharacterBody3D 的场景 API 不需要换一套。变的是 Manager 创建的 `PhysicsServer3D` 实现：Godot Physics 的 `body_test_motion()` 在 `godot_physics_server_3d.cpp:940`，Jolt 的 `body_test_motion()` 在 `jolt_physics_server_3d.cpp:958`，但上层仍调用同一个抽象函数。"
      },
      {
        type: "subheading",
        title: "案例六：多线程物理包装为什么会影响时机"
      },
      {
        type: "paragraph",
        text: "`PhysicsServer3DWrapMT::step()` 在 `physics_server_3d_wrap_mt.cpp:60`，创建线程时会把真实 `step()` push 到命令队列；`sync()` 在 `:68`，必要时 `push_and_sync()` 或 `flush_all()`。这解释了为什么读多线程物理 bug 时，不能只看后端函数，还要看调用是不是经过 WrapMT。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "PhysicsServer 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Node / SceneTree", "管理节点生命周期、树结构、通知、脚本回调。", "不负责物理 broadphase 和求解。"],
          ["CollisionObject2D/3D", "把节点 transform、space、layer/mask、shape owner 同步到 PhysicsServer。", "不是具体物理后端对象。"],
          ["Shape2D/Shape3D Resource", "保存可复用形状资源，并持有 shape RID。", "不决定某个 body 是否参与碰撞。"],
          ["PhysicsServer2D/3D", "统一管理 space/body/area/shape/joint/query API。", "不绑定某一个后端实现。"],
          ["Godot Physics / Jolt", "实现碰撞检测、约束求解、运动查询和 step。", "不管理场景树。"],
          ["NavigationServer", "处理导航地图、路径和避障。", "不是碰撞物理求解器，但在固定物理步附近推进。"],
          ["RenderingServer", "用 RID 管可见对象和渲染资源。", "不参与物理碰撞。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：RigidBody3D 节点本身就是后端刚体。实际是节点持有 body RID，后端另有真实对象。",
          "误区二：CollisionShape3D 就是碰撞几何体。它是场景节点，几何体来自 Shape3D Resource 和 Server shape RID。",
          "误区三：所有物理 setter 都会立即求解。多数只是改 Server 状态，真正求解在固定物理步的 `step()`。",
          "误区四：DirectSpaceState 是另一个物理世界。它只是当前 space 的查询视图。",
          "误区五：CharacterBody 和 RigidBody 走同一种运动逻辑。CharacterBody 主要靠 motion test 裁剪运动；RigidBody 由后端积分和约束求解。",
          "误区六：切换 Jolt 会让上层 API 改名。上层仍调用 PhysicsServer3D 抽象，只是后端实现变了。",
          "误区七：只看 `step()` 就能解释所有物理问题。进入 space、shape 同步、layer/mask、sync/end_sync、query 时机同样关键。",
          "误区八：NavigationServer 属于物理碰撞后端。它在物理节奏附近推进，但职责是导航和避障，不是刚体碰撞求解。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `servers/physics_3d/physics_server_3d.h:236` 和 `servers/physics_2d/physics_server_2d.h:213`，把抽象接口和 RID 类型列出来。",
          "读 `scene/3d/physics/physics_body_3d.cpp:57`、`area_3d.cpp:818`，看节点怎样创建 body/area RID。",
          "读 `scene/3d/physics/collision_object_3d.cpp:41`、`:145`、`:616`，看 space、transform、layer/mask、shape 如何提交。",
          "读 `scene/3d/physics/collision_shape_3d.cpp:83`，确认 CollisionShape3D 和 Shape3D Resource 的边界。",
          "读 `main/main.cpp:4965` 到 `5016`，建立 `sync()`、`physics_process()`、`end_sync()`、`step()` 的时间线。",
          "读 `modules/godot_physics_3d/godot_physics_server_3d.cpp:442`、`:940`、`:1674`，看 Godot Physics 后端 body、motion test、step。",
          "再读 `modules/jolt_physics/jolt_physics_server_3d.cpp:530`、`:958`、`:1623`，对比 Jolt 如何实现同一套接口。",
          "最后读 `servers/physics_3d/physics_server_3d_wrap_mt.cpp:60`、`:68`，理解多线程包装怎样改变执行时机。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "PhysicsServer2D/3D 是物理系统的契约层：场景节点用 RID 提交 space/body/area/shape，固定物理步决定同步和求解时机，Godot Physics 或 Jolt 负责真正计算。"
      }
    ]
  },
  {
    id: "audioserver",
    title: "AudioServer",
    aliases: ["AudioServer", "AudioDriver", "AudioDriverManager", "AudioStream", "AudioStreamPlayback", "AudioStreamPlayer", "AudioStreamPlayer2D", "AudioStreamPlayer3D", "AudioSamplePlayback", "AudioBusLayout", "AudioEffect", "AudioEffectInstance", "bus", "Master bus", "mix bus", "audio mix", "_mix_step", "_driver_process", "audio_server_process", "start_playback_stream", "stop_playback_stream", "playback_list", "mix_count", "mix_rate", "sample playback"],
    summary: "AudioServer 是 Godot 的音频混音中心：播放节点只负责发起播放和控制参数，AudioServer 维护 playback、bus、effect 和混音缓冲，AudioDriver 在线程或平台回调中请求混音并把结果送到系统设备。",
    article: [
      {
        type: "lead",
        text: "AudioServer 是 Godot 声音系统的运行时中心。`AudioStreamPlayer`、`AudioStreamPlayer2D`、`AudioStreamPlayer3D` 是场景节点，负责给用户提供 play、stop、bus、volume、pitch、空间化等 API；`AudioStream` 是资源，负责创建 `AudioStreamPlayback`；`AudioServer` 接管活跃 playback、bus、effect、音量路由和混音缓冲；`AudioDriver` 则把混好的 PCM 数据交给 WASAPI、XAudio2、ALSA、WebAudio 等平台后端。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把一首声音想成“乐器”，AudioStreamPlayer 是你按下播放键的人，AudioServer 是调音台，AudioDriver 是音箱和声卡。你在节点上点 `play()`，并不是节点自己把声音送到系统设备；它只是把“我要播放这个音频、走哪个 bus、音量多少”告诉 AudioServer。"
      },
      {
        type: "paragraph",
        text: "AudioServer 每隔很短一段时间把所有正在播放的声音混在一起：先把每个 playback 混到对应 bus，再跑 effect，再按 send 路由到 Master，最后把 Master 的采样写给 AudioDriver。AudioDriver 运行在自己的线程或平台音频回调里，持续向 AudioServer 要下一小块音频缓冲。"
      },
      {
        type: "flow",
        title: "一段声音从节点到设备",
        steps: [
          { title: "AudioStreamPlayer.play", text: "场景节点检查 stream，创建 AudioStreamPlayback。" },
          { title: "AudioServer.start_playback_stream", text: "播放对象进入 AudioServer 的 playback_list，记录 bus、volume、pitch 和开始时间。" },
          { title: "AudioDriver 请求混音", text: "驱动线程调用 `AudioDriver::audio_server_process()`。" },
          { title: "AudioServer._driver_process", text: "按设备需要的 frames 取 Master bus 缓冲；不够时触发 `_mix_step()`。" },
          { title: "AudioServer._mix_step", text: "混合 stream、处理 bus/effect/send、更新峰值和状态。" },
          { title: "平台输出", text: "驱动把 int32 或转换后的采样提交给系统音频 API。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`AudioServer` 的类入口在 `servers/audio/audio_server.h:180`。它继承 `Object`，对脚本暴露 bus、effect、设备、播放速度、输入设备、播放控制等 API；但真正实时混音相关字段在私有区：`buffer_size`、`mix_count`、`to_mix`、`mix_buffer`、`temp_buffer`、`buses`、`bus_map`、`playback_list`。这些字段解释了为什么音频代码读起来不像渲染/物理的 RID Server：音频核心不是用 RID 查对象，而是围绕实时安全的播放链表和混音缓冲运行。"
      },
      {
        type: "paragraph",
        text: "`AudioDriver` 的抽象在 `audio_server.h:48`，`AudioDriverManager` 在 `:159` 管理可用驱动。`AudioDriverManager::initialize()` 在 `audio_server.cpp:227`，会按选择或注册顺序初始化驱动，成功后把它设为 singleton，失败时可落到 Dummy driver。驱动线程不是每帧由 `Main::iteration()` 推动，而是按设备缓冲节奏不断调用 `AudioDriver::audio_server_process()`。"
      },
      {
        type: "table",
        title: "音频系统分层边界",
        headers: ["层级", "代表源码", "负责什么", "不负责什么"],
        rows: [
          ["AudioStream Resource", "`servers/audio/audio_stream.h`、`scene/resources/audio_stream_wav.*`、`modules/vorbis`、`modules/mp3`", "保存或解码音频资源，创建 `AudioStreamPlayback`。", "不管理场景节点，也不直接输出到设备。"],
          ["AudioStreamPlayback", "`servers/audio/audio_stream.h:70`", "真正提供 `start()`、`stop()`、`seek()`、`mix()` 的播放实例。", "不是 Node；不能自己决定 bus 路由。"],
          ["AudioStreamPlayer 节点", "`scene/audio`、`scene/2d`、`scene/3d`", "暴露用户 API，管理 autoplay、polyphony、bus、volume、pitch、2D/3D 空间化参数。", "不直接向平台音频设备写缓冲。"],
          ["AudioServer", "`servers/audio/audio_server.h:180`、`audio_server.cpp`", "维护 playback list、bus、effect、send、mix buffer、sample playback、设备查询和混音。", "不实现具体系统 API。"],
          ["AudioDriver", "`servers/audio/audio_server.h:48`、`drivers/*/audio_driver_*`", "初始化设备、开线程或回调、请求混音、提交采样。", "不理解场景节点和编辑器属性。"],
          ["平台音频 API", "WASAPI、XAudio2、ALSA、CoreAudio、WebAudio 等", "把 PCM 送到系统设备或浏览器音频图。", "不认识 Godot 的 AudioStreamPlayer。"]
        ]
      },
      {
        type: "heading",
        title: "播放链路"
      },
      {
        type: "paragraph",
        text: "`AudioStreamPlayer::play()` 在 `scene/audio/audio_stream_player.cpp:109`。它先调用 `AudioStreamPlayerInternal::play_basic()`，后者在 `audio_stream_player_internal.cpp:140` 检查节点是否在树里、处理 monophonic、调用 `stream->instantiate_playback()`，并把 playback 放进节点自己的 `stream_playbacks`。随后 `AudioStreamPlayer::play()` 在 `:114` 调 `AudioServer::start_playback_stream()`。"
      },
      {
        type: "table",
        title: "播放对象的生命周期",
        headers: ["阶段", "关键函数", "发生了什么", "注意点"],
        rows: [
          ["创建 playback", "`AudioStreamPlayerInternal::play_basic()`", "从 AudioStream 实例化 `AudioStreamPlayback`，设置参数，记录到节点内部列表。", "节点必须在 scene tree 内，否则播放会失败。"],
          ["交给 AudioServer", "`AudioServer::start_playback_stream()` 在 `audio_server.cpp:1239`、`:1248`", "创建 `AudioStreamPlaybackListNode`，调用 playback `start()`，记录 bus volume、pitch、滤波参数和 lookahead buffer。", "从这一步开始混音状态进入 AudioServer。"],
          ["混音中", "`AudioServer::_mix_step()` 在 `audio_server.cpp:352`", "音频线程遍历 `playback_list`，调用 playback `mix()` 取样本，并混到目标 bus。", "不要在这里做会阻塞或分配不受控的操作。"],
          ["暂停", "`AudioServer::set_playback_paused()` 在 `audio_server.cpp:1409`", "状态从 PLAYING 进入 FADE_OUT_TO_PAUSE，再由音频线程转成 PAUSED。", "用短淡出避免爆音。"],
          ["停止/删除", "`AudioServer::stop_playback_stream()` 在 `audio_server.cpp:1289`", "状态进入 FADE_OUT_TO_DELETION，音频线程混完淡出后标记等待删除。", "真正释放在主线程清理列表时做，避免实时线程释放复杂对象。"],
          ["Sample 播放", "`start_sample_playback()` 在 `audio_server.cpp:1997`", "可采样资源可走驱动级 sample playback。", "不支持 sample 的驱动会回退或警告。"]
        ]
      },
      {
        type: "flow",
        title: "AudioServer 混音循环",
        steps: [
          { title: "驱动拿锁", text: "平台驱动线程进入自己的 lock，并调用 `audio_server_process()`。" },
          { title: "_driver_process", text: "递增 `mix_count`，按设备请求的 frames 从 Master bus 拷贝到输出 buffer。" },
          { title: "需要新缓冲", text: "`to_mix == 0` 时调用 `_mix_step()` 生成一块内部混音缓冲。" },
          { title: "播放流混到 bus", text: "遍历 `playback_list`，调用 `AudioStreamPlayback::mix()`，按 bus/volume 混入 channel buffer。" },
          { title: "bus effect/send", text: "从后往前处理 bus effect，再按 send 路由到上游 bus，最终汇入 Master。" },
          { title: "输出 Master", text: "`_driver_process()` 把 Master bus 转成 int32 PCM，驱动再提交给系统 API。" }
        ]
      },
      {
        type: "heading",
        title: "bus、effect 和 send"
      },
      {
        type: "paragraph",
        text: "AudioServer 的 bus 系统就是 Godot 的调音台。`set_bus_count()` 在 `audio_server.cpp:794`，`add_bus()` 在 `:873`，`add_bus_effect()` 在 `:1119`，`set_bus_layout()` 在 `:1754`。Master bus 永远是 0 号；其他 bus 可以 send 到另一个 bus，如果 send 不存在或形成不合法顺序，就回到 Master。"
      },
      {
        type: "table",
        title: "bus 字段速查",
        headers: ["字段/概念", "作用", "源码线索"],
        rows: [
          ["name / bus_map", "通过名称找到 bus，并允许节点使用字符串选择输出路径。", "`thread_find_bus_index()` 在 `audio_server.cpp:776`，找不到则回 Master。"],
          ["channels", "每个输出通道有自己的 AudioFrame buffer、peak_volume 和 effect instance。", "`init_channels_and_buffers()` 在 `audio_server.cpp:1508`。"],
          ["volume_db", "bus 总音量，混音后转线性值乘到 buffer。", "`_mix_step()` 在 `audio_server.cpp:629` 附近应用。"],
          ["solo / mute", "solo 模式只保留 solo 链；mute 让 bus 音量变 0。", "`_mix_step()` 开始会标记 solo 链。"],
          ["bypass", "跳过当前 bus 的 effect 链。", "`_mix_step()` 处理 effect 前检查 `bus->bypass`。"],
          ["effects", "AudioEffect 资源会实例化为每个通道的 AudioEffectInstance。", "`_update_bus_effects()` 在 `audio_server.cpp:1106`。"],
          ["send", "把当前 bus 输出加到另一个 bus；Master 没有 send。", "`_mix_step()` 从后往前处理 bus，保证 send 顺序。"]
        ]
      },
      {
        type: "heading",
        title: "驱动线程和实时边界"
      },
      {
        type: "paragraph",
        text: "音频和画面不同：它不等主循环一帧一帧来推，而是设备持续要数据。`AudioDriver::audio_server_process()` 在 `audio_server.cpp:72`，它更新 mix 时间后调用 `AudioServer::_driver_process()`。XAudio2 的线程例子在 `drivers/xaudio2/audio_driver_xaudio2.cpp:82`，线程持锁、统计时间、调用 `audio_server_process()`，再把采样提交给 XAudio2。Dummy driver 也能展示最小结构：`audio_driver_dummy.cpp:37` 初始化，`:56` 线程循环，`:113` 手动混音。"
      },
      {
        type: "table",
        title: "实时线程要特别小心的点",
        headers: ["问题", "AudioServer 的做法", "为什么重要"],
        rows: [
          ["播放停止产生爆音", "停止和暂停先进入 FADE_OUT 状态，再由 `_mix_step()` 淡出。", "直接截断波形会产生 click/pop。"],
          ["bus 参数跨线程变化", "新的 `AudioStreamPlaybackBusDetails` 用原子指针替换，旧对象放进 graveyard 延后清理。", "音频线程不能在读结构时被主线程释放内存。"],
          ["effect 实例需要通道隔离", "`_update_bus_effects()` 为每个 bus channel 建 effect instance。", "滤波器、混响、压缩器都有内部状态，不能乱共享。"],
          ["设备缓冲长度和内部混音块不同", "`_driver_process()` 用 `to_mix` 从内部 512-frame 块中分段拷贝。", "设备一次要的 frames 可能不等于 AudioServer 内部 buffer_size。"],
          ["声道数变化", "`_driver_process()` 检查 `channel_count != get_channel_count()` 并重新初始化 bus buffer。", "切换输出设备可能改变 stereo/5.1/7.1。"],
          ["读取播放进度", "通过 AudioServer 或 AudioDriver 的时间函数读取，不直接猜主线程帧数。", "音频线程节奏独立于 `Main::iteration()`。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：普通 AudioStreamPlayer 播放一段声音"
      },
      {
        type: "code",
        code: [
          "void AudioStreamPlayer::play(float p_from_pos) {",
          "    Ref<AudioStreamPlayback> stream_playback = internal->play_basic();",
          "    if (stream_playback.is_null()) {",
          "        return;",
          "    }",
          "    AudioServer::get_singleton()->start_playback_stream(",
          "        stream_playback, internal->bus, _get_volume_vector(), p_from_pos, internal->pitch_scale);",
          "    internal->ensure_playback_limit();",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这段在 `scene/audio/audio_stream_player.cpp:109`。最关键的是：节点没有混音，它只是把 playback 和参数提交给 AudioServer。`ensure_playback_limit()` 处理最大复音数，超出的旧 playback 会停止。"
      },
      {
        type: "subheading",
        title: "案例二：为什么停止不是立刻删除"
      },
      {
        type: "paragraph",
        text: "`stop_playback_stream()` 不直接在调用线程释放播放节点，而是把状态改成 `FADE_OUT_TO_DELETION`。`_mix_step()` 在混完淡出后调用 `_delete_stream_playback_list_node()`。这符合实时音频的基本约束：音频线程要稳定地产生下一块样本，不能因为主线程 stop 而突然释放正在读的对象。"
      },
      {
        type: "subheading",
        title: "案例三：3D 声音为什么会影响多个 bus/channel"
      },
      {
        type: "paragraph",
        text: "`AudioStreamPlayer3D` 会根据监听器、距离、衰减、声道布局和空间化结果更新每个 playback 的 bus volume；最后仍然通过 `AudioServer::set_playback_bus_volumes_linear()` 或相关路径把这些音量交给 AudioServer。`_mix_step_for_channel()` 会把左右声道音量从上一次平滑插值到这一次，并可应用 high-shelf/filter 参数，避免空间化参数突变产生爆音。"
      },
      {
        type: "subheading",
        title: "案例四：换 bus 名字后声音去哪了"
      },
      {
        type: "paragraph",
        text: "混音线程用 `thread_find_bus_index()` 从 `bus_map` 找 bus；如果找不到，返回 0，也就是 Master。排查“声音没有走到预期 bus”时，先看节点的 bus 名、AudioServer 的 bus layout、bus 是否被重命名、send 是否有效，而不是先查音频文件本身。"
      },
      {
        type: "subheading",
        title: "案例五：平台驱动怎样要音频数据"
      },
      {
        type: "code",
        code: [
          "void AudioDriverXAudio2::thread_func(void *p_udata) {",
          "    ad->lock();",
          "    ad->audio_server_process(ad->buffer_size, ad->samples_in);",
          "    ad->unlock();",
          "    // samples_in -> samples_out -> SubmitSourceBuffer(...)",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "XAudio2 线程在 `drivers/xaudio2/audio_driver_xaudio2.cpp:82`。不同平台的驱动代码会不同，但共同模式是：设备线程或回调锁住音频系统，向 AudioServer 要一段混好的样本，再提交给平台 API。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "AudioServer 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["AudioStream", "音频资源和解码/播放实例创建。", "不管理场景树和设备输出。"],
          ["AudioStreamPlayback", "一次播放的状态、seek、mix。", "不决定节点生命周期。"],
          ["AudioStreamPlayer", "用户可见节点 API：play/stop/bus/volume/pitch/autoplay。", "不实现设备线程。"],
          ["AudioServer", "playback list、bus、effect、mix buffer、sample playback、设备信息。", "不调用 WASAPI/XAudio2/ALSA 的具体 API。"],
          ["AudioDriver", "平台设备初始化、线程或回调、请求混音、提交 PCM。", "不处理 Node 属性和编辑器 bus 面板。"],
          ["Main::iteration", "游戏主循环和场景回调。", "不是音频混音节拍；音频由驱动持续拉取。"],
          ["ResourceLoader", "加载 `.wav`、`.ogg`、`.mp3` 等资源。", "不负责实时混音。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：AudioStreamPlayer 自己把声音输出到设备。它只是创建 playback 并提交给 AudioServer。",
          "误区二：AudioStream 就是正在播放的声音。AudioStream 是资源，AudioStreamPlayback 才是一次播放实例。",
          "误区三：音频每帧跟着 Main::iteration 混一次。实际是 AudioDriver 按设备缓冲节奏拉取混音。",
          "误区四：stop 应该立即释放对象。实时音频里通常要淡出并延迟清理，避免爆音和线程读悬空对象。",
          "误区五：bus 只是一个名字。bus 有 buffer、effect、send、volume、solo/mute、peak meter，是调音台的一条完整通道。",
          "误区六：找不到 bus 就应该静音。Godot 混音线程找不到 bus 时回 Master，这能避免无声但可能掩盖配置错误。",
          "误区七：effect 只有一个全局实例。AudioServer 为每个 bus channel 创建 effect instance，保留独立状态。",
          "误区八：设备驱动失败就一定没有 AudioServer。AudioDriverManager 可以落到 Dummy driver，AudioServer 结构仍然存在。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/audio/audio_stream_player.cpp:109`，确认节点 play 如何提交到 AudioServer。",
          "读 `scene/audio/audio_stream_player_internal.cpp:140`，看 playback 如何由 AudioStream 实例化并被节点保存。",
          "读 `servers/audio/audio_stream.h:70`，理解 AudioStreamPlayback 的 `start/stop/seek/mix` 契约。",
          "读 `servers/audio/audio_server.h:180`，把 AudioServer 的 bus、playback list、buffer 字段圈出来。",
          "读 `AudioServer::start_playback_stream()`、`stop_playback_stream()`、`set_playback_bus_volumes_linear()`，理解主线程怎样改播放状态。",
          "读 `AudioDriver::audio_server_process()` 和 `AudioServer::_driver_process()`，确认驱动如何拉取混音。",
          "读 `AudioServer::_mix_step()` 和 `_mix_step_for_channel()`，看 stream、bus、effect、send、peak、fade 的核心循环。",
          "最后读一个平台驱动，比如 `drivers/xaudio2/audio_driver_xaudio2.cpp:82` 或 `servers/audio/audio_driver_dummy.cpp:56`，把线程和设备提交补齐。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "AudioServer 是 Godot 的实时调音台：节点发起播放，AudioStreamPlayback 产出样本，AudioServer 按 bus/effect/send 混成 Master，AudioDriver 再把结果送到系统设备。"
      }
    ]
  },
  {
    id: "inputsystem",
    title: "Input / InputMap / InputEvent",
    aliases: ["Input", "InputMap", "InputEvent", "InputEventKey", "InputEventMouseButton", "InputEventMouseMotion", "InputEventJoypadButton", "InputEventJoypadMotion", "InputEventAction", "parse_input_event", "Input::parse_input_event", "_input", "_shortcut_input", "_unhandled_key_input", "_unhandled_input", "gui_input", "accept_event", "Viewport::push_input", "Viewport::_gui_input_event", "SceneTree::_call_input_pause", "action_press", "action_release", "is_action_pressed", "is_action_just_pressed", "DisplayServer input"],
    summary: "Godot 输入系统分四层：平台/DisplayServer 产生 InputEvent，Input 更新全局状态和 action cache，InputMap 判断事件是否匹配 action，Viewport/Control/SceneTree 决定事件被 GUI 或脚本谁先处理。",
    article: [
      {
        type: "lead",
        text: "Godot 的输入系统不是一个函数从平台直接调到脚本，而是一条分层链路：平台 DisplayServer 把系统按键、鼠标、触摸、手柄事件转换成 `InputEvent`；`Input::parse_input_event()` 更新全局按键、鼠标、手柄和 action 状态；`InputMap` 负责把事件和 action 名字匹配起来；Window/Viewport 再按 `_input`、GUI、shortcut、unhandled 的顺序分发。排查输入 bug 时，必须先问事件在哪一层被改写、缓存、消费或转发。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把输入系统想成“快递分拣”。平台层把系统消息打包成包裹，也就是 InputEvent；Input 像总登记处，记录现在有哪些键按下、鼠标在哪里、哪些 action 正处于 pressed；InputMap 像地址簿，告诉你“这个按键是不是 jump”；Viewport 和 Control 像收件路线，决定这个包裹先给 UI，还是继续给游戏脚本。"
      },
      {
        type: "paragraph",
        text: "所以 `Input.is_action_pressed(\"jump\")` 和 `_input(event)` 不是一回事。前者查的是全局状态缓存，后者收到的是本次事件对象。GUI 按钮如果接受了事件，后面的 `_unhandled_input()` 就可能收不到它，但 `Input` 的按键状态仍然可能已经更新。"
      },
      {
        type: "flow",
        title: "输入事件主链路",
        steps: [
          { title: "平台事件", text: "Win32、Cocoa、浏览器、Android 等平台收到系统输入。" },
          { title: "DisplayServer 转换", text: "平台 DisplayServer 创建 InputEventKey/Mouse/Touch/Joypad 等对象。" },
          { title: "Input.parse_input_event", text: "更新全局 pressed、mouse、joypad、action cache，并处理事件累积/模拟。" },
          { title: "Window 回调", text: "DisplayServer 的 input_event_callback 把事件交给对应 Window。" },
          { title: "Viewport.push_input", text: "本地坐标转换，先 `_input`，再 GUI，再 shortcut/unhandled。" },
          { title: "脚本或 GUI 处理", text: "Control 可 `accept_event()`；未消费事件继续到 `_unhandled_input()` 等回调。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Input` 的类入口在 `core/input/input.h:78`。它是全局单例，主要保存运行时状态：按下的 key、physical key、key label、鼠标按钮 mask、鼠标位置和速度、触摸速度、手柄按钮/轴、action_states、buffered_events、手柄映射、鼠标模式函数指针等。`Input` 不是事件分发器本身，它更像“输入状态机 + 平台到 Viewport 的派发入口”。"
      },
      {
        type: "paragraph",
        text: "`InputEvent` 的基类在 `core/input/input_event.h:52`，它是 Resource，派生出 Key、MouseButton、MouseMotion、JoypadButton、JoypadMotion、ScreenTouch、ScreenDrag、Action、Shortcut、Gesture、MIDI 等。事件对象不仅保存数据，还提供 `is_action()`、`is_action_pressed()`、`get_action_strength()`、`xformed_by()`、`accumulate()` 这类方法。"
      },
      {
        type: "paragraph",
        text: "`InputMap` 的类入口在 `core/input/input_map.h:40`。它维护 `action -> events + deadzone` 的映射，`event_is_action()` 在 `input_map.cpp:279`，核心匹配在 `event_get_action_status()`。它不会把事件送给节点，也不会保存“现在 jump 是否按下”；这些运行时状态在 Input 的 action_states 里。"
      },
      {
        type: "table",
        title: "输入系统分层边界",
        headers: ["层级", "代表源码", "负责什么", "不负责什么"],
        rows: [
          ["平台/DisplayServer", "`platform/*/display_server_*`、`servers/display/display_server.h:394`", "把系统消息转换成 InputEvent，并通过窗口输入回调派发。", "不直接调用用户脚本 `_input()`。"],
          ["InputEvent", "`core/input/input_event.h:52`", "保存一次具体事件的数据，并提供 action 匹配、文本描述、坐标变换、累积能力。", "不保存全局按键状态。"],
          ["Input", "`core/input/input.h:78`、`input.cpp:1519`", "维护全局输入状态、action cache、鼠标/触摸/手柄状态，处理事件缓冲和模拟。", "不决定 GUI 控件是否消费事件。"],
          ["InputMap", "`core/input/input_map.h:40`、`input_map.cpp:279`", "定义 action 到具体事件的匹配规则和 deadzone。", "不负责事件分发和节点回调。"],
          ["Window/Viewport", "`scene/main/window.cpp:1452`、`viewport.cpp:3501`", "把 InputEvent 转本地坐标并按输入阶段分发。", "不维护全局 action pressed 缓存。"],
          ["Control/SceneTree", "`control.cpp:2407`、`scene_tree.cpp:1429`", "GUI 优先处理、accept_event、调用 `_shortcut_input` 和 `_unhandled_input`。", "不创建平台事件。"]
        ]
      },
      {
        type: "heading",
        title: "Input::parse_input_event 做什么"
      },
      {
        type: "paragraph",
        text: "`Input::parse_input_event()` 在 `core/input/input.cpp:1519`。它有三种路径：开启 accumulated input 时先放入 `buffered_events` 并尝试 `accumulate()`，开启 agile flushing 时也先缓冲，否则直接进入 `_parse_input_event_impl()`。真正更新状态和派发的函数是 `_parse_input_event_impl()`，入口在 `input.cpp:801`。源码里明确断言最终交付必须发生在主线程。"
      },
      {
        type: "table",
        title: "Input 更新的状态",
        headers: ["事件类型", "Input 做什么", "对应源码线索"],
        rows: [
          ["InputEventKey", "更新 `keys_pressed`、`physical_keys_pressed`、`key_label_pressed`，忽略 echo 对 pressed 集合的影响。", "`input.cpp:801` 后的 Key 分支。"],
          ["InputEventMouseButton", "更新 mouse button mask、鼠标全局位置，并可模拟 touch。", "`input.cpp:839` 附近。"],
          ["InputEventMouseMotion", "更新鼠标位置、relative/screen_relative 速度，并可模拟 drag。", "`input.cpp:871` 附近。"],
          ["InputEventScreenTouch/Drag", "维护触摸速度，按项目设置模拟鼠标事件。", "`input.cpp:899`、`:955` 附近。"],
          ["InputEventJoypadButton/Motion", "更新手柄按钮集合和轴值。", "`input.cpp:982`、`:991`。"],
          ["action 状态", "遍历 InputMap action，更新每个 action 的 per-device 状态、strength、raw_strength、pressed/released frame。", "`input.cpp:1008` 到 `1048`。"],
          ["最终派发", "如果有 `event_dispatch_function`，把事件交给 DisplayServer/Window 注册的派发函数。", "`input.cpp:1049` 附近。"]
        ]
      },
      {
        type: "flow",
        title: "Viewport 分发顺序",
        steps: [
          { title: "Window._window_input", text: "窗口收到 DisplayServer 回调后调用 `push_input()`。" },
          { title: "Viewport.push_input", text: "检查禁用状态、转本地坐标、更新鼠标悬停和子窗口转发。" },
          { title: "_input", text: "先对 `_vp_input*` 组调用 `_input(event)`。" },
          { title: "GUI", text: "如果还没 handled，进入 `_gui_input_event()`，Control 可接受事件。" },
          { title: "shortcut", text: "仍未 handled 时，Key/Shortcut/JoypadButton 进入 `_shortcut_input()`。" },
          { title: "unhandled", text: "最后进入 `_unhandled_key_input()` 和 `_unhandled_input()`；仍未 handled 才可能做物理拾取。" }
        ]
      },
      {
        type: "heading",
        title: "Viewport 和 GUI 消费"
      },
      {
        type: "paragraph",
        text: "`Viewport::push_input()` 在 `scene/main/viewport.cpp:3501`。顺序写在源码注释里：`_input -> gui input -> _unhandled input`。它先调用 `SceneTree::_call_input_pause(input_group, CALL_INPUT_TYPE_INPUT, ...)`，然后如果事件未处理就调用 `_gui_input_event()`，再未处理才进入 `_push_unhandled_input_internal()`。"
      },
      {
        type: "paragraph",
        text: "`Viewport::_gui_input_event()` 在 `viewport.cpp:1932`，会根据鼠标位置、focus、modal、drag、mouse_filter、shortcut context 等信息找到 Control。Control 的 `_call_gui_input()` 在 `scene/gui/control.cpp:2407`，先发 `gui_input` 信号，再调脚本虚方法和 C++ `gui_input()`。Control 调 `accept_event()` 后，Viewport 的 `is_input_handled()` 为 true，后面的 shortcut/unhandled 阶段会被跳过。"
      },
      {
        type: "table",
        title: "输入回调阶段",
        headers: ["阶段", "谁会收到", "典型用途", "消费关系"],
        rows: [
          ["`_input(event)`", "开启 input processing 的 Node。", "全局快捷键、调试输入、低层事件观察。", "最早；如果 handled，GUI 后续会清理内部状态。"],
          ["`gui_input` / `_gui_input`", "命中的 Control、焦点 Control、modal/drag 相关 Control。", "按钮、文本框、滑条、编辑器控件。", "GUI 最常消费事件，阻止 unhandled。"],
          ["`_shortcut_input(event)`", "开启 shortcut input 的 Node，尤其 Control shortcut context。", "菜单快捷键、UI 快捷键。", "只处理 Key、Shortcut、JoypadButton 等类型。"],
          ["`_unhandled_key_input(event)`", "开启 unhandled key input 的 Node。", "游戏按键但跳过鼠标移动等高频事件。", "发生在 shortcut 之后、unhandled 之前。"],
          ["`_unhandled_input(event)`", "开启 unhandled input 的 Node。", "游戏角色输入、点击空白区域、没有被 UI 吃掉的事件。", "最后阶段；GUI 已消费的事件不会到这里。"],
          ["物理拾取 input_event", "CollisionObject2D/3D。", "鼠标点击 2D/3D 物理对象。", "通常在 unhandled 后仍未处理才排队拾取。"]
        ]
      },
      {
        type: "heading",
        title: "InputMap 和 action"
      },
      {
        type: "paragraph",
        text: "InputMap 的核心问题只有一个：某个 InputEvent 是否匹配某个 action。`InputEvent::is_action()` 在 `input_event.cpp:50`，内部调用 `InputMap::event_is_action()`；`Input.is_action_pressed()` 则查 Input 的 action cache。前者是“这个事件是不是 jump”，后者是“jump 当前是否处于按下状态”。"
      },
      {
        type: "table",
        title: "action 相关 API 区别",
        headers: ["API", "读的是哪里", "适合场景", "注意点"],
        rows: [
          ["`event.is_action_pressed(\"jump\")`", "当前这个 InputEvent + InputMap。", "在 `_input(event)`、`_unhandled_input(event)` 中判断本事件。", "如果事件已被 GUI 消费，unhandled 里不会看到它。"],
          ["`Input.is_action_pressed(\"jump\")`", "Input 的 action_states 缓存。", "在 `_physics_process()` 或 `_process()` 中持续读取状态。", "不是事件对象，不能告诉你本次是谁触发。"],
          ["`Input.is_action_just_pressed()`", "action_states 里的事件 id 和 process/physics frame。", "只在当前帧或物理帧响应一次。", "源码把输入半途到达的情况推到下一个 physics tick。"],
          ["`Input.action_press()`", "脚本直接写 Input 的 action state。", "自动化、模拟输入、虚拟按钮。", "不会自动生成真实平台事件，也不会经过 GUI。"],
          ["`InputMap.action_add_event()`", "修改 action 映射表。", "运行时改键位或编辑器输入映射。", "只改“映射规则”，不表示 action 已按下。"],
          ["`InputMap.load_from_project_settings()`", "从 ProjectSettings 的 `input/*` 项重建映射。", "启动或重载项目输入配置。", "默认 UI action 也在 InputMap 初始化路径里加入。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：为什么按钮吃掉了按键"
      },
      {
        type: "paragraph",
        text: "如果一个 focused Button 或 LineEdit 在 `_gui_input()` 里接受了 Enter/Escape，Viewport 会把事件标记为 handled。此时 `_input(event)` 可能已经看到了事件，但 `_unhandled_input(event)` 收不到。排查这类问题时要从 `Viewport::push_input()` 的顺序看，而不是只查 InputMap。"
      },
      {
        type: "subheading",
        title: "案例二：事件判断和状态判断别混用"
      },
      {
        type: "code",
        code: [
          "func _unhandled_input(event):",
          "    if event.is_action_pressed(\"jump\"):",
          "        jump_once_from_this_event()",
          "",
          "func _physics_process(_delta):",
          "    if Input.is_action_pressed(\"move_left\"):",
          "        velocity.x -= speed"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "第一段关心“本次没被 UI 消费的事件是不是 jump”；第二段关心“当前帧 move_left 是否保持按下”。前者依赖事件能走到 unhandled，后者依赖 Input 的全局状态缓存。"
      },
      {
        type: "subheading",
        title: "案例三：脚本模拟 action 不等于真实按键"
      },
      {
        type: "code",
        code: [
          "Input.action_press(\"attack\")",
          "# Input.is_action_pressed(\"attack\") 会变成 true。",
          "# 但这不会生成 InputEventKey，也不会触发 Control 的 gui_input。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`Input::action_press()` 在 `input.cpp:1410` 直接修改 action state，并设置 pressed frame。它适合虚拟按钮或测试，但如果你要测试完整事件分发链路，需要创建新的 InputEvent 并调用 `parse_input_event()`，还要注意源码在 debug 下警告同一事件对象不能同帧重复 parse。"
      },
      {
        type: "subheading",
        title: "案例四：鼠标事件为什么坐标变了"
      },
      {
        type: "paragraph",
        text: "平台事件里的坐标是窗口/全局语义；Viewport 在 `push_input()` 里会通过 `_make_input_local()` 转成当前 Viewport 本地坐标。SubViewport、嵌入窗口、缩放、canvas transform 都可能改变最终交给 Control 的位置。排查点击不准时，要同时看 InputEvent 的 global_position、position 和 Viewport transform。"
      },
      {
        type: "subheading",
        title: "案例五：手柄 action strength 来自 deadzone"
      },
      {
        type: "paragraph",
        text: "Joypad motion 的 `action_match()` 会结合 InputMap action 的 deadzone 计算 strength/raw_strength。`InputMap::event_get_action_status()` 把 pressed、strength、raw_strength 传回 Input，Input 再更新 action cache。所以“摇杆轻推没触发”通常先查 action deadzone 和 joy mapping，而不是 Viewport 分发。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "Input 和周边概念",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["DisplayServer", "平台窗口、系统事件、输入法、鼠标模式、事件回调。", "不管理 action cache。"],
          ["InputEvent", "一次具体事件的数据和匹配能力。", "不代表持续状态。"],
          ["Input", "全局输入状态、action just pressed/released、mouse/joy/touch 状态。", "不决定 GUI 控件是否消费。"],
          ["InputMap", "action 映射、deadzone、默认 UI action。", "不分发事件给节点。"],
          ["Viewport", "本地坐标、GUI 命中、输入阶段顺序、物理拾取。", "不创建平台事件。"],
          ["Control", "GUI 输入、focus、mouse_filter、accept_event。", "不维护全局 action 状态。"],
          ["SceneTree", "按组调用 `_input`、`_shortcut_input`、`_unhandled_input`。", "不解析平台扫描码。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：InputMap 负责分发事件。它只负责 action 匹配。",
          "误区二：`Input.is_action_pressed()` 和 `event.is_action_pressed()` 一样。一个查全局状态，一个查当前事件。",
          "误区三：GUI 之后还能保证 `_unhandled_input()` 收到所有事件。Control 接受事件后，后续阶段会跳过。",
          "误区四：平台代码直接调用脚本。平台只创建 InputEvent 并交给 Input/Window/Viewport 链路。",
          "误区五：同一个 InputEvent 可以随便同帧重复 parse。debug 下源码会警告，应 duplicate 或新建事件。",
          "误区六：`action_press()` 会模拟完整按键事件。它只修改 action state，不走 Control/Viewport。",
          "误区七：鼠标位置只有一个坐标。InputEventMouse 同时有 position/global_position，Viewport 还会做本地变换。",
          "误区八：unhandled 是唯一适合游戏输入的地方。持续移动通常更适合在 physics/process 里读取 Input 状态。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/input/input_event.h:52`，把 InputEvent 的派生类型和基础方法列出来。",
          "读 `core/input/input_map.h:40` 和 `input_map.cpp:279`，理解 action 匹配只是一张映射表。",
          "读 `core/input/input.cpp:1519` 的 `parse_input_event()`，看缓冲、累积和直接解析的分支。",
          "读 `Input::_parse_input_event_impl()`，按 Key、Mouse、Touch、Joypad、Action cache 的顺序看状态如何更新。",
          "读平台例子：Windows `display_server_windows.cpp:4979`、`:6865` 或 Web/macOS 对应路径，确认系统事件如何变成 InputEvent。",
          "读 `scene/main/window.cpp:1452`，看窗口如何注册输入回调。",
          "读 `scene/main/viewport.cpp:3501`，建立 `_input -> GUI -> shortcut/unhandled` 的分发顺序。",
          "最后读 `scene/main/scene_tree.cpp:1429` 和 `Node::_call_*input`，看节点组如何被调用。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "InputEvent 是一次事件，Input 是全局状态机，InputMap 是 action 地址簿，Viewport/Control/SceneTree 才决定事件先给谁、在哪里被消费、哪些脚本还能收到。"
      }
    ]
  },
  {
    id: "controlgui",
    title: "Control / GUI / Theme / TextServer",
    aliases: ["Control", "GUI", "Theme", "ThemeDB", "ThemeOwner", "ThemeContext", "TextServer", "TextServerManager", "Container", "Label", "Button", "BaseButton", "mouse_filter", "gui_input", "accept_event", "minimum_size", "minimum_size_changed", "update_minimum_size", "get_combined_minimum_size", "anchor", "offset", "size flags", "queue_sort", "fit_child_in_rect", "NOTIFICATION_DRAW", "NOTIFICATION_THEME_CHANGED", "BIND_THEME_ITEM", "shaped_text", "font_draw_glyph"],
    summary: "Control 是 Godot GUI 的节点层：它在 CanvasItem 之上增加布局、焦点、鼠标过滤、主题查找和最小尺寸；Container 负责排布子控件，Theme 决定外观，TextServer 负责复杂文字 shaping 和测量。",
    article: [
      {
        type: "lead",
        text: "Control / GUI / Theme / TextServer 是 Godot UI 系统的主链路。Control 负责“这个控件在哪里、能不能点、需要多大”；Container 负责“子控件怎样排”；Theme 负责“按钮、标签、面板默认长什么样”；TextServer 负责“文字怎样排成可绘制的字形”。所以 GUI 问题不能只看绘制层，很多视觉 bug 实际发生在布局、主题或文字测量阶段。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Control 想成游戏里的 UI 盒子。Label、Button、LineEdit、Panel、Slider 都是盒子。每个盒子有位置、大小、最小尺寸、能不能接鼠标、能不能获得焦点；有些盒子还能装别的盒子。"
      },
      {
        type: "paragraph",
        text: "Container 就像自动排版器。你把按钮放进 VBoxContainer，它会问每个按钮“你最小要多大”，再把它们从上到下摆好。Theme 像皮肤包，提供按钮背景、字体、字号、颜色、边距。TextServer 像排字工，负责中英文混排、阿拉伯语方向、emoji fallback、断行、字形大小这些复杂细节。"
      },
      {
        type: "flow",
        title: "一个 Label/按钮从数据到屏幕",
        steps: [
          { title: "Control 属性", text: "控件保存 anchor、offset、custom_minimum_size、mouse_filter、focus mode 等状态。" },
          { title: "Container 排版", text: "父 Container 读取子控件 minimum_size，再用 `fit_child_in_rect()` 设置最终矩形。" },
          { title: "Theme 查外观", text: "Control 查 stylebox、font、font_size、color、constant 等主题项。" },
          { title: "TextServer 排文字", text: "Label/TextEdit/RichTextLabel 把文本交给 TextServer shaping、断行、测量。" },
          { title: "NOTIFICATION_DRAW", text: "控件在 draw 通知里绘制背景、文字、图标和焦点框。" },
          { title: "Canvas/Rendering", text: "CanvasItem 命令进入渲染链路，最后显示在窗口中。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Control` 的类入口在 `scene/gui/control.h:45`，它继承 `CanvasItem`。这说明 GUI 控件最终仍走 2D canvas 绘制，但 Control 不是普通 Node2D。它额外维护 anchor、offset、layout mode、grow direction、size flags、minimum size cache、focus、mouse_filter、theme owner、theme override、theme cache 和 accessibility 相关状态。"
      },
      {
        type: "paragraph",
        text: "Control 的关键职责集中在三个方向：第一是布局，`update_minimum_size()`、`get_minimum_size()`、`get_combined_minimum_size()` 和 `_size_changed()` 决定控件怎样报告和更新尺寸；第二是输入，Viewport 命中 Control 后会调用 `_call_gui_input()`，Control 可以用 `accept_event()` 消费事件；第三是主题，Control 会按局部 override、控件主题、父主题上下文、项目默认主题、fallback 的顺序查 Theme item。"
      },
      {
        type: "table",
        title: "GUI 系统分层边界",
        headers: ["层级", "代表源码", "负责什么", "不负责什么"],
        rows: [
          ["Control", "`scene/gui/control.h:45`、`control.cpp`", "UI 节点基础能力：矩形、anchor/offset、minimum size、focus、mouse_filter、theme lookup、GUI 输入。", "不自动决定复杂容器布局，也不自己实现文字 shaping。"],
          ["Container", "`scene/gui/container.h:35`、`box_container.cpp`、`grid_container.cpp`", "监听子控件尺寸变化，延迟排序，用 `fit_child_in_rect()` 设置子控件矩形。", "不决定按钮皮肤和字体 fallback。"],
          ["Theme / ThemeDB", "`scene/resources/theme.h:38`、`scene/theme/theme_db.cpp`", "保存 stylebox、font、font_size、color、constant、icon，并提供默认主题和 fallback。", "不执行控件排版，也不接收输入事件。"],
          ["TextServer", "`servers/text/text_server.h:47`", "字体 RID、glyph、shaping、fallback、双向文本、断行和测量。", "不决定 Control 在父容器中的最终矩形。"],
          ["Viewport/Input", "`viewport.cpp:1932`、`control.cpp:2407`", "根据鼠标位置、focus、modal、drag 和 mouse_filter 分发 GUI 输入。", "不保存 Theme item。"],
          ["CanvasItem/RenderingServer", "`scene/main/canvas_item.*`、`servers/rendering`", "把 draw 命令变成 canvas 渲染数据。", "不理解按钮最小尺寸和主题继承。"],
          ["Editor UI", "`editor/` 下大量 Control 子类", "编辑器界面也复用同一套 Control/Theme/TextServer。", "不是另一套独立 GUI 引擎。"]
        ]
      },
      {
        type: "heading",
        title: "布局：anchor、offset 和 minimum size"
      },
      {
        type: "paragraph",
        text: "Control 的定位由 anchor 和 offset 共同决定。anchor 是相对父矩形的比例位置，offset 是像素偏移。`_size_changed()` 会用父控件可锚定矩形、四条边的 anchor 和 offset 计算新的 position/size。Container 模式下，子控件的 anchor/offset 通常被容器管理，开发者更常调 size flags 和 custom minimum size。"
      },
      {
        type: "paragraph",
        text: "最小尺寸是 GUI 布局的核心信号。`Control::update_minimum_size()` 在 `scene/gui/control.cpp:1881`，它会向上失效 minimum size cache，并延迟调用 `_update_minimum_size()`。`get_minimum_size()` 在 `:1923` 调虚方法 `_get_minimum_size`，`get_combined_minimum_size()` 在 `:2026` 把控件自己的最小尺寸和 `custom_minimum_size` 合并。"
      },
      {
        type: "table",
        title: "布局相关函数速查",
        headers: ["函数/字段", "源码线索", "作用", "排查价值"],
        rows: [
          ["`anchor` / `offset`", "`control.h` 的布局字段、`_size_changed()`", "把父矩形比例和像素偏移转成最终 Rect。", "控件位置或伸缩异常时先看。"],
          ["`get_minimum_size()`", "`control.cpp:1923`", "询问控件自己需要的最小尺寸，脚本可实现 `_get_minimum_size`。", "文本或图标被裁剪时要查。"],
          ["`custom_minimum_size`", "`set_custom_minimum_size()`", "用户强制附加的最小尺寸。", "临时撑开控件或排查布局约束。"],
          ["`get_combined_minimum_size()`", "`control.cpp:2026`", "把控件计算值和 custom minimum size 合并。", "Container 看到的通常是这个结果。"],
          ["`update_minimum_size()`", "`control.cpp:1881`", "失效缓存、延迟更新并发出 `minimum_size_changed`。", "文字、主题、图标变化后必须触发。"],
          ["`Container::queue_sort()`", "`container.cpp:146`", "延迟重排子控件。", "避免在一连串变化中反复重排。"],
          ["`Container::fit_child_in_rect()`", "`container.cpp:102`", "真正设置子 Control 的位置和大小。", "Box/Grid 等容器最终都落到类似动作。"]
        ]
      },
      {
        type: "flow",
        title: "布局变更怎样传播",
        steps: [
          { title: "内容变化", text: "Label `set_text()`、Theme 变化、字号变化或脚本改变 custom minimum size。" },
          { title: "更新最小尺寸", text: "Control 调 `update_minimum_size()`，失效自己和父链缓存。" },
          { title: "发信号", text: "延迟 `_update_minimum_size()` 发现尺寸变化后发 `minimum_size_changed`。" },
          { title: "Container 收到", text: "Container 连接子控件尺寸/可见性变化，调用 `queue_sort()`。" },
          { title: "排序通知", text: "延迟 `_sort_children()`，发送 `NOTIFICATION_SORT_CHILDREN`。" },
          { title: "设置 Rect", text: "具体容器根据 size flags 和 minimum size 调 `fit_child_in_rect()`。" },
          { title: "重绘", text: "Control 收到 resize/theme/visibility 等通知后 queue_redraw 或刷新缓存。" }
        ]
      },
      {
        type: "code",
        code: [
          "var label := Label.new()",
          "label.text = \"A very long label\"",
          "label.custom_minimum_size = Vector2(160, 0)",
          "vbox.add_child(label)",
          "",
          "# set_text() 会让 Label 标记文本布局为 dirty，更新 minimum size。",
          "# VBoxContainer 收到 minimum_size_changed 后延迟重排子控件。"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个例子里，Label 的文字长度影响 `_get_minimum_size()`，custom minimum size 又会参与 `get_combined_minimum_size()`。如果父节点是 VBoxContainer，真正位置不是 Label 自己决定，而是容器下一次排序时设置。"
      },
      {
        type: "heading",
        title: "输入：gui_input 和 accept_event"
      },
      {
        type: "paragraph",
        text: "GUI 输入的入口不在 Control 自己轮询，而在 Viewport。`Viewport::_gui_input_event()` 在 `scene/main/viewport.cpp:1932`，它根据鼠标位置、焦点、modal、drag、shortcut context 和 `mouse_filter` 找到目标 Control。随后 `Control::_call_gui_input()` 在 `scene/gui/control.cpp:2407` 先发 `gui_input` 信号，再调用脚本虚方法和 C++ `gui_input()`。"
      },
      {
        type: "paragraph",
        text: "`Control::accept_event()` 在 `control.cpp:2427`。控件接受事件后，Viewport 会把当前输入标记为 handled，后面的 `_shortcut_input()`、`_unhandled_key_input()`、`_unhandled_input()` 可能不会再收到它。这就是按钮、文本框、菜单为什么能“吃掉”按键或鼠标事件。"
      },
      {
        type: "table",
        title: "mouse_filter 的含义",
        headers: ["值", "直觉解释", "典型用途", "影响"],
        rows: [
          ["`MOUSE_FILTER_STOP`", "命中后自己处理，并阻止鼠标事件继续往后找。", "Button、Slider、LineEdit 等可交互控件。", "最容易让 `_unhandled_input` 收不到鼠标事件。"],
          ["`MOUSE_FILTER_PASS`", "自己可以处理，但没有接受时允许继续传给后面的控件/父链。", "透明面板、叠加控件、可选装饰层。", "适合既监听又不完全拦截的 UI。"],
          ["`MOUSE_FILTER_IGNORE`", "鼠标命中测试直接忽略它。", "纯装饰、背景、不可交互遮罩。", "控件仍可绘制，但鼠标不会把它当目标。"]
        ]
      },
      {
        type: "code",
        code: [
          "func _gui_input(event):",
          "    if event is InputEventMouseButton and event.pressed:",
          "        accept_event()",
          "        pressed.emit()",
          "",
          "# accept_event() 后，后续 unhandled 阶段通常不会再处理这次点击。"
        ].join("\n")
      },
      {
        type: "heading",
        title: "Theme：控件外观从哪里来"
      },
      {
        type: "paragraph",
        text: "`Theme` 是 Resource，入口在 `scene/resources/theme.h:38`。它保存的不是一个大图片，而是分类型的主题项：icon、stylebox、font、font_size、color、constant。Control 会通过 `get_theme_stylebox()`、`get_theme_font()`、`get_theme_color()` 等函数查询这些项，相关实现集中在 `scene/gui/control.cpp:3585`、`:3609`、`:3657` 附近。"
      },
      {
        type: "paragraph",
        text: "主题查找有层级。控件自己的 override 优先；然后查控件所属 ThemeOwner/ThemeContext、父控件或窗口主题、项目主题、默认主题和 fallback。`ThemeDB` 在 `scene/theme/theme_db.*` 维护默认主题、项目主题、fallback font/style/icon/base scale 等全局上下文，默认主题创建入口在 `scene/theme/default_theme.cpp:1375` 附近。"
      },
      {
        type: "table",
        title: "Theme item 类别",
        headers: ["类别", "例子", "常见控件", "排查点"],
        rows: [
          ["icon", "按钮图标、折叠箭头、复选框图标。", "Button、Tree、OptionButton。", "图标不显示时查 item name、theme type 和 fallback。"],
          ["stylebox", "normal/hover/pressed/focus 背景。", "Button、Panel、LineEdit、Popup。", "边框、圆角、内边距和背景大多在这里。"],
          ["font", "默认字体或控件专用字体。", "Label、Button、LineEdit、RichTextLabel。", "字体缺字时还要看 Font fallback 和 TextServer。"],
          ["font_size", "默认字号或控件字号。", "所有文字控件。", "文字被裁剪时字号会影响 minimum size。"],
          ["color", "font_color、font_hover_color、outline_color。", "Label、Button、Tree、TextEdit。", "颜色不对通常先查状态名和 theme type。"],
          ["constant", "separation、margin、outline_size。", "BoxContainer、Button、Label。", "控件间距和内部留白常来自 constant。"]
        ]
      },
      {
        type: "paragraph",
        text: "很多控件用 `BIND_THEME_ITEM` 或类似宏把自己需要的主题项声明出来。例如 Label 会绑定 font、font_size、font_color、outline_color、shadow_color、line_spacing 等项。主题变化时，Control 的 `_theme_changed()`、`_notify_theme_override_changed()` 和 `NOTIFICATION_THEME_CHANGED` 会让缓存失效、更新 minimum size，并触发重绘。"
      },
      {
        type: "heading",
        title: "TextServer：文字不是简单画字符"
      },
      {
        type: "paragraph",
        text: "`TextServer` 的类入口在 `servers/text/text_server.h:47`。它负责字体 RID、glyph、shaping、方向、fallback、断行和测量。TextServerManager 会选择 primary interface；GUI 层通常通过 Font、TextLine、TextParagraph 或控件内部的 shaped text 间接使用它。"
      },
      {
        type: "paragraph",
        text: "Label 是最容易读的例子。`Label : public Control` 在 `scene/gui/label.h:36`。`Label::_shape()` 会创建 shaped text，设置方向、添加字符串、处理 bidi override、获取 line breaks，并根据 overrun 行为做裁剪或省略；绘制阶段在 `NOTIFICATION_DRAW` 中确保 shaped text 可用，再调用 TextServer 的字形绘制接口，例如 `font_draw_glyph`。"
      },
      {
        type: "table",
        title: "Label 文本处理链路",
        headers: ["阶段", "源码线索", "做什么", "常见问题"],
        rows: [
          ["set_text", "`label.cpp:1134`", "保存文本，标记 shaping dirty，queue_redraw，并更新 minimum size。", "改文本后布局没变，先查是否触发最小尺寸更新。"],
          ["shape", "`Label::_shape()`", "创建 shaped_text、设置方向、添加字符串、处理 bidi/line break/trim。", "阿拉伯语、emoji、组合字符、换行问题在这里附近看。"],
          ["minimum size", "`Label::get_minimum_size()`", "根据 font、shaped text、autowrap、overrun 计算控件最小需求。", "文字被裁剪或容器不撑开时看。"],
          ["draw", "`NOTIFICATION_DRAW`", "按行绘制 glyph、outline、shadow、visible characters。", "文字可见但位置/颜色不对时看 draw 和 theme cache。"],
          ["TextServer", "`servers/text/text_server.h:47`", "真正处理 glyph、shaping、font fallback 和测量。", "缺字、方向、断行和复杂脚本问题不要只看 Control。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：Label 文字被裁剪"
      },
      {
        type: "paragraph",
        text: "先看 Label 的 `set_text()` 是否让 font_dirty/shaping dirty 生效，再看 `Label::get_minimum_size()` 是否能返回足够尺寸；如果父节点是 Container，再看子控件是否发出 `minimum_size_changed`，父容器是否执行 `queue_sort()`。如果文本本身需要换行或省略，还要看 autowrap、overrun behavior 和 TextServer line break。"
      },
      {
        type: "subheading",
        title: "案例二：Button 样式不生效"
      },
      {
        type: "paragraph",
        text: "这类问题先查 Theme lookup，而不是渲染。确认控件 theme type 是否正确、状态名是否对应 normal/hover/pressed/disabled/focus，局部 override 是否覆盖了主题，项目 default theme 是否已经设置，最后再查 fallback。Theme item 找不到时，Control 可能用默认主题或 fallback，看起来就像你的样式被忽略。"
      },
      {
        type: "subheading",
        title: "案例三：按钮吃掉了游戏输入"
      },
      {
        type: "paragraph",
        text: "当 Button 或 LineEdit 在 `_gui_input()` 中调用 `accept_event()`，Viewport 会把事件标为 handled。此时 `_input()` 可能已经收到过事件，但 `_unhandled_input()` 不会再收到。排查这类 bug 时同时看 `mouse_filter`、focus、modal popup 和 `accept_event()` 调用点。"
      },
      {
        type: "subheading",
        title: "案例四：编辑器插件界面为什么和游戏 UI 逻辑一样"
      },
      {
        type: "paragraph",
        text: "Godot 编辑器自身大量继承 Control。插件添加 Dock、Inspector 插件或自定义面板时，用的仍是 Button、Tree、LineEdit、Container、Theme 和 TextServer。同一套 minimum size、theme lookup 和 gui_input 规则适用于编辑器 UI，只是 theme、快捷键和上下文更复杂。"
      },
      {
        type: "subheading",
        title: "案例五：中文、阿拉伯语或 emoji 显示异常"
      },
      {
        type: "paragraph",
        text: "如果控件矩形正确、字体主题也正确，但文字方向、断行、缺字或 emoji fallback 异常，应进入 Font/TextServer 路线。复杂文字不是逐字符绘制，必须经过 shaping。排查时看 Font 资源、fallback 字体、TextServer interface、shaped_text 内容和 line break 结果。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "Control 和周边系统",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Node", "树、生命周期、process、groups、owner。", "不提供 UI 布局和主题查找。"],
          ["CanvasItem", "2D 绘制命令、可见性、canvas transform。", "不处理 focus、mouse_filter、minimum size。"],
          ["Control", "GUI 矩形、输入、focus、theme lookup、minimum size。", "不实现具体字体 shaping。"],
          ["Container", "自动布局子 Control。", "不决定具体控件如何绘制文字。"],
          ["Theme", "控件默认外观资源。", "不参与 Viewport 输入命中。"],
          ["TextServer", "文字 shaping、glyph、fallback、测量。", "不决定按钮 hover 状态。"],
          ["RenderingServer", "最终渲染 canvas item 和字形。", "不理解 Label 的文本语义和省略规则。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Control 只是 Node2D 加 UI 外观。Control 继承 CanvasItem，但布局、主题、焦点、输入处理是独立体系。",
          "误区二：控件位置永远由 position/size 决定。Container 子控件通常由父容器重新设置 Rect。",
          "误区三：文字被裁剪一定是渲染 bug。多数情况先查 minimum size、Container、font size、autowrap 和 TextServer 测量。",
          "误区四：Theme 只是颜色表。Theme 同时包含 stylebox、font、font_size、icon、constant。",
          "误区五：InputMap 能解释所有 UI 输入问题。GUI 输入是否继续传播取决于 Viewport、Control、mouse_filter 和 accept_event。",
          "误区六：编辑器 UI 是另一套系统。编辑器也复用 Control、Theme、TextServer，只是控件数量和上下文更大。",
          "误区七：复杂文字可以按字符宽度累加。实际需要 TextServer shaping、fallback、bidi 和 line break。",
          "误区八：主题改了只需要 redraw。很多主题项会影响 minimum size，因此还要触发布局更新。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/gui/control.h:45`，列出 Control 相比 CanvasItem 多出的布局、输入和主题字段。",
          "读 `Control::update_minimum_size()`、`get_minimum_size()`、`get_combined_minimum_size()` 和 `_size_changed()`，建立尺寸传播模型。",
          "读 `scene/gui/container.cpp:37`、`:87`、`:102`、`:146`，理解 Container 如何监听子控件并延迟排序。",
          "读 `scene/main/viewport.cpp:1932` 和 `scene/gui/control.cpp:2407`，跟 GUI 输入命中、`gui_input` 和 `accept_event()`。",
          "读 `Control::_theme_changed()`、`get_theme_stylebox()`、`get_theme_font()`、`get_theme_color()`，再读 `scene/resources/theme.h:38` 和 `scene/theme/theme_db.cpp`。",
          "读 `scene/gui/label.cpp` 的 `_shape()`、`get_minimum_size()` 和 `NOTIFICATION_DRAW`，确认 Label 如何连接 Theme、TextServer 和绘制。",
          "最后回到编辑器源码中任意 Dock 或 Inspector 控件，验证编辑器 UI 也走同一套 Control 机制。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Control 是 Godot GUI 的高层节点契约：它管矩形、输入、焦点、主题和最小尺寸；Container 管排版，Theme 管外观，TextServer 管文字，RenderingServer 只负责把最终绘制结果显示出来。"
      }
    ]
  },
  {
    id: "animationtween",
    title: "Animation / Tween",
    aliases: ["Animation", "AnimationPlayer", "AnimationMixer", "AnimationTree", "AnimationLibrary", "AnimationNode", "AnimationNodeBlendTree", "AnimationNodeStateMachine", "AnimationNodeStateMachinePlayback", "Animation track", "value track", "method track", "audio track", "root motion", "UPDATE_CONTINUOUS", "UPDATE_DISCRETE", "UPDATE_CAPTURE", "callback_mode_process", "callback_mode_method", "Tween", "Tweener", "PropertyTweener", "MethodTweener", "CallbackTweener", "IntervalTweener", "SubtweenTweener", "AwaitTweener", "SceneTree::process_tweens", "create_tween", "tween_property", "tween_method", "custom_step"],
    summary: "Animation 是可保存的时间轴资源，AnimationPlayer/AnimationTree 通过 AnimationMixer 把轨道应用到目标对象；Tween 是 SceneTree 管理的运行时插值对象，更适合临时、脚本驱动的一次性变化。",
    article: [
      {
        type: "lead",
        text: "Animation 和 Tween 都是在“时间过去后改变东西”，但它们不是同一个系统。Animation 是资源和时间轴，适合编辑器里可见、可复用、可导出、可混合的动画；Tween 是运行时对象，适合脚本临时创建的 UI 过渡、数值插值、延迟回调和一次性效果。读源码时最重要的问题是：这一帧到底是谁最后写了这个属性。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "Animation 可以想成剪辑软件里的时间轴。你在 0 秒放一个位置，在 1 秒放另一个位置，AnimationPlayer 播放时就按时间把节点的位置改过去。这个时间轴可以存成资源，被场景保存、复制、导入，也能被 AnimationTree 混合成角色走路、跑步、攻击等复杂状态。"
      },
      {
        type: "paragraph",
        text: "Tween 更像脚本临时写的一条过渡命令：把按钮透明度在 0.2 秒内从 0 变到 1，或者等 1 秒后调用某个函数。它通常不需要编辑器时间轴，也不需要保存成动画资源。Tween 创建后交给 SceneTree 每帧推进，完成后就失效。"
      },
      {
        type: "flow",
        title: "AnimationPlayer 播放一段动画",
        steps: [
          { title: "Animation Resource", text: "保存轨道、关键帧、长度、循环模式、插值和更新模式。" },
          { title: "AnimationPlayer.play", text: "`AnimationPlayer::play()` 选择动画、处理 blend、设置播放位置和速度。" },
          { title: "AnimationMixer 进程", text: "按 physics/idle/manual 模式在 `_process_animation()` 里推进。" },
          { title: "生成 AnimationInstance", text: "AnimationPlayer 或 AnimationTree 产生本帧要参与混合的动画实例和权重。" },
          { title: "_blend_process", text: "按轨道类型插值、调用方法、播放音频或驱动子 AnimationPlayer。" },
          { title: "_blend_apply", text: "把缓存结果写回 Node3D、Skeleton、Object 属性、AudioStreamPlayer 等目标。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Animation` 是 Resource，入口在 `scene/resources/animation.h:38`。它保存 `Track` 列表、关键帧时间和值、插值类型、循环模式和轨道路径。`Animation` 本身不每帧运行，它只是数据；真正把时间推进成属性变化的是 `AnimationMixer` 及其子类 `AnimationPlayer`、`AnimationTree`。"
      },
      {
        type: "paragraph",
        text: "`AnimationPlayer` 的类入口在 `scene/animation/animation_player.h:36`，继承 `AnimationMixer`。`play()` 在 `animation_player.cpp:416`，负责选择当前动画、处理 section/marker、blend、播放队列、速度和起止时间；`advance()` 在 `:716`，手动推进时直接调用 `AnimationMixer::advance()`。"
      },
      {
        type: "paragraph",
        text: "`AnimationMixer` 的类入口在 `scene/animation/animation_mixer.h:42`。它管理 AnimationLibrary、动画集合、root_node、track cache、callback mode 和混合缓存。核心入口是 `_process_animation()`，源码在 `animation_mixer.cpp:1015`：先 `_blend_init()`，再让子类 `_blend_pre_process()` 生成本帧动画实例，随后 `_blend_capture()`、`_blend_calc_total_weight()`、`_blend_process()`、`_blend_apply()`，最后发出 `mixer_applied`。"
      },
      {
        type: "table",
        title: "动画系统分层边界",
        headers: ["层级", "代表源码", "负责什么", "不负责什么"],
        rows: [
          ["Animation Resource", "`scene/resources/animation.h:38`", "保存轨道、关键帧、插值、循环、长度和 marker。", "不自己每帧写属性。"],
          ["AnimationLibrary", "`scene/resources/animation_library.*`", "把多段 Animation 组织成库，供 AnimationMixer/Player 管理。", "不决定播放状态。"],
          ["AnimationMixer", "`scene/animation/animation_mixer.h:42`", "统一处理缓存、混合、轨道应用、callback mode、root node。", "不提供编辑器时间轴 UI。"],
          ["AnimationPlayer", "`animation_player.h:36`、`play()`", "管理播放状态、队列、blend、section、marker、seek、finished 信号。", "不表达复杂状态机图。"],
          ["AnimationTree", "`animation_tree.h:433`", "用 AnimationNode 图混合多个动画，维护参数、状态机、BlendTree。", "不保存底层关键帧数据。"],
          ["Tween", "`scene/animation/tween.h:67`", "运行时插值、延迟、回调、方法 tween、subtween/await。", "不保存为可导入的时间轴资源。"],
          ["SceneTree", "`scene_tree.cpp:825`、`:1779`", "创建和推进 Tween，按 physics/idle、pause、time scale 过滤。", "不理解 Animation 轨道语义。"]
        ]
      },
      {
        type: "heading",
        title: "Animation 轨道类型"
      },
      {
        type: "paragraph",
        text: "轨道类型定义在 `Animation::TrackType`，源码在 `scene/resources/animation.h:48`。轨道路径通常指向某个 Node 或对象属性；AnimationMixer 会把这些路径解析进 track cache。连续型轨道会先混合到缓存里，再在 `_blend_apply()` 写回目标；方法、音频和子动画轨道则在 `_blend_process()` 中按关键帧时机触发。"
      },
      {
        type: "table",
        title: "主要轨道类型",
        headers: ["轨道", "保存什么", "执行方式", "典型用途"],
        rows: [
          ["`TYPE_VALUE`", "Variant 属性关键帧和 update mode。", "连续/离散/capture 三种模式，最终 `set_indexed()` 写回目标。", "颜色、数值、Node 属性、资源属性。"],
          ["`TYPE_POSITION_3D`", "Vector3 位置关键帧，可压缩。", "插值后写 Node3D 或 Skeleton bone position，并可参与 root motion。", "角色骨骼和 3D 节点位移。"],
          ["`TYPE_ROTATION_3D`", "Quaternion 旋转关键帧，可压缩。", "插值后写 rotation 或 bone rotation。", "骨骼动画、摄像机转向。"],
          ["`TYPE_SCALE_3D`", "Vector3 缩放关键帧，可压缩。", "插值后写 scale 或 bone scale。", "角色骨骼、3D 节点缩放。"],
          ["`TYPE_BLEND_SHAPE`", "blend shape 权重。", "混合后调用 MeshInstance3D 的 blend shape setter。", "表情、形变动画。"],
          ["`TYPE_METHOD`", "方法名和参数。", "按关键帧调用目标 Object，可 deferred 或 immediate。", "脚步声、事件、切状态。"],
          ["`TYPE_BEZIER`", "贝塞尔曲线值和手柄。", "按曲线插值成数值。", "自定义曲线运动。"],
          ["`TYPE_AUDIO`", "AudioStream 关键帧和起止偏移。", "驱动 AudioStreamPlayer 播放/停止 stream。", "过场、角色动作音效。"],
          ["`TYPE_ANIMATION`", "另一个 AnimationPlayer 的动画名。", "触发/seek 子 AnimationPlayer。", "组合动画和嵌套播放。"]
        ]
      },
      {
        type: "heading",
        title: "AnimationMixer 的执行模型"
      },
      {
        type: "paragraph",
        text: "AnimationMixer 按 `callback_mode_process` 决定在哪个阶段跑。`_set_process()` 在 `animation_mixer.cpp:446` 会根据 Physics、Idle、Manual 选择内部 physics_process、process，或完全交给手动 `advance()`。`_notification()` 在 `:2397` 中处理 `NOTIFICATION_INTERNAL_PROCESS` 和 `NOTIFICATION_INTERNAL_PHYSICS_PROCESS`，分别调用 `_process_animation(get_process_delta_time())` 或 `_process_animation(get_physics_process_delta_time())`。"
      },
      {
        type: "table",
        title: "Animation callback mode",
        headers: ["设置", "源码含义", "适合场景", "注意点"],
        rows: [
          ["`ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS`", "内部 physics process 推进。", "影响物理相关属性或需要固定 tick 的动画。", "和普通 process 写同一属性时要注意先后顺序。"],
          ["`ANIMATION_CALLBACK_MODE_PROCESS_IDLE`", "内部 idle process 推进，默认更常见。", "UI、普通场景动画、视觉过渡。", "帧率变化会影响 delta，但动画时间仍按秒推进。"],
          ["`ANIMATION_CALLBACK_MODE_PROCESS_MANUAL`", "不自动处理，调用 `advance()` 推进。", "过场回放、编辑器预览、确定性控制。", "忘记 advance 就不会动。"],
          ["`callback_mode_method`", "方法轨道 deferred 或 immediate。", "决定 Method track 何时真正调用。", "immediate 可能在动画处理中改变播放状态，源码注释建议谨慎。"],
          ["`callback_mode_discrete`", "离散 value track 的主导/从属/强制连续策略。", "处理离散关键帧和混合冲突。", "离散轨道 seek、blend 时的表现要单独确认。"]
        ]
      },
      {
        type: "paragraph",
        text: "`_blend_process()` 在 `animation_mixer.cpp:1223`。它遍历本帧所有 AnimationInstance，再遍历每条轨道。连续 value/transform/bezier 会按权重累加到 TrackCache；离散 value 会按关键帧范围直接 `set_indexed()`；method track 调 `_call_object()`；audio track 维护 stream 播放信息；animation track 驱动另一个 AnimationPlayer。`_blend_apply()` 在 `:1902`，把混合后的缓存写回 Node3D、Skeleton3D、MeshInstance3D、Object 属性和音频播放状态。"
      },
      {
        type: "flow",
        title: "AnimationTree 混合路径",
        steps: [
          { title: "AnimationTree active", text: "AnimationTree 也是 AnimationMixer，active 时设置内部 process。" },
          { title: "验证图", text: "`_blend_pre_process()` 更新参数、验证 AnimationNode 图和连接。" },
          { title: "root node process", text: "从 root AnimationNode 开始递归处理 BlendTree/StateMachine。" },
          { title: "make_animation_instance", text: "AnimationNode 把实际动画名、时间、delta、权重写入 AnimationMixer。" },
          { title: "Mixer 统一处理", text: "后续仍走 `_blend_process()` 和 `_blend_apply()` 应用轨道。" }
        ]
      },
      {
        type: "paragraph",
        text: "`AnimationTree` 的类入口在 `scene/animation/animation_tree.h:433`，`_blend_pre_process()` 在 `animation_tree.cpp:650`。它先更新属性和验证图，然后构建 `AnimationNode::ProcessState`，从 root AnimationNode 开始处理。BlendTree 的入口在 `animation_blend_tree.cpp:1678`，StateMachine 的 `travel()` 在 `animation_node_state_machine.cpp:265`，但它们最后都会通过 `make_animation_instance()` 把动画实例交回 AnimationMixer。"
      },
      {
        type: "heading",
        title: "Tween：运行时插值队列"
      },
      {
        type: "paragraph",
        text: "`Tween` 是 RefCounted，入口在 `scene/animation/tween.h:67`。它不是节点，也不能直接 `Tween.new()`；默认构造函数在 `tween.cpp:551` 会报错，正确入口是 `Node::create_tween()` 或 `SceneTree::create_tween()`。`Node::create_tween()` 在 `scene/main/node.cpp:2619`，会让 SceneTree 创建 Tween 并默认 `bind_node(this)`。"
      },
      {
        type: "paragraph",
        text: "`SceneTree::create_tween()` 在 `scene/main/scene_tree.cpp:1779`，创建 `Ref<Tween>` 并放进 `tweens` 列表。SceneTree 在 physics 阶段和 idle 阶段分别调用 `process_tweens()`，实现位于 `scene_tree.cpp:825`。它会根据 Tween 的 process mode、pause mode、绑定节点、time scale 决定是否调用 `Tween::step()`；`step()` 返回 false 时，SceneTree 会清理并移除 Tween。"
      },
      {
        type: "flow",
        title: "Tween 的推进链路",
        steps: [
          { title: "create_tween", text: "Node 或 SceneTree 创建 Tween，SceneTree 保存到 tweens 列表。" },
          { title: "追加 Tweener", text: "`tween_property()`、`tween_method()`、`tween_callback()` 等创建具体 Tweener。" },
          { title: "chain/parallel", text: "`append()` 根据 parallel_enabled 决定加入当前 step 还是下一个 step。" },
          { title: "process_tweens", text: "SceneTree 按 process mode、pause mode 和 time scale 过滤。" },
          { title: "Tween.step", text: "启动当前 step 的 tweeners，按 delta 推进，完成后进入下一 step 或 loop。" },
          { title: "finish/kill", text: "完成后发 `finished`，dead/valid 状态改变，SceneTree 移除它。" }
        ]
      },
      {
        type: "table",
        title: "Tween/Tweener 组件",
        headers: ["组件", "源码线索", "作用", "注意点"],
        rows: [
          ["Tween", "`tween.h:67`、`tween.cpp:335`", "保存步骤队列、循环、速度、pause/process mode、绑定节点和默认 trans/ease。", "完成后 dead，不能当永久对象反复播放。"],
          ["PropertyTweener", "`tween_property()`、`PropertyTweener::step()`", "读取初始值，计算 delta，用 `set_indexed()` 写目标属性。", "目标对象失效时会结束；RefCounted 目标会保留引用副本。"],
          ["MethodTweener", "`tween_method()`", "每步把插值结果作为参数调用 Callable。", "适合没有直接属性或要自定义 setter 的情况。"],
          ["CallbackTweener", "`tween_callback()`", "到时间后调用一次 Callable。", "可以和 interval/chain 做延迟流程。"],
          ["IntervalTweener", "`tween_interval()`", "只消耗时间，不写属性。", "常用于 wait/延迟。"],
          ["SubtweenTweener", "`tween_subtween()`", "把另一个 Tween 作为子流程。", "会从原 parent_tree 移除 subtween，避免双重推进。"],
          ["AwaitTweener", "`tween_await()`", "等待一个 Signal。", "适合把信号等待插入 tween 链。"]
        ]
      },
      {
        type: "code",
        code: [
          "func fade_in(panel: Control) -> void:",
          "    panel.modulate.a = 0.0",
          "    var tween := panel.create_tween()",
          "    tween.set_trans(Tween.TRANS_SINE)",
          "    tween.set_ease(Tween.EASE_OUT)",
          "    tween.tween_property(panel, \"modulate:a\", 1.0, 0.2)",
          "    tween.tween_callback(func(): panel.mouse_filter = Control.MOUSE_FILTER_STOP)"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这段适合 Tween，不适合专门做 Animation 资源：它是一次性 UI 过渡，目标是运行时的 panel，最后还有一个脚本回调。`PropertyTweener::step()` 会把 `modulate:a` 从初始值插到目标值；`CallbackTweener` 到点后调用闭包。"
      },
      {
        type: "code",
        code: [
          "func play_attack(anim: AnimationPlayer) -> void:",
          "    anim.play(\"attack\")",
          "",
          "func scrub_preview(anim: AnimationPlayer, t: float) -> void:",
          "    anim.seek(t, true)",
          "    # 或在 manual 模式下用 anim.advance(delta) 精确推进。"
        ].join("\n")
      },
      {
        type: "heading",
        title: "Animation 和 Tween 怎么选"
      },
      {
        type: "table",
        title: "取舍表",
        headers: ["需求", "优先用", "原因", "反例"],
        rows: [
          ["角色骨骼、导入动画、可视化时间轴", "Animation / AnimationPlayer / AnimationTree", "需要资源化、编辑器编辑、轨道混合、导入导出。", "一个按钮淡入不需要整条 Animation。"],
          ["临时 UI 过渡", "Tween", "脚本创建、完成即丢弃、链式回调方便。", "复杂可复用 UI 状态也可以做 Animation。"],
          ["状态机和多动画混合", "AnimationTree", "BlendTree/StateMachine 能按参数权重产生多个 AnimationInstance。", "简单播放一段动画用 AnimationPlayer 足够。"],
          ["一次性数值插值", "Tween", "不用创建资源，也不污染场景动画库。", "需要美术调整曲线和关键帧时用 Animation。"],
          ["方法/音频严格绑在时间轴上", "Animation", "Method track、Audio track 与关键帧位置绑定。", "简单延迟回调用 Tween callback 更轻。"],
          ["确定性手动推进", "两者都可", "AnimationPlayer 可 manual advance；Tween 可 custom_step。", "要先明确 pause/time scale/process mode。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：同一个属性被谁最后写入"
      },
      {
        type: "paragraph",
        text: "如果角色位置抖动，常见原因是 AnimationPlayer、AnimationTree、Tween、脚本 `_process()` 或物理代码都在写同一个属性。排查时先看写入发生在哪个阶段：AnimationMixer 的 idle/physics/manual，SceneTree 的 `process_tweens()`，脚本 process/physics_process，物理同步。最后写入者决定本帧结果。"
      },
      {
        type: "subheading",
        title: "案例二：Method track 导致动画中途切换"
      },
      {
        type: "paragraph",
        text: "Method track 在 `_blend_process()` 中按关键帧范围调用 `_call_object()`。如果 callback mode 是 immediate，方法可能在动画处理过程中调用 `play()` 或 `stop()`，改变当前动画。源码里 AnimationPlayer 的 blend 逻辑也提醒：动画事件触发新动画时，deferred 调用通常更安全。"
      },
      {
        type: "subheading",
        title: "案例三：AnimationTree travel 没效果"
      },
      {
        type: "paragraph",
        text: "先查 AnimationTree 是否 active、root AnimationNode 是否存在、animation_player 路径是否正确、参数路径是否更新、状态机图是否验证成功。`AnimationTree::_blend_pre_process()` 会在验证失败时返回 false，后续就不会生成 AnimationInstance，也就不会进入 Mixer 的轨道应用。"
      },
      {
        type: "subheading",
        title: "案例四：Tween 节点离树后停止"
      },
      {
        type: "paragraph",
        text: "`Node::create_tween()` 默认把 Tween 绑定到当前节点。`Tween::can_process()` 和 `get_bound_node()` 会检查绑定节点是否仍在树里、是否能 process。节点离树或释放后，Tween 可能停止或失效。想让 Tween 不随节点暂停，要显式考虑 `bind_node()`、`set_pause_mode()` 和创建位置。"
      },
      {
        type: "subheading",
        title: "案例五：Audio track 和 AudioServer 的边界"
      },
      {
        type: "paragraph",
        text: "Animation 的 audio track 不直接混音。它在 `_blend_process()` 中找到 AudioStreamPlayer，设置 stream、play、获取 playback，并在 `_blend_apply()` 维护 stream volume 和停止条件。真正混音仍然属于 AudioServer。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "Animation/Tween 和周边系统",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Animation", "关键帧和轨道数据。", "不决定每帧何时推进。"],
          ["AnimationPlayer", "播放一段或一组动画，处理队列、blend、seek、section。", "不表达复杂混合图。"],
          ["AnimationTree", "状态机/BlendTree/参数驱动混合。", "不保存原始关键帧。"],
          ["AnimationMixer", "统一缓存、混合、轨道应用。", "不负责资源导入。"],
          ["Tween", "运行时链式插值、等待和回调。", "不生成可编辑动画时间轴。"],
          ["SceneTree", "推进 Tween、Timer、process/physics 调度。", "不执行 Animation track 语义。"],
          ["AudioServer", "实际音频混音和设备输出。", "不决定 Animation audio track 的关键帧时机。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Animation 和 Tween 是同一个系统。Animation 是资源时间轴，Tween 是运行时插值对象。",
          "误区二：Animation Resource 自己会播放。它只是数据，播放靠 AnimationPlayer/Tree/Mixer。",
          "误区三：Tween 可以直接 new。源码明确要求通过 create_tween() 创建。",
          "误区四：Method track 总是安全立刻调用。immediate 回调可能改变正在处理的动画状态，deferred 往往更稳。",
          "误区五：AnimationTree 只是更复杂的 AnimationPlayer。它用 AnimationNode 图生成多个带权重的 AnimationInstance，再交给 Mixer。",
          "误区六：Tween 完成后还能反复 play。完成后 dead/valid 状态变化，通常应重新创建。",
          "误区七：属性变化异常先怪插值算法。先找谁最后写了属性，再看 process mode、pause mode、time scale。",
          "误区八：Audio track 负责混音。它只按时间驱动 AudioStreamPlayer，混音仍在 AudioServer。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `scene/resources/animation.h:38`，理解 Animation 是 Resource，重点看 TrackType、UpdateMode、LoopMode。",
          "读 `AnimationPlayer::play()` 和 `advance()`，确认播放状态、blend、section、marker、队列和手动推进。",
          "读 `AnimationMixer::_process_animation()`，记住 `_blend_init -> _blend_pre_process -> _blend_process -> _blend_apply` 主流程。",
          "读 `_blend_process()` 的 TYPE_VALUE、TYPE_METHOD、TYPE_AUDIO、TYPE_ANIMATION 分支，理解不同轨道何时插值、调用或播放。",
          "读 `_blend_apply()`，看最终如何写回 Object、Node3D、Skeleton、MeshInstance3D 和音频播放状态。",
          "读 `AnimationTree::_blend_pre_process()`、BlendTree 和 StateMachine 的入口，理解它们怎样生成 AnimationInstance。",
          "读 `Node::create_tween()`、`SceneTree::create_tween()`、`SceneTree::process_tweens()`，确认 Tween 由 SceneTree 管理。",
          "最后读 `Tween::step()` 和 `PropertyTweener::step()`，理解 chain/parallel、loop、pause、bound node 和属性写回。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Animation 是可保存、可混合、可编辑的时间轴；AnimationMixer 负责把轨道应用到对象；Tween 是 SceneTree 推进的运行时插值队列。排查动画问题时，先确定这一帧谁在什么阶段写了目标属性。"
      }
    ]
  },
  {
    id: "debugperformance",
    title: "Performance / EngineDebugger / Editor Debugger",
    aliases: ["Performance", "EngineDebugger", "EngineProfiler", "RemoteDebugger", "LocalDebugger", "ScriptDebugger", "ScriptEditorDebugger", "EditorDebuggerNode", "EditorProfiler", "EditorVisualProfiler", "EditorPerformanceProfiler", "Monitors", "Profiler", "Visual Profiler", "performance:profile_frame", "performance:profile_names", "servers:profile_frame", "servers:profile_total", "profiler_enable", "toggle_profiler", "get_monitor", "add_custom_monitor", "get_monitor_modification_time", "MONITOR_MAX", "TIME_FPS", "TIME_PROCESS", "TIME_PHYSICS_PROCESS", "TIME_NAVIGATION_PROCESS"],
    summary: "Godot 的调试和性能数据分两半：运行时的 Performance/EngineDebugger 负责采样、聚合和发消息，编辑器的 EditorDebuggerNode/ScriptEditorDebugger/Profiler 面板负责接收、解析和展示。",
    article: [
      {
        type: "lead",
        text: "Performance / EngineDebugger / Editor Debugger 这条线的核心不是“编辑器自己测项目性能”，而是运行中的引擎把采样数据发给编辑器。Performance 是运行时监视器集合，EngineDebugger 是调试消息和 profiler 回调中心，RemoteDebugger/LocalDebugger 是传输和调度实现，editor/debugger 里的控件只是把这些数据画成表格和曲线。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Performance 想成游戏运行时自己的仪表盘：FPS、process 时间、物理时间、内存、对象数、渲染 draw call、物理碰撞对、音频延迟等读数都在这里。脚本里读 `Performance.get_monitor(...)`，本质就是向这个仪表盘问当前读数。"
      },
      {
        type: "paragraph",
        text: "EngineDebugger 像运行时和编辑器之间的邮局。编辑器打开 Profiler 或 Monitors 后，会通过调试连接告诉运行时启用某个 profiler；运行时每帧或每秒采样，把数据打包成消息发回编辑器；编辑器面板再把消息变成曲线、列表、调用栈或错误面板。"
      },
      {
        type: "flow",
        title: "Monitors 面板的数据路径",
        steps: [
          { title: "Main::setup", text: "创建 `Performance`，注册为 Engine singleton：`Performance`。" },
          { title: "Main::iteration", text: "每帧统计 frame/process/physics/navigation 时间，周期性写回 Performance。" },
          { title: "EngineDebugger::iteration", text: "主循环把帧时间交给 EngineDebugger，触发活跃 profiler 的 tick。" },
          { title: "RemoteDebugger PerformanceProfiler", text: "每秒读取 Performance monitor 和 custom monitor。" },
          { title: "send_message", text: "发送 `performance:profile_names` 和 `performance:profile_frame`。" },
          { title: "ScriptEditorDebugger", text: "按消息名分发到 `_msg_performance_profile_*`。" },
          { title: "EditorPerformanceProfiler", text: "更新 monitor 树、历史数据和曲线绘制。" }
        ]
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Performance` 的类入口在 `main/performance.h:39`。它继承 Object，是一个 Engine singleton。`Main::setup()` 在 `main/main.cpp:1070` 创建它，随后 `engine->add_singleton(Engine::Singleton(\"Performance\", performance))` 把它暴露给脚本和编辑器。"
      },
      {
        type: "paragraph",
        text: "Performance 内部有两类数据：一类是固定 `Monitor` 枚举，包含 FPS、process/physics/navigation 时间、内存、对象数、渲染、物理、音频、导航、pipeline compilation 等；另一类是用户自定义 monitor，通过 `add_custom_monitor()` 注册 Callable。`get_monitor()` 在 `main/performance.cpp:180` 附近按 monitor 类型去 Engine、Memory、ObjectDB、ResourceCache、RenderingServer、PhysicsServer、AudioServer、NavigationServer 等地方取数。"
      },
      {
        type: "table",
        title: "Performance monitor 来源",
        headers: ["类别", "例子", "数据来源", "排查价值"],
        rows: [
          ["时间", "`time/fps`、`time/process`、`time/physics_process`、`time/navigation_process`", "Engine FPS 和 Main 写入的 process/physics/navigation 时间。", "判断瓶颈在空闲帧、固定物理步还是导航。"],
          ["内存", "`memory/static`、`memory/msg_buf_max`", "Memory 和 MessageQueue。", "排查内存增长、消息队列峰值。"],
          ["对象", "`object/objects`、`object/resources`、`object/nodes`、`object/orphan_nodes`", "ObjectDB、ResourceCache、SceneTree/Node 统计。", "发现对象泄漏、孤儿节点、资源缓存异常。"],
          ["渲染", "`raster/*`、`video/*`、`pipeline/compilations_*`", "RenderingServer 的 rendering info。", "draw call、显存、pipeline 编译卡顿。"],
          ["物理", "`physics_2d/*`、`physics_3d/*`", "PhysicsServer2D/3D process info。", "活动对象、碰撞对、island 数量。"],
          ["音频", "`audio/driver/output_latency`", "AudioServer 输出延迟。", "音频延迟和设备相关问题。"],
          ["导航", "`navigation_2d/*`、`navigation_3d/*`", "NavigationServer2D/3D process info。", "地图、区域、agent、边数量。"],
          ["自定义", "`custom:<name>`", "用户注册 Callable 返回数字。", "项目自定义业务指标。"]
        ]
      },
      {
        type: "heading",
        title: "Main 循环怎样写入性能时间"
      },
      {
        type: "paragraph",
        text: "Performance 不是每个字段都自己测。主循环里已经有 process、physics、navigation 等阶段的计时变量。`main/main.cpp:5090` 在调试器活跃时调用 `EngineDebugger::iteration(frame_time, process_ticks, physics_process_ticks, physics_step)`；随后每隔一段统计窗口更新 FPS，并在 `main/main.cpp:5111`、`:5112`、`:5113` 分别调用 `performance->set_process_time()`、`set_physics_process_time()`、`set_navigation_process_time()`。"
      },
      {
        type: "table",
        title: "时间数据的含义",
        headers: ["数据", "写入/读取", "直觉解释", "注意点"],
        rows: [
          ["`TIME_FPS`", "`Engine::get_frames_per_second()`", "最近统计窗口的帧率。", "不是每一帧瞬时 FPS。"],
          ["`TIME_PROCESS`", "`set_process_time()`", "空闲帧 process 阶段统计值。", "适合判断 `_process`、idle tween、普通场景更新压力。"],
          ["`TIME_PHYSICS_PROCESS`", "`set_physics_process_time()`", "固定物理阶段统计值。", "适合判断 `_physics_process`、物理同步、物理查询压力。"],
          ["`TIME_NAVIGATION_PROCESS`", "`set_navigation_process_time()`", "导航处理统计值。", "导航 agent/map 更新压力会体现在这里。"],
          ["`EngineDebugger::iteration` 参数", "`frame_time`、`process_ticks`、`physics_process_ticks`、`physics_step`", "发给 profiler tick 的帧级时间。", "profiler 可以用这组数据构建更细的面板。"]
        ]
      },
      {
        type: "heading",
        title: "EngineDebugger 和 EngineProfiler"
      },
      {
        type: "paragraph",
        text: "`EngineDebugger` 的类入口在 `core/debugger/engine_debugger.h:43`。它保存静态的 profiler 表、message capture 表和 URI 协议表。`Profiler` 结构在 `engine_debugger.h:53`，包含 `toggle`、`add`、`tick` 三个回调和 active 状态。`EngineDebugger::iteration()` 在 `engine_debugger.cpp:107`，会把帧时间转成秒，然后遍历 active profiler，调用它们的 tick。"
      },
      {
        type: "paragraph",
        text: "`EngineProfiler` 是脚本/扩展可用的 RefCounted 包装层，入口在 `core/debugger/engine_profiler.h:36`。`EngineProfiler::bind()` 在 `engine_profiler.cpp:54` 创建一个 `EngineDebugger::Profiler`，把 `_toggle`、`_add_frame`、`_tick` 虚方法接到 EngineDebugger 上。这样核心 profiler 和用户 profiler 都能走同一套启停和 tick 机制。"
      },
      {
        type: "flow",
        title: "Profiler 启用链路",
        steps: [
          { title: "编辑器 UI", text: "用户在 Profiler/Visual Profiler/Monitors 面板点启用，或 autostart 打开。" },
          { title: "ScriptEditorDebugger.toggle_profiler", text: "发送 `profiler:<name>` 消息，数据里带 enable 和 options。" },
          { title: "RemoteDebugger capture", text: "`RemoteDebugger::_profiler_capture()` 解析消息并调用 `profiler_enable()`。" },
          { title: "EngineDebugger.profiler_enable", text: "找到对应 profiler，调用 toggle，设置 active。" },
          { title: "主循环 iteration", text: "每帧调用 active profiler 的 tick。" },
          { title: "send_message 回编辑器", text: "profiler 把 frame data 通过 RemoteDebugger 发回。" }
        ]
      },
      {
        type: "table",
        title: "调试器核心对象",
        headers: ["对象", "源码入口", "职责", "边界"],
        rows: [
          ["EngineDebugger", "`core/debugger/engine_debugger.h:43`", "profiler/capture/协议注册、每帧 tick、发消息抽象。", "不绘制编辑器 UI。"],
          ["EngineProfiler", "`core/debugger/engine_profiler.h:36`", "把 RefCounted 虚方法适配成 EngineDebugger profiler 回调。", "不负责网络传输。"],
          ["RemoteDebugger", "`core/debugger/remote_debugger.cpp`", "远程调试连接、消息封包、错误/输出/性能数据发送。", "不保存 Performance monitor 定义。"],
          ["LocalDebugger", "`core/debugger/local_debugger.cpp`", "本地调试实现，服务命令行/本地调试场景。", "不等同编辑器 UI。"],
          ["ScriptDebugger", "`core/debugger/script_debugger.*`", "断点、脚本栈、错误中断、语言调试状态。", "不负责 Monitors 曲线绘制。"],
          ["ScriptEditorDebugger", "`editor/debugger/script_editor_debugger.h:59`", "编辑器端会话 UI 和消息分发。", "不直接测运行时性能。"]
        ]
      },
      {
        type: "heading",
        title: "RemoteDebugger 的 PerformanceProfiler"
      },
      {
        type: "paragraph",
        text: "性能监视器面板最直接的数据来源是 `RemoteDebugger::PerformanceProfiler`，定义在 `core/debugger/remote_debugger.cpp:47`。它继承 `EngineProfiler`，构造时保存 Performance 对象。`RemoteDebugger` 构造函数在 `remote_debugger.cpp:775` 获取 Engine singleton 里的 Performance，创建 performance_profiler，`bind(\"performance\")`，并默认 `profiler_enable(\"performance\", true)`。"
      },
      {
        type: "paragraph",
        text: "`PerformanceProfiler::tick()` 每秒运行一次。它读取 custom monitor 名称和类型，检查 `get_monitor_modification_time()`，如果自定义 monitor 有变化就发 `performance:profile_names`；然后按 `MONITOR_MAX` 遍历 `get_monitor(i)`，再追加 custom monitor 值，最后发 `performance:profile_frame`。所以 Monitors 面板是一秒级采样，不是每帧完整采样。"
      },
      {
        type: "table",
        title: "调试消息通道",
        headers: ["消息", "发送侧", "接收侧", "用途"],
        rows: [
          ["`performance:profile_names`", "RemoteDebugger PerformanceProfiler", "`_msg_performance_profile_names()`", "同步自定义 monitor 名称和类型。"],
          ["`performance:profile_frame`", "RemoteDebugger PerformanceProfiler", "`_msg_performance_profile_frame()`", "发送一帧 monitor 数值数组。"],
          ["`servers:profile_frame` / `servers:profile_total`", "服务器/脚本 profiler", "`_msg_servers_profile_common()`", "构建脚本函数和服务器函数耗时表。"],
          ["`visual:profile_frame`", "Visual profiler", "`_msg_visual_profile_frame()`", "显示 CPU/GPU 区域耗时。"],
          ["`error` / `output` / `stack_dump`", "ScriptDebugger/RemoteDebugger", "对应 `_msg_*` handler", "错误、输出、断点栈和变量查看。"],
          ["`scene:*`", "Scene debugger", "远程场景树/对象检查 handler", "编辑器查看运行中场景树和远程对象。"]
        ]
      },
      {
        type: "heading",
        title: "编辑器端怎样展示"
      },
      {
        type: "paragraph",
        text: "编辑器入口是 `EditorDebuggerNode`，类入口在 `editor/debugger/editor_debugger_node.h:47`。构造函数在 `editor_debugger_node.cpp:58` 创建 Debugger 底部 Dock、TabContainer、默认 ScriptEditorDebugger 和远程场景树面板。每个运行会话对应一个 `ScriptEditorDebugger`，它继承 `MarginContainer`，入口在 `script_editor_debugger.h:59`。"
      },
      {
        type: "paragraph",
        text: "`ScriptEditorDebugger` 构造时创建多个标签页：Stack Trace、Errors、Evaluator、Profiler、Visual Profiler、Monitors、Video RAM 等。源码在 `script_editor_debugger.cpp:2351`、`:2359`、`:2366` 分别创建 `EditorProfiler`、`EditorVisualProfiler`、`EditorPerformanceProfiler`。`_init_parse_message_handlers()` 把 `performance:profile_frame`、`performance:profile_names`、`servers:profile_frame`、`visual:profile_frame` 等消息名映射到对应处理函数。"
      },
      {
        type: "paragraph",
        text: "`EditorPerformanceProfiler` 在 `editor/debugger/editor_performance_profiler.cpp`。`update_monitors()` 在 `:330` 根据运行时发来的 custom monitor 更新列表；`add_profile_frame()` 在 `:369` 把数值写入每个 monitor 的历史链表并重绘曲线；构造函数在 `:414` 创建左侧 Tree 和右侧绘图 Control。"
      },
      {
        type: "code",
        code: [
          "func _ready():",
          "    Performance.add_custom_monitor(",
          "        &\"game/enemy_count\",",
          "        Callable(self, \"_enemy_count\"),",
          "        [],",
          "        Performance.MONITOR_TYPE_QUANTITY",
          "    )",
          "",
          "func _enemy_count() -> int:",
          "    return get_tree().get_nodes_in_group(\"enemies\").size()"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个自定义 monitor 会进入 RemoteDebugger 的 PerformanceProfiler。下一次 `get_monitor_modification_time()` 变化后，运行时会发 `performance:profile_names`，编辑器 Monitors 面板会把它加到 Custom 分类下；后续 `performance:profile_frame` 会带上它的数值。"
      },
      {
        type: "code",
        code: [
          "var fps := Performance.get_monitor(Performance.TIME_FPS)",
          "var process_time := Performance.get_monitor(Performance.TIME_PROCESS)",
          "var physics_time := Performance.get_monitor(Performance.TIME_PHYSICS_PROCESS)",
          "",
          "# 这些读数来自运行时 Performance，不是编辑器面板临时计算出来的。"
        ].join("\n")
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：FPS 低但 process 时间不高"
      },
      {
        type: "paragraph",
        text: "先看 `TIME_PHYSICS_PROCESS`、渲染 draw call、pipeline compilation、显存和物理 monitor。`TIME_PROCESS` 只代表空闲帧 process 侧，不包括所有渲染/GPU/驱动等待。Performance 的 monitor 是不同系统的入口集合，不是一个总耗时树。"
      },
      {
        type: "subheading",
        title: "案例二：自定义 monitor 不显示"
      },
      {
        type: "paragraph",
        text: "检查 `add_custom_monitor()` 是否真的执行、id 是否重复、Callable 是否有效、返回值是否是数字。RemoteDebugger 会根据 `get_monitor_modification_time()` 判断是否重发 monitor 名称；如果只改了 Callable 内部逻辑但没有增删 monitor，名称列表不会变化，但 frame 值仍会更新。"
      },
      {
        type: "subheading",
        title: "案例三：Profiler 面板没有数据"
      },
      {
        type: "paragraph",
        text: "先确认运行时是否启用了调试连接：`EngineDebugger::is_active()` 需要 singleton 和 ScriptDebugger 都存在。再看编辑器是否发了 `profiler:<name>`，RemoteDebugger 是否有对应 profiler，`profiler_enable()` 是否把 active 设为 true。没有 active profiler，`EngineDebugger::iteration()` 不会调用 tick。"
      },
      {
        type: "subheading",
        title: "案例四：Monitors 和 Profiler 不是同一类数据"
      },
      {
        type: "paragraph",
        text: "Monitors 是 Performance monitor 数值曲线，通常每秒一组；Profiler 是脚本函数和服务器函数的耗时采样；Visual Profiler 偏 CPU/GPU 区域耗时。把 Monitors 的 FPS 曲线当作函数调用火焰图，或者把 Profiler 的函数耗时当作显存读数，都会走错源码路径。"
      },
      {
        type: "subheading",
        title: "案例五：调试器本身有开销"
      },
      {
        type: "paragraph",
        text: "开启 profiler、远程场景树刷新、视频内存检查、错误捕获都会增加消息和采样成本。`RemoteDebugger` 还有每秒字符和错误/警告限制。排查性能时要对比无调试器、只开 Monitors、开启脚本/视觉 profiler 的差异。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "调试和性能边界",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Performance", "提供 monitor 名称、类型、数值和 custom monitor。", "不负责网络传输和编辑器 UI。"],
          ["EngineDebugger", "维护 profiler/capture/协议，调用 tick，抽象 send_message。", "不生成具体 monitor 数值。"],
          ["RemoteDebugger", "把运行时调试消息发给编辑器，处理 profiler capture。", "不绘制曲线。"],
          ["ScriptDebugger", "断点、脚本栈、错误中断、调试状态。", "不管理 Performance monitor 枚举。"],
          ["ScriptEditorDebugger", "编辑器会话、消息解析、Profiler/Monitors 标签页。", "不直接测项目运行时。"],
          ["EditorPerformanceProfiler", "展示 Monitors 数据曲线和树。", "不采样 RenderingServer/PhysicsServer。"],
          ["EditorProfiler/VisualProfiler", "展示函数耗时和视觉 profiler 数据。", "不替代 Performance.get_monitor。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：编辑器自己测运行中项目性能。大多数数据来自运行时发回的消息。",
          "误区二：Performance 是 profiler。Performance 是 monitor 读数集合；Profiler 是按帧/函数/区域采样的调试通道。",
          "误区三：Monitors 每帧实时全量发送。RemoteDebugger 的 PerformanceProfiler 当前是约每秒发送一次。",
          "误区四：FPS 低只看 TIME_PROCESS。物理、渲染、GPU、导航、pipeline 编译、调试器开销都可能影响帧率。",
          "误区五：custom monitor 可以返回任意 Variant。PerformanceProfiler 要求数值，否则会报错并填空。",
          "误区六：Profiler 打开但没有数据一定是 UI bug。先查 EngineDebugger 是否 active、profiler 是否 registered/active。",
          "误区七：Visual Profiler、Profiler、Monitors 是一张表。它们走不同消息和不同 UI 控件。",
          "误区八：调试器没有性能影响。采样、序列化、网络传输和 UI 绘制都可能影响结果。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `main/performance.h:39`，把 Monitor 枚举和 MonitorType 列出来。",
          "读 `main/performance.cpp` 的 `get_monitor_name()`、`get_monitor()`、`get_monitor_type()`，确认每个读数来自哪个系统。",
          "读 `Main::setup()` 创建 Performance 的位置，再读 `main/main.cpp:5090` 和 `:5111` 附近的主循环更新时间。",
          "读 `core/debugger/engine_debugger.h:53` 和 `engine_debugger.cpp:107`，理解 profiler toggle/add/tick 和每帧 iteration。",
          "读 `core/debugger/engine_profiler.cpp:54`，看 EngineProfiler 怎样 bind 到 EngineDebugger。",
          "读 `core/debugger/remote_debugger.cpp:47` 和 `:775`，跟 PerformanceProfiler 如何绑定、启用和发送 `performance:*` 消息。",
          "读 `editor/debugger/script_editor_debugger.cpp` 的 `_init_parse_message_handlers()`、`_msg_performance_profile_frame()`、`_msg_servers_profile_common()`，理解编辑器消息分发。",
          "最后读 `editor/debugger/editor_performance_profiler.cpp:330` 和 `:369`，看 Monitors 面板如何维护树和历史曲线。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Performance 负责运行时读数，EngineDebugger 负责 profiler 回调和消息通道，RemoteDebugger 把数据送出，ScriptEditorDebugger 和各个 EditorProfiler 控件只负责解析并展示。"
      }
    ]
  },
  {
    id: "scriptextension",
    title: "ScriptLanguage / GDScript / GDExtension",
    aliases: ["ScriptServer", "ScriptLanguage", "Script", "ScriptInstance", "ScriptLanguageExtension", "ScriptInstanceExtension", "GDScript", "GDScriptLanguage", "GDScriptParser", "GDScriptAnalyzer", "GDScriptCompiler", "GDScriptByteCodeGenerator", "GDScriptFunction", "GDScriptInstance", "CSharpScript", "CSharpInstance", "CSharpLanguage", "Mono", "GDExtension", "GDExtensionManager", "GDExtensionLoader", "classdb_register_extension_class6", "ClassDB::register_extension_class", "Global Script Class", "class_name", "Variant", "Callable", "MethodBind", "script_instance", "set_script", "instance_create", "reload_all_scripts"],
    summary: "Godot 的脚本和扩展不是 Node 的特例：脚本语言通过 ScriptLanguage 注册，脚本文件是 Script 资源，Object 持有 ScriptInstance；GDExtension 则通过 C ABI 和 ClassDB 把动态库类接入同一套 Object/Variant/MethodBind 类型系统。",
    article: [
      {
        type: "lead",
        text: "脚本、模块与 GDExtension 这一层要抓住三个边界：语言怎样注册进引擎，脚本资源怎样挂到 Object 上运行，外部原生动态库怎样进入 ClassDB。GDScript、C#、扩展类看起来都能被 Inspector、脚本、场景和编辑器识别，是因为它们最终都接到 Object、ScriptInstance、Variant、Callable、MethodBind、ClassDB 这套共同运行时系统上。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 ScriptServer 想成“语言登记处”。GDScript、C# 或其他脚本语言先来登记：我叫什么、文件扩展名是什么、怎么加载、怎么补全、怎么调试、怎么创建实例。之后 ResourceLoader 看到 `.gd` 或 `.cs`，就能找到对应语言。"
      },
      {
        type: "paragraph",
        text: "Script 是脚本文件这个资源，ScriptInstance 是脚本挂到某个 Object 上以后真正运行的那份实例。一个 Node 还是 C++ Node，但它可以持有一个脚本实例；脚本里的导出变量、方法、信号会通过这个实例和 Object 的属性/调用系统合并起来。"
      },
      {
        type: "paragraph",
        text: "GDExtension 不是普通脚本语言，更像“外部 C++ 动态库入口”。扩展库通过一张 C ABI 函数表告诉引擎：我要注册一个类、它继承谁、有哪些方法属性信号、怎么创建实例。注册完成后，ClassDB 就能像认识内置 C++ 类一样认识它。"
      },
      {
        type: "flow",
        title: "从脚本文件到对象调用",
        steps: [
          { title: "语言注册", text: "模块创建 ScriptLanguage，并调用 `ScriptServer::register_language()`。" },
          { title: "脚本加载", text: "ResourceLoader 按扩展名找到语言和 ResourceFormatLoader，得到 Script 资源。" },
          { title: "脚本编译/加载", text: "GDScript reload 走 parse、analyze、compile；C# 走程序集和托管运行时。" },
          { title: "挂到 Object", text: "`Object::set_script()` 调 `Script::instance_create(this)` 创建 ScriptInstance。" },
          { title: "属性/方法桥接", text: "Object 的 set/get/call/notification/property list 会接入 ScriptInstance。" },
          { title: "公共值系统", text: "参数、返回值、属性都用 Variant/Callable/MethodBind/ClassDB 统一表示。" }
        ]
      },
      {
        type: "heading",
        title: "核心抽象"
      },
      {
        type: "paragraph",
        text: "`ScriptServer` 的入口在 `core/object/script_language.h:46`，实现集中在 `script_language.cpp`。它保存最多 16 个语言实例、按扩展名查语言、初始化/结束所有语言、维护 global script class 表，并给线程进入/退出脚本运行时提供 `thread_enter()`/`thread_exit()` 钩子。"
      },
      {
        type: "paragraph",
        text: "`ScriptLanguage` 的入口在 `script_language.h:208`。它不是只有“执行代码”一个函数，而是一整套语言插件接口：初始化、文件扩展名、模板、语法检查、补全、调试、profiling、热重载、外部编辑器、global class、运行时 frame 等。新增脚本语言要实现的是整个语言生态接口。"
      },
      {
        type: "paragraph",
        text: "`Script` 的入口在 `script_language.h:113`，它继承 Resource。`.gd`、`.cs` 等脚本文件最终都是 Script 资源，因此它们会和 ResourceLoader、缓存、路径、热重载、保存/导入逻辑交织。`ScriptInstance` 的接口在 `core/object/script_instance.h:36`，负责实例属性、方法、通知、脚本信号、RPC 配置、fallback 和语言对象。"
      },
      {
        type: "table",
        title: "脚本核心对象分工",
        headers: ["抽象", "源码入口", "负责什么", "不负责什么"],
        rows: [
          ["ScriptServer", "`core/object/script_language.h:46`、`script_language.cpp:239`", "语言登记、按扩展名找语言、初始化/结束语言、global class 表、线程钩子。", "不解析某一种脚本语法。"],
          ["ScriptLanguage", "`script_language.h:208`", "语言插件接口：加载、语法检查、补全、调试、profiling、reload、模板。", "不保存某个具体脚本文件的源码状态。"],
          ["Script", "`script_language.h:113`", "脚本资源基类，能实例化、reload、给出属性/方法/信号列表。", "不代表某个 Node 上的运行实例。"],
          ["ScriptInstance", "`script_instance.h:36`", "挂在 Object 上的脚本运行时实例，处理 set/get/call/notification。", "不拥有 Object 本体的 C++ 内存布局。"],
          ["Global Script Class", "`script_language.cpp:399` 附近", "把 `class_name` 变成全局类型名，供 Inspector、创建对话框、类型提示使用。", "不等同 ClassDB 原生类注册。"],
          ["ScriptLanguageExtension", "`core/object/script_language_extension.h:226`", "允许扩展实现 ScriptLanguage/ScriptInstance 接口。", "不等同普通 GDExtension 原生类注册路径。"]
        ]
      },
      {
        type: "heading",
        title: "ScriptServer 注册语言"
      },
      {
        type: "paragraph",
        text: "`ScriptServer::register_language()` 在 `core/object/script_language.cpp:239`。它加锁后检查语言非空、未超过 `MAX_LANGUAGES`、扩展名/名称/类型不重复，然后把语言放进 `_languages`。如果语言系统已经初始化过，注册新语言时会立刻调用 `p_language->init()`，这对编辑器首次导入某些扩展语言场景很重要。"
      },
      {
        type: "table",
        title: "ScriptServer 维护的状态",
        headers: ["状态", "含义", "典型使用"],
        rows: [
          ["`_languages` / `_language_count`", "已注册 ScriptLanguage 列表。", "ResourceLoader、调试器、热重载遍历语言。"],
          ["`languages_ready`", "语言是否已经 init。", "晚注册语言时决定是否立即 init。"],
          ["`global_classes`", "`class_name` 到语言/路径/基类/工具标记的映射。", "创建节点、类型提示、Inspector、脚本继承。"],
          ["`inheriters_cache`", "按基类缓存脚本继承者。", "编辑器列出可创建类型。"],
          ["`scripting_enabled`", "脚本运行是否启用。", "导入、恢复模式、tool 脚本行为。"],
          ["`thread_entered`", "线程是否进入过语言运行时。", "WorkerThreadPool 或自建线程要让语言准备线程状态。"]
        ]
      },
      {
        type: "heading",
        title: "Object 怎样持有脚本实例"
      },
      {
        type: "paragraph",
        text: "`Object` 内部有 `script_instance` 指针，声明在 `core/object/object.h:449`。`Object::set_script()` 在 `object.cpp:967`：先校验传入的是 Script，抽象脚本不能挂；如果已有脚本实例先删除；如果新脚本 `can_instantiate()`，就调用 `s->instance_create(this)`；编辑器下不能实例化时可创建 placeholder；最后通知属性列表变化并发 `script_changed`。"
      },
      {
        type: "paragraph",
        text: "当前源码里，`Object::set()` 和 `Object::get()` 都先问 `script_instance`。`Object::set()` 在 `object.cpp:233`，脚本实例 set 成功就返回；否则再问 GDExtension set、ClassDB 属性、script/metadata、fallback 和 `_setv()`。`Object::get()` 在 `object.cpp:316` 也先问脚本实例，再问 extension 和 ClassDB。"
      },
      {
        type: "paragraph",
        text: "`Object::callp()` 在 `object.cpp:768`。除 `free` 这个特殊方法外，它先调用 `script_instance->callp()`；如果脚本返回 `CALL_OK` 就直接返回；如果脚本说方法不存在，才继续查 `ClassDB::get_method(get_class_name(), p_method)` 并通过 MethodBind 调 C++。这意味着脚本方法、C++ 绑定方法和 extension 方法会在同一套 Object 调用入口里协作，但先后顺序要按源码确认。"
      },
      {
        type: "flow",
        title: "Object 属性和方法桥接",
        steps: [
          { title: "set/get 属性", text: "先尝试 ScriptInstance，再尝试 GDExtension set/get，再尝试 ClassDB 属性。" },
          { title: "script 属性", text: "导出变量和脚本成员能被 Inspector 看到，靠属性列表合并和 ScriptInstance get/set。" },
          { title: "callp 方法", text: "先让 ScriptInstance 处理；脚本没有该方法时落到 ClassDB MethodBind。" },
          { title: "notification", text: "Object 通知可以继续分发到脚本实例的 `_notification`。" },
          { title: "信号/属性列表", text: "Object 会把脚本信号、脚本属性、C++ 属性合并给编辑器和运行时。" }
        ]
      },
      {
        type: "code",
        code: [
          "# player.gd",
          "class_name PlayerController",
          "extends Node",
          "",
          "@export var speed := 320.0",
          "",
          "func _ready():",
          "    print(\"script instance is attached to this Node\")"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "`class_name` 会让脚本进入 global script class 表；`@export` 变量会进入脚本属性列表；`_ready()` 是脚本实例的方法。Node 本体仍是 C++ Node，但 Object 的属性、通知和方法调用路径会把这些脚本成员接进来。"
      },
      {
        type: "heading",
        title: "GDScript 编译和运行"
      },
      {
        type: "paragraph",
        text: "GDScript 模块在 `modules/gdscript/register_types.cpp:137` 的 `initialize_gdscript_module()` 中接入。SERVERS 初始化级别会 `GDREGISTER_CLASS(GDScript)`，创建 `GDScriptLanguage`，调用 `ScriptServer::register_language(script_language_gd)`，并注册 `.gd` 的 ResourceFormatLoader/Saver、GDScriptCache 和 utility functions。"
      },
      {
        type: "paragraph",
        text: "`GDScript` 类入口在 `modules/gdscript/gdscript.h:58`。`GDScript::reload()` 在 `gdscript.cpp:737` 串起完整编译链：创建 `GDScriptParser` 并 `parse()`，创建 `GDScriptAnalyzer` 并 `analyze()`，再用 `GDScriptCompiler` 编译到运行时函数。失败时会把 parse/analyze/compile 错误发给调试器或错误处理器。"
      },
      {
        type: "paragraph",
        text: "`GDScriptByteCodeGenerator` 在 `modules/gdscript/gdscript_byte_codegen.h:40`。`write_start()`/`write_end()` 生成 `GDScriptFunction`，`write_call_method_bind()` 在 `gdscript_byte_codegen.cpp:1301` 会把 MethodBind 调用写成 VM opcode。运行时 `GDScriptFunction::call()` 在 `gdscript_vm.cpp:499` 执行这些 opcode；`OPCODE_CALL_METHOD_BIND` 分支会调用已经解析好的 C++ MethodBind。"
      },
      {
        type: "table",
        title: "GDScript 阶段",
        headers: ["阶段", "源码入口", "输入", "输出/结果"],
        rows: [
          ["模块注册", "`modules/gdscript/register_types.cpp:137`", "模块初始化级别。", "GDScript 类、语言、loader/saver、cache 注册完成。"],
          ["资源加载", "`GDScript::load_source_code()`", "`.gd` 路径和文本。", "GDScript Resource 保存源码。"],
          ["Parser", "`GDScriptParser`", "源码文本或 binary tokens。", "语法树、parse errors、warnings。"],
          ["Analyzer", "`GDScriptAnalyzer`", "语法树。", "类型、名称、继承、成员解析结果。"],
          ["Compiler", "`GDScriptCompiler`", "分析后的语法树。", "GDScriptFunction、成员表、常量、内部类。"],
          ["Bytecode VM", "`GDScriptFunction::call()`", "opcode、Variant 参数、GDScriptInstance。", "执行脚本函数，调用 C++ MethodBind 或脚本函数。"],
          ["Language frame", "`GDScriptLanguage::frame()`", "每帧语言维护。", "profiling 计数滚动、热重载相关维护。"]
        ]
      },
      {
        type: "flow",
        title: ".gd reload 主流程",
        steps: [
          { title: "检查实例", text: "非 keep_state 且已有实例时拒绝 reload，避免破坏运行中实例。" },
          { title: "缓存处理", text: "更新 GDScriptCache 的 shallow script 和 parser cache。" },
          { title: "parse", text: "源码或 binary tokens 进入 GDScriptParser。" },
          { title: "analyze", text: "GDScriptAnalyzer 解析类型、符号、继承和错误。" },
          { title: "compile", text: "GDScriptCompiler 生成运行时函数和类结构。" },
          { title: "debug/doc", text: "工具构建生成文档，debug 构建处理警告和调试器断点。"}
        ]
      },
      {
        type: "heading",
        title: "C# / Mono 是同一接口下的另一套运行时"
      },
      {
        type: "paragraph",
        text: "C# 也走 ScriptLanguage/Script/ScriptInstance，但底层完全不同。`CSharpScript` 在 `modules/mono/csharp_script.h:56`，`CSharpInstance` 在 `:315`，`CSharpLanguage` 在 `:398`。mono 模块在 `modules/mono/register_types.cpp:47` 的 SCENE 初始化级别创建 `CSharpLanguage`，调用 `ScriptServer::register_language(script_language_cs)`，并注册 C# 资源 loader/saver。"
      },
      {
        type: "table",
        title: "GDScript 和 C# 的共同点与差异",
        headers: ["维度", "GDScript", "C# / Mono", "共同边界"],
        rows: [
          ["语言注册", "SERVERS 级别注册 `GDScriptLanguage`。", "SCENE 级别注册 `CSharpLanguage`。", "都进入 ScriptServer。"],
          ["脚本资源", "`GDScript : Script`。", "`CSharpScript : Script`。", "都可挂到 Object。"],
          ["脚本实例", "`GDScriptInstance`。", "`CSharpInstance`。", "都实现 ScriptInstance。"],
          ["编译/加载", "Parser -> Analyzer -> Compiler -> bytecode VM。", "程序集、托管类型、GDMono/.NET bridge。", "对 Object 侧都呈现属性/方法/信号。"],
          ["热重载", "围绕源码、parser/cache、函数替换和状态保存。", "围绕程序集、domain、托管对象句柄和实例恢复。", "都要保护已存在实例和编辑器状态。"]
        ]
      },
      {
        type: "heading",
        title: "GDExtension 进入 ClassDB"
      },
      {
        type: "paragraph",
        text: "`GDExtensionManager` 的入口在 `core/extension/gdextension_manager.h:38`。`load_extension()` 在 `gdextension_manager.cpp:115`，创建 loader，打开扩展库，并把 `GDExtension` 放进 manager 的 map。批量加载项目扩展在 `load_extensions()`，会读取扩展列表并调用平台额外扩展加载。"
      },
      {
        type: "paragraph",
        text: "`GDExtension::open_library()` 在 `core/extension/gdextension.cpp:804`，让 loader 打开动态库并调用初始化函数。`initialize_library()` 在 `:846` 按 CORE、SERVERS、SCENE、EDITOR 初始化级别进入扩展自己的初始化函数。接口函数表由 `GDExtension::initialize_gdextensions()` 建立，其中包括 `classdb_register_extension_class6`、注册方法、属性、信号、常量和 main loop callbacks 的 C ABI 函数。"
      },
      {
        type: "paragraph",
        text: "扩展注册类时会调用 `classdb_register_extension_class6`，进入 `GDExtension::_register_extension_class6()`，再到 `_register_extension_class_internal()`。内部校验类名、父类、热重载限制，填充 `ObjectGDExtension` 回调表，调用 `create_gdtype()`，最后在 `gdextension.cpp:566` 调 `ClassDB::register_extension_class(&extension->gdextension)`。`ClassDB::register_extension_class()` 在 `core/object/class_db.cpp:2246`，把扩展类放进与内置类同一张类表。"
      },
      {
        type: "flow",
        title: "GDExtension 注册类路径",
        steps: [
          { title: "发现 .gdextension", text: "GDExtensionManager 读取项目扩展列表或平台扩展。" },
          { title: "打开动态库", text: "GDExtensionLoader 打开库并拿到初始化函数。" },
          { title: "初始化级别", text: "Manager 按 CORE/SERVERS/SCENE/EDITOR 调 initialize_library。" },
          { title: "取接口函数", text: "扩展通过 get_proc_address 获取 `classdb_register_extension_class6` 等函数。" },
          { title: "注册类信息", text: "扩展提交类名、父类、set/get/call/create/free 等回调。" },
          { title: "ClassDB", text: "Godot 把 ObjectGDExtension 写入 ClassDB，扩展类成为运行时类型。" }
        ]
      },
      {
        type: "table",
        title: "GDExtension 和脚本语言的边界",
        headers: ["机制", "是什么", "接入点", "典型能力"],
        rows: [
          ["GDExtension", "原生动态库 ABI。", "`core/extension/gdextension.*`。", "注册原生类、方法、属性、信号、常量、虚函数、main loop callbacks。"],
          ["ScriptLanguageExtension", "通过 GDExtension 实现脚本语言接口的扩展类。", "`script_language_extension.h`。", "让扩展实现 ScriptLanguage/Script/ScriptInstance。"],
          ["普通 GDScript", "内置脚本语言模块。", "`modules/gdscript` + ScriptServer。", "解析 `.gd`，运行字节码。"],
          ["C# / Mono", "模块提供的托管语言 bridge。", "`modules/mono` + ScriptServer。", "加载程序集，桥接托管对象。"],
          ["ClassDB extension class", "扩展注册后的 Object 类型。", "`ClassDB::register_extension_class()`。", "像内置 C++ 类一样被脚本和 Inspector 识别。"]
        ]
      },
      {
        type: "heading",
        title: "共同语言：Variant、Callable、MethodBind、ClassDB"
      },
      {
        type: "paragraph",
        text: "不同语言能互通，靠的是统一运行时类型系统。ClassDB 保存类、继承、方法、属性、信号、常量和扩展类；MethodBind 把 C++ 成员函数包装成可由 Variant 参数调用的对象；Variant 是跨 C++、GDScript、C#、GDExtension、Inspector 和序列化的公共值容器；Callable 把对象方法、脚本函数、lambda、扩展回调和信号连接统一成可调用目标。"
      },
      {
        type: "table",
        title: "公共运行时机制",
        headers: ["机制", "源码入口", "脚本/扩展为什么依赖它"],
        rows: [
          ["ClassDB", "`core/object/class_db.h:97`", "统一保存内置类和扩展类的元数据。"],
          ["MethodBind", "`core/object/method_bind.h:38`", "C++ 方法能被脚本和扩展以统一签名调用。"],
          ["Variant", "`core/variant/variant.h:93`", "跨语言参数、返回值、属性和序列化值容器。"],
          ["Callable", "`core/variant/callable.h:48`", "统一表示可调用目标，支持延迟调用、信号、tween callback。"],
          ["PropertyInfo / MethodInfo", "`core/object/object.h`、`method_bind.h`", "Inspector、文档、补全和序列化理解属性/方法的元数据。"],
          ["ObjectDB / ObjectID", "`core/object/object.h`", "Callable、Variant Object、脚本实例桥接要确认对象是否还活着。"]
        ]
      },
      {
        type: "code",
        code: [
          "// C++ 侧概念化路径，省略错误处理：",
          "Object *obj = some_node;",
          "Ref<Script> script = ResourceLoader::load(\"res://player.gd\");",
          "obj->set_script(script); // 内部调用 script->instance_create(obj)。",
          "",
          "Callable::CallError ce;",
          "obj->callp(SNAME(\"_ready\"), nullptr, 0, ce);",
          "// callp 会先给 script_instance，脚本没有该方法才落到 ClassDB MethodBind。"
        ].join("\n")
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "subheading",
        title: "案例一：导出变量为什么出现在 Inspector"
      },
      {
        type: "paragraph",
        text: "脚本实例会提供 property list，Object 的属性列表会合并脚本属性。Inspector 不需要为每个脚本手写 UI，它读取 PropertyInfo，再根据 Variant 类型创建编辑控件。修改时 Object::set() 先给 ScriptInstance，所以导出变量能像 C++ 属性一样被设置。"
      },
      {
        type: "subheading",
        title: "案例二：GDScript 调 C++ 方法为什么能快"
      },
      {
        type: "paragraph",
        text: "GDScript 编译阶段能解析出 MethodBind 的调用，会生成 `OPCODE_CALL_METHOD_BIND` 或 validated 变体。运行时 VM 不一定每次都从字符串慢查，而是直接用编译期写入的 MethodBind 和 Variant 参数调用 C++。"
      },
      {
        type: "subheading",
        title: "案例三：脚本热重载失败"
      },
      {
        type: "paragraph",
        text: "GDScript reload 会检查是否已有实例；非 keep_state 且有实例时会拒绝 reload。再往下才是 parse/analyze/compile。排查时先分清是“实例存在导致不能 reload”、语法错误、类型分析错误，还是编译错误。"
      },
      {
        type: "subheading",
        title: "案例四：GDExtension 类在 Inspector 里不可见"
      },
      {
        type: "paragraph",
        text: "先查扩展是否被 GDExtensionManager 加载，初始化级别是否到达，注册类时父类是否存在，`is_exposed` 是否正确，是否最终调用到 `ClassDB::register_extension_class()`。如果类注册成功但属性不可见，再查 extension 提供的 property list、get/set 回调和 PropertyInfo。"
      },
      {
        type: "subheading",
        title: "案例五：C# 和 GDScript 热重载不能套同一套结论"
      },
      {
        type: "paragraph",
        text: "二者都挂 ScriptInstance，但 GDScript 主要围绕源码和字节码函数，C# 还要处理程序集、托管对象句柄、domain/bridge、状态备份和恢复。看到同样的 Object::set_script() 入口，不代表语言内部 reload 细节相同。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "脚本与扩展边界",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["Object", "持有 script_instance，统一 set/get/call/notification。", "不解析脚本语言语法。"],
          ["Script", "脚本资源生命周期和实例化入口。", "不代表某个对象上的运行状态。"],
          ["ScriptInstance", "对象上的脚本运行实例。", "不注册内置 C++ 类。"],
          ["ScriptLanguage", "语言生态接口。", "不等于某个脚本文件。"],
          ["GDScript", "内置 `.gd` 语言实现。", "不负责 C# 程序集和托管对象。"],
          ["GDExtension", "动态库 ABI 和扩展类注册。", "不是默认的脚本语言系统。"],
          ["ClassDB", "统一元数据和方法绑定表。", "不保存每个脚本实例的成员值。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：脚本修改了 Node 的 C++ 类定义。实际是 Object 持有 ScriptInstance，反射路径把脚本成员合并进来。",
          "误区二：GDScript 只是解释器。它有 Parser、Analyzer、Compiler、ByteCodeGenerator、VM、debugger、profiler、LSP 和热重载。",
          "误区三：Object 调用一定先查 C++。当前 `Object::callp()` 先尝试 script_instance，脚本方法不存在才落到 ClassDB。",
          "误区四：C# 和 GDScript 内部机制相同。它们共享 ScriptLanguage 边界，但运行时和 reload 实现不同。",
          "误区五：GDExtension 是普通脚本语言。它主要是原生 ABI 和 ClassDB 扩展类注册机制。",
          "误区六：Variant 只是方便传参的容器。它是脚本、编辑器、序列化、GDExtension ABI 和 MethodBind 的共同值边界。",
          "误区七：class_name 等于 ClassDB 原生类。它进入 global script class 表，不等于内置 C++ 类注册。",
          "误区八：扩展类注册成功就一定能显示所有属性。属性还要靠 property list、get/set、PropertyInfo 和 Inspector 侧处理。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/object/script_language.h`，区分 ScriptServer、ScriptLanguage、Script、ScriptInstance。",
          "读 `ScriptServer::register_language()`、`init_languages()` 和 global class 相关函数，理解语言和 class_name 如何登记。",
          "读 `Object::set_script()`，确认 Script 怎样创建 ScriptInstance 并挂到 Object。",
          "读 `Object::set()`、`get()`、`callp()`，确认脚本实例、extension 和 ClassDB 的先后关系。",
          "读 `modules/gdscript/register_types.cpp`，看 GDScript 模块怎样注册语言和资源 loader/saver。",
          "读 `GDScript::reload()`，按 Parser、Analyzer、Compiler、ByteCodeGenerator、GDScriptFunction 建立编译链。",
          "读 `modules/mono/register_types.cpp` 和 `csharp_script.h/cpp`，只对比 ScriptLanguage/ScriptInstance 边界，不要套用 GDScript 内部实现。",
          "读 `GDExtensionManager::load_extension()`、`GDExtension::open_library()`、`initialize_library()` 和 `_register_extension_class_internal()`，跟到 `ClassDB::register_extension_class()`。",
          "最后回到 ClassDB、MethodBind、Variant、Callable，理解不同语言为什么能共享同一套 Object API。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "脚本语言通过 ScriptServer/ScriptLanguage 接入，脚本文件是 Script 资源，Object 上运行的是 ScriptInstance；GDExtension 通过 C ABI 把动态库类注册进 ClassDB。所有这些最终都靠 Variant、Callable、MethodBind 和 ClassDB 互通。"
      }
    ]
  },
  {
    id: "editorarchitecture",
    title: "Editor / EditorNode / EditorPlugin / Inspector",
    aliases: ["Editor", "EditorNode", "EditorData", "EditorSelection", "EditorUndoRedoManager", "Inspector", "InspectorDock", "EditorInspector", "EditorProperty", "EditorInspectorPlugin", "EditorPlugin", "EditorPlugins", "SceneTreeDock", "FileSystemDock", "ImportDock", "EditorFileSystem", "ResourceFormatImporter", "ResourceImporter", "EditorExport", "EditorExportPreset", "EditorExportPlatform", "EditorRunBar", "EditorDebuggerNode", "ProjectManager", "TOOLS_ENABLED", "Docks", "Import", "Export"],
    summary: "Godot 编辑器本身也是运行在引擎里的工具场景：Main 创建 SceneTree 后挂上 EditorNode，EditorData 保存编辑状态，Inspector 通过 Object/ClassDB/ScriptInstance 的属性列表生成 UI，EditorPlugin 提供主屏、dock、导入导出、gizmo、调试器等扩展点。",
    article: [
      {
        type: "lead",
        text: "编辑器这一层最容易读乱，因为它把 UI、资源、场景、撤销、导入、导出、插件、调试器都串在一起。稳定的理解方式是：`EditorNode` 是主控节点，`EditorData` 管编辑状态，Dock 是可见工具面板，Inspector 把 Object 的属性元数据变成控件，EditorPlugin 把专用工具接入编辑器，导入导出和运行调试再分别走自己的管线。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Godot 编辑器想成一个用 Godot 自己做出来的超大型应用。普通游戏里有一棵 SceneTree，编辑器里也有一棵 SceneTree；只不过这棵树的主角不是你的玩家角色，而是 `EditorNode`，它下面挂着菜单、场景面板、Inspector、文件系统面板、运行按钮、导入导出窗口和各种插件。"
      },
      {
        type: "paragraph",
        text: "Inspector 不是给每个节点手写一套属性面板。它先问对象：“你有哪些属性？”对象会把 C++ 绑定属性、脚本导出变量、GDExtension 属性合在一起返回；Inspector 再按 Variant 类型、hint、usage 生成勾选框、数字输入、资源选择器等控件。改属性时也不是直接硬写，而是尽量通过 `EditorUndoRedoManager` 记录可撤销操作。"
      },
      {
        type: "paragraph",
        text: "`EditorPlugin` 像编辑器的插座。2D/3D 编辑器、动画编辑器、调试器、版本控制、很多资源专用编辑器，以及用户写的 addons，都可以通过它声明自己能处理什么对象、要不要主屏、要不要 dock、要不要自定义 Inspector、要不要导入器或导出器。"
      },
      {
        type: "flow",
        title: "编辑器启动总流程",
        steps: [
          { title: "Main::start", text: "`main/main.cpp:3988` 判断这次是 editor、project manager、导出命令还是游戏。" },
          { title: "创建 SceneTree", text: "主循环仍然是 Godot 的 SceneTree/窗口/Server 体系，不是外部 GUI 程序。" },
          { title: "挂 EditorNode", text: "`main/main.cpp:4585` 创建 `memnew(EditorNode)`，`4586` 加到 root 下。" },
          { title: "EditorNode 构造", text: "`editor/editor_node.cpp:8377` 开始装配设置、快捷键、Server 状态、UndoRedo、GDExtension 重载等。" },
          { title: "创建 Dock", text: "`9188` 创建 SceneTreeDock，`9194` 创建 FileSystemDock，`9201` 创建 InspectorDock。" },
          { title: "加载插件", text: "`9445` 批量创建内置插件，`9448` 以后接入 GDExtension editor plugin。" }
        ]
      },
      {
        type: "heading",
        title: "核心对象分工"
      },
      {
        type: "table",
        title: "编辑器不是一个类做完所有事",
        headers: ["对象", "源码入口", "负责什么", "读源码时先问什么"],
        rows: [
          ["EditorNode", "`editor/editor_node.h:120`、`editor_node.cpp:8377`", "编辑器主控节点：菜单、dock、场景标签、打开保存、运行栏、插件、导入导出入口。", "用户点的按钮最终连到 EditorNode 哪个方法，还是转交给子系统？"],
          ["EditorData", "`editor/editor_data.h:105`、`editor_data.cpp:490`", "保存打开的场景标签、当前场景、插件列表、选择历史、UndoRedo 管理器。", "当前编辑状态存在这里，还是存在具体 Dock 或资源里？"],
          ["EditorSelection", "`editor/editor_data.h:275`、`editor_data.cpp:1240`", "保存当前选中的节点，并允许插件为选中对象提供额外编辑数据。", "问题是选择集变化，还是 Inspector 对象变化？"],
          ["EditorUndoRedoManager", "`editor/editor_undo_redo_manager.h:36`、`editor_undo_redo_manager.cpp:58`", "按场景、全局、远程对象划分 UndoRedo history，记录 do/undo property 和 method。", "这次修改是否进入撤销历史，是否标记场景 dirty？"],
          ["InspectorDock / EditorInspector", "`editor/docks/inspector_dock.h:46`、`editor_inspector.h:731`", "把选中对象的 PropertyInfo 列表转成 EditorProperty 控件，并处理属性写入。", "属性没显示是元数据问题，还是 Inspector 插件/usage/filter 问题？"],
          ["EditorPlugin", "`editor/plugins/editor_plugin.h:59`、`editor_plugin.cpp:361`", "编辑器扩展点：主屏、dock、Inspector 插件、导入器、导出器、gizmo、调试器、运行参数。", "这个功能是内置插件、addon、GDExtension editor plugin，还是 EditorNode 本体？"]
        ]
      },
      {
        type: "paragraph",
        text: "`EditorNode::EditorNode()` 一开始会调整引擎运行环境：编辑器里默认关闭 2D/3D 物理活动、关闭普通脚本执行、开启一些资源和渲染调试状态，并连接 `EditorUndoRedoManager`、`ProjectSettings`、`GDExtensionManager` 等信号。源码在 `editor/editor_node.cpp:8377` 到 `8433`。这说明编辑器虽然复用运行时，但它运行时的默认开关和游戏不同。"
      },
      {
        type: "table",
        title: "启动装配关键点",
        headers: ["阶段", "源码锚点", "发生什么", "影响"],
        rows: [
          ["决定模式", "`main/main.cpp:3988`", "`Main::start()` 解析 editor/project manager/export/game 分支。", "同一个可执行文件根据参数进入不同入口。"],
          ["创建编辑器", "`main/main.cpp:4585`", "创建 `EditorNode` 并加到 SceneTree root。", "编辑器是场景树里的节点，不是独立外壳。"],
          ["编辑器环境", "`editor_node.cpp:8401`、`8410`", "编辑器默认不跑普通物理和普通脚本，只允许 tool 脚本等工具侧行为。", "读运行时 bug 时要区分 editor hint 和游戏运行。"],
          ["核心 Dock", "`editor_node.cpp:9188`、`9194`、`9201`", "创建 SceneTreeDock、FileSystemDock、InspectorDock。", "左侧树、文件系统、右侧属性面板都是 dock 子系统。"],
          ["内置插件", "`editor_node.cpp:9444`", "遍历 `EditorPlugins::create(i)` 加入大量内置插件。", "很多编辑工具不在 EditorNode 本体里。"],
          ["扩展插件", "`editor_node.cpp:9448`、`4415`", "GDExtension 注册的 EditorPlugin 类也能被实例化并加入。", "原生扩展可以参与编辑器能力。"]
        ]
      },
      {
        type: "heading",
        title: "打开和保存场景"
      },
      {
        type: "flow",
        title: "打开 .tscn 的编辑器路径",
        steps: [
          { title: "通用入口", text: "`EditorNode::load_scene_or_resource()` 在 `editor_node.cpp:1713`，先判断路径是场景还是普通资源。" },
          { title: "建立标签", text: "`EditorNode::load_scene()` 在 `4854`，必要时 `editor_data.add_edited_scene(-1)` 创建一个编辑场景槽。" },
          { title: "加载 PackedScene", text: "`4906` 调 `ResourceLoader::load()` 得到 PackedScene。" },
          { title: "实例化编辑树", text: "`4953` 调 `PackedScene::instantiate()`，使用编辑态生成参数。" },
          { title: "同步当前场景", text: "`4973` 调 `set_edited_scene()`，再把根节点同步给 SceneTreeDock 和 SceneTree。" },
          { title: "恢复编辑状态", text: "后续读取折叠、选择、插件状态、live edit root 等编辑器状态。"}
        ]
      },
      {
        type: "paragraph",
        text: "`set_edited_scene_root()` 在 `editor/editor_node.cpp:4610`。它会写入 `EditorData::set_edited_scene_root()`，再调用 `SceneTreeDock::set_edited_scene()`，并把当前编辑根节点交给 `SceneTree::set_edited_scene_root()`。所以“当前场景”不是一个全局字符串，而是一组编辑状态、UI 面板和 SceneTree 标记一起同步。"
      },
      {
        type: "flow",
        title: "保存场景的真实路径",
        steps: [
          { title: "保存入口", text: "`EditorNode::_save_scene()` 在 `editor_node.cpp:2481`。" },
          { title: "保存前通知", text: "`2496` 给场景发 `NOTIFICATION_EDITOR_PRE_SAVE`，`2498` 让编辑器插件 apply changes。" },
          { title: "打包场景", text: "`2520` 用 `PackedScene::pack(scene)` 从当前 Node 树生成可保存状态。" },
          { title: "保存资源", text: "`2533` 调 `ResourceSaver::save()` 写到磁盘。" },
          { title: "通知插件", text: "`2536` 发 `scene_saved`，`2537` 调 `editor_data.notify_scene_saved()`。" }
        ]
      },
      {
        type: "table",
        title: "打开/保存链上的常见误判",
        headers: ["现象", "真正相关的系统", "检查点"],
        rows: [
          ["打开场景后树里节点不对", "ResourceLoader、PackedScene、SceneState、EditorData", "先看 `ResourceLoader::load()` 得到的 PackedScene，再看 instantiate 编辑态。"],
          ["保存后 .tscn 缺节点", "owner、PackedScene::pack、ResourceSaver", "保存不是 dump 当前指针；只有 owner 合法、可序列化的节点和资源会进入场景。"],
          ["切换标签状态丢失", "EditorData、插件 state、编辑器状态 cfg", "看 `save_edited_scene_state()`、`restore_edited_scene_state()` 和插件 state。"],
          ["外部资源变化后场景跟着变", "EditorFileSystem、ResourceCache、reload_instances_with_path_in_edited_scenes", "看 `EditorNode::reload_scene()` 和实例重载链。"]
        ]
      },
      {
        type: "heading",
        title: "Inspector 属性编辑"
      },
      {
        type: "flow",
        title: "从选中节点到属性控件",
        steps: [
          { title: "选择对象", text: "`EditorSelection::add_node()` 在 `editor_data.cpp:1240`，保存选中节点 ObjectID。" },
          { title: "Inspector 接收", text: "`EditorInspector::edit(Object *)` 在 `editor_inspector.cpp:5232` 设置当前编辑对象。" },
          { title: "取属性列表", text: "`editor_inspector.cpp:4230` 调 `object->get_property_list(&plist, true)`。" },
          { title: "插件解析", text: "`EditorInspectorPlugin::parse_begin()` 和属性解析钩子可以插入自定义控件。" },
          { title: "生成控件", text: "按 `PropertyInfo` 的 Variant type、hint、usage 创建 `EditorProperty`。" },
          { title: "写入属性", text: "`EditorInspector::_edit_set()` 在 `5481`，优先通过 UndoRedo 记录 do/undo property。" }
        ]
      },
      {
        type: "table",
        title: "Inspector 能看到哪些属性",
        headers: ["来源", "进入属性列表的方式", "写入时会发生什么"],
        rows: [
          ["C++ 绑定属性", "`_bind_methods()` 里的 `ADD_PROPERTY` 或 `_get_property_list()`。", "`object->set()` 通过 ClassDB setter、Object set 或自定义 set 路径。"],
          ["GDScript 导出变量", "脚本实例把导出变量合并到 Object 属性列表。", "先走 `ScriptInstance` 的 set/get，再进入脚本变量。"],
          ["GDExtension 属性", "扩展类注册 set/get/property list 回调后进入 ClassDB/ObjectGDExtension。", "Object 的 extension set/get/call 回调处理真实读写。"],
          ["Inspector 插件控件", "`EditorPlugin::add_inspector_plugin()` 调 `EditorInspector::add_inspector_plugin()`。", "插件可以添加专用控件，但仍应走对象属性或 UndoRedo。"],
          ["内部或只读属性", "由 `PropertyUsageFlags`、hint、read-only 状态决定是否显示或可编辑。", "可能显示但禁用，也可能被过滤器或 usage 隐藏。"]
        ]
      },
      {
        type: "code",
        language: "cpp",
        title: "Inspector 写属性的关键形状",
        code: `void EditorInspector::_edit_set(const String &p_name, const Variant &p_value, bool p_refresh_all, const String &p_changed_field) {
    EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

    if (!undo_redo || bool(object->call("_dont_undo_redo"))) {
        object->set(p_name, p_value);
        return;
    }

    undo_redo->create_action(vformat(TTR("Set %s"), p_name), UndoRedo::MERGE_ENDS, nullptr, false, mark_unsaved);
    undo_redo->add_do_property(object, p_name, p_value);
    undo_redo->add_undo_property(object, p_name, object->get(p_name));
    undo_redo->commit_action();
}`
      },
      {
        type: "paragraph",
        text: "上面的代码是简化后的形状，实际源码还会处理 MultiNodeEdit、远程调试对象、Control 布局状态、关联属性、数组移动、刷新请求和资源引用计数。重点是：Inspector 不是孤立 UI，它依赖 Object 的属性系统，也依赖编辑器专用 UndoRedo。"
      },
      {
        type: "heading",
        title: "EditorPlugin 扩展口"
      },
      {
        type: "paragraph",
        text: "`EditorPlugin` 的默认 `make_visible()`、`edit()`、`handles()` 在 `editor/plugins/editor_plugin.cpp:361` 到 `369`，本质是转调脚本/扩展虚函数。`EditorNode::add_editor_plugin()` 在 `editor_node.cpp:4384`，会把插件加入主屏、`EditorData` 和节点树。addons 启用时，`set_addon_plugin_enabled()` 会加载 plugin.cfg 指向的脚本，检查它是否继承 `EditorPlugin` 且是 tool 脚本，再实例化并 `set_script()`。关键检查在 `editor_node.cpp:4511`、`4532`、`4542`。"
      },
      {
        type: "table",
        title: "EditorPlugin 能接入的编辑器位置",
        headers: ["扩展点", "源码入口", "典型用途"],
        rows: [
          ["主屏 / 可见性", "`has_main_screen()`、`make_visible()`、`edit()`、`handles()`", "2D、3D、Script、Animation 等大编辑器主屏。"],
          ["底部面板 / Dock", "`add_control_to_bottom_panel()`、`add_control_to_dock()`", "调试器、输出、版本控制、资源专用工具面板。"],
          ["Inspector 插件", "`EditorPlugin::add_inspector_plugin()` 在 `editor_plugin.cpp:489`", "给某类对象增加自定义属性 UI 或专用编辑器。"],
          ["导入插件", "`add_import_plugin()` 在 `editor_plugin.cpp:439`", "支持新资源格式，或者覆盖导入选项和生成资源。"],
          ["导出插件/平台", "`add_export_plugin()`、`add_export_platform()` 在 `459`、`469`", "修改导出文件集合、添加平台导出能力。"],
          ["3D gizmo", "`add_node_3d_gizmo_plugin()` 在 `479`", "给 3D 节点提供可拖拽手柄和视口辅助显示。"],
          ["调试器插件", "`add_debugger_plugin()` 在 `588`", "增加调试会话 UI 或协议处理。"]
        ]
      },
      {
        type: "code",
        language: "gdscript",
        title: "最小 EditorPlugin 形状",
        code: `@tool
extends EditorPlugin

var panel := VBoxContainer.new()

func _enter_tree():
    panel.name = "My Tool"
    add_control_to_dock(DOCK_SLOT_RIGHT_UL, panel)

func _exit_tree():
    remove_control_from_docks(panel)
    panel.queue_free()

func _handles(object):
    return object is MeshInstance3D

func _edit(object):
    # 这里更新 panel，让它显示当前 MeshInstance3D 的专用编辑 UI。
    pass`
      },
      {
        type: "heading",
        title: "导入、导出、运行和调试"
      },
      {
        type: "table",
        title: "编辑器外围管线",
        headers: ["管线", "核心入口", "关键行为", "和 EditorNode 的关系"],
        rows: [
          ["导入", "`ResourceFormatImporter` 在 `core/io/resource_importer.h:41`，`EditorFileSystem::reimport_files()` 在 `editor_file_system.cpp:3248`", "读取 `.import` 元数据，按 importer order 调 `ResourceImporter::import()` 生成内部资源。", "EditorNode 构造时创建内置 importer，Dock/文件系统触发扫描和重导入。"],
          ["文件系统", "`EditorFileSystem` 在 `editor_file_system.h:145`", "扫描 `res://`，记录资源类型、UID、脚本类、导入状态和依赖。", "FileSystemDock 显示它，导出和资源加载也会依赖它的结果。"],
          ["导出", "`EditorExportPreset` 在 `editor_export_preset.h:38`，`EditorExportPlatform::export_project_files()` 在 `editor_export_platform.cpp:1312`", "按 preset 的过滤规则、平台、脚本导出模式和插件决定包内文件。", "EditorNode 提供导出 UI 和命令行导出入口，具体打包由 export 子系统做。"],
          ["运行", "`EditorRunBar::_run_scene()` 在 `editor_run_bar.cpp:256`，`EditorNode::call_run_scene()` 在 `editor_node.cpp:7707`", "保存/检查当前场景，组装运行参数，启动项目运行实例。", "插件有机会通过 `run_scene()` 修改参数或阻止运行。"],
          ["调试", "`EditorDebuggerNode` 在 `editor_debugger_node.h:47`，`start()` 在 `editor_debugger_node.cpp:283`", "打开调试服务器、接收运行项目发回的 profiler、消息和远程对象。", "DebuggerEditorPlugin 把调试器面板接入编辑器 UI。"]
        ]
      },
      {
        type: "flow",
        title: "导入资源和加载资源的关系",
        steps: [
          { title: "看到源文件", text: "EditorFileSystem 扫描 `res://texture.png`、`model.glb`、`sound.wav` 等源文件。" },
          { title: "选择 importer", text: "ResourceFormatImporter 根据扩展名、`.import` 元数据和 importer 名称选择 ResourceImporter。" },
          { title: "生成内部资源", text: "EditorFileSystem 重导入时调用 importer，写出 `.ctex`、`.res`、内部 PackedScene 等资源。" },
          { title: "ResourceLoader 加载", text: "运行时或编辑器加载源路径时，ResourceFormatImporter 映射到内部资源路径。" },
          { title: "导出打包", text: "EditorExportPlatform 决定源文件、导入结果、依赖和插件额外文件如何进入导出包。" }
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "table",
        title: "用源码链解释常见问题",
        headers: ["问题", "应该沿哪条链看", "不要只看哪里"],
        rows: [
          ["为什么导出变量会出现在 Inspector？", "脚本导出变量 -> ScriptInstance property list -> Object::get_property_list -> EditorInspector::update_tree -> EditorProperty。", "不要只看 InspectorDock UI。"],
          ["为什么改 Inspector 属性能撤销？", "EditorProperty emit_changed -> EditorInspector::_edit_set -> EditorUndoRedoManager -> Object::set。", "不要直接在控件里 `object->set()` 后结束。"],
          ["为什么启用 addon 后没有面板？", "plugin.cfg -> ResourceLoader 加载脚本 -> 检查继承 EditorPlugin 和 tool -> add_editor_plugin -> add_control_to_dock/main screen。", "不要只看 addons 目录是否存在。"],
          ["为什么模型导入后不是原始 glb？", "EditorFileSystem reimport -> ResourceImporterScene -> 内部 PackedScene/资源 -> ResourceFormatImporter 映射加载。", "不要把源文件和运行时加载资源当成同一个文件。"],
          ["为什么导出包少文件？", "EditorExportPreset filter -> EditorExportPlatform::export_project_files -> export plugins -> dependency/resource filter。", "不要只看 FileSystemDock 里能不能看到文件。"],
          ["为什么编辑器运行和游戏运行结果不同？", "EditorNode 构造时关闭普通脚本/物理，tool 脚本、editor hint、remote debug、运行实例是不同上下文。", "不要把编辑器内预览当成最终游戏主循环。"]
        ]
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "编辑器和周边系统的边界",
        headers: ["概念", "编辑器这层负责", "边界外的系统"],
        rows: [
          ["EditorNode vs SceneTree", "EditorNode 是挂在 SceneTree 里的主控节点。", "SceneTree 仍负责节点生命周期、窗口、当前 edited_scene_root。"],
          ["Inspector vs Object", "Inspector 把属性元数据变成 UI。", "属性从 Object/ClassDB/ScriptInstance/GDExtension 来，真实读写仍回到 Object。"],
          ["EditorPlugin vs Module", "EditorPlugin 是编辑器扩展实例。", "模块是编译进引擎的代码组织；模块可以注册插件，但不是同一层概念。"],
          ["Import vs ResourceLoader", "导入生成内部资源和 .import 元数据。", "ResourceLoader 负责按路径和 loader 加载资源，可能通过 ResourceFormatImporter 跳到内部资源。"],
          ["Export vs Save Scene", "导出决定整个项目哪些资源和二进制进入平台包。", "保存场景只是把当前 Node 树 pack 成 PackedScene 并写文件。"],
          ["Editor Debugger vs EngineDebugger", "编辑器端显示和控制调试会话。", "运行项目端通过 EngineDebugger/RemoteDebugger 发送消息。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "`EditorNode` 不是所有编辑器逻辑的唯一所在地；它更多是装配和调度中心。",
          "Dock 主要是 UI 面板，复杂业务通常会进入 EditorData、UndoRedo、ResourceImporter、ResourceSaver、Export、Debugger 等子系统。",
          "Inspector 自动显示属性靠的是 Object 属性系统和 PropertyInfo，不是编辑器给每个 Node 手写面板。",
          "addon 的 EditorPlugin 必须是 tool 脚本，并且脚本基类必须继承 EditorPlugin；否则加载会被 EditorNode 拒绝。",
          "编辑器能打开资源不代表导出包一定包含它；导出由 preset、平台、依赖、过滤规则和导出插件共同决定。",
          "`TOOLS_ENABLED` 下的 editor 代码通常不会进入普通导出模板，运行时代码不能随意依赖 editor-only 类型。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `main/main.cpp:3988` 附近的 `Main::start()`，看 editor、project manager、export、game 分支如何分开。",
          "读 `editor/editor_node.h:120` 和 `EditorNode::EditorNode()`，建立“主控节点装配编辑器”的整体图。",
          "沿 `editor_node.cpp:9188`、`9194`、`9201` 看 SceneTreeDock、FileSystemDock、InspectorDock 怎样挂入 UI。",
          "读 `EditorNode::load_scene_or_resource()`、`load_scene()`、`set_edited_scene_root()` 和 `_save_scene()`，串起打开/保存场景链。",
          "读 `editor/editor_data.h`、`editor_data.cpp:490`、`613`、`1240`，理解插件列表、场景标签和选择集。",
          "读 `editor/inspector/editor_inspector.cpp:4230` 和 `5481`，把属性列表、EditorProperty、UndoRedo 串起来。",
          "读 `editor/plugins/editor_plugin.h/cpp`，再回到 `EditorNode::add_editor_plugin()` 和 addon 启用逻辑。",
          "最后读 `EditorFileSystem::reimport_files()`、`ResourceFormatImporter`、`EditorExportPlatform::export_project_files()`、`EditorRunBar::_run_scene()`、`EditorDebuggerNode::start()`，把导入、导出、运行、调试补全。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Godot 编辑器是一棵运行在引擎里的工具场景：EditorNode 负责装配和调度，EditorData 保存编辑状态，Inspector 把 Object 属性系统变成可编辑 UI，EditorPlugin 把专用工具插入主屏、dock、导入导出、gizmo 和调试链。"
      }
    ]
  },
  {
    id: "platformdrivers",
    title: "Platform / OS / DisplayServer / Drivers / Thirdparty",
    aliases: ["Platform", "platform", "drivers", "thirdparty", "OS", "OS_Windows", "OS_LinuxBSD", "OS_Android", "OS_Web", "DisplayServer", "DisplayServerWindows", "DisplayServerX11", "DisplayServerWayland", "DisplayServerMacOS", "DisplayServerAndroid", "DisplayServerWeb", "AudioDriver", "AudioDriverManager", "FileAccess", "DirAccess", "FileAccessWindows", "FileAccessUnix", "WASAPI", "XAudio2", "PulseAudio", "ALSA", "CoreAudio", "Emscripten", "JNI", "InputEvent", "SConstruct", "SCsub", "modules", "builtin_*"],
    summary: "平台层把 Godot 的抽象接口落到真实系统：平台入口创建 OS 子类并驱动主循环，DisplayServer 适配窗口和输入，AudioDriver 连接设备，FileAccess/DirAccess 映射真实文件系统，drivers/modules 包装跨平台能力，thirdparty 保存外部库源码。",
    article: [
      {
        type: "lead",
        text: "读平台、驱动和第三方代码时，核心问题不是“Windows 文件在哪里”，而是“引擎抽象最终由谁实现”。Godot 上层通常只认识 `OS`、`DisplayServer`、`AudioDriver`、`FileAccess`、`DirAccess`、`InputEvent` 这些抽象；真正的 Win32、X11、Wayland、Cocoa、Android JNI、Web/Emscripten、WASAPI、PulseAudio、libpng、freetype 等细节，都在 platform、drivers、modules、thirdparty 的边界里接上。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Godot 想成一套通用插座。游戏代码只说“我要开窗口、读文件、播放声音、处理输入”，不直接说“我要调用 Win32 API”或“我要调用浏览器 JavaScript”。平台层就是把这些通用请求翻译成各系统能听懂的操作。"
      },
      {
        type: "paragraph",
        text: "`platform` 像每个系统的转接头，`drivers` 像可复用的设备驱动和格式支持，`thirdparty` 像 Godot 随源码带上的外部库仓库，`modules` 则常把外部库包装成 Godot 功能。遇到问题时先看 Godot 的包装层，不要一开始就跳进 thirdparty 读几万行外部库源码。"
      },
      {
        type: "paragraph",
        text: "桌面平台通常有一个 Godot 自己控制的循环：处理窗口事件，再跑一帧 `Main::iteration()`。Android 和 Web 不一样，它们由系统 Activity 或浏览器 animation frame 回调 Godot，所以没有传统的阻塞式 while(true) 主循环。"
      },
      {
        type: "flow",
        title: "平台层的通用读法",
        steps: [
          { title: "先找抽象", text: "从 `OS`、`DisplayServer`、`AudioDriver`、`FileAccess`、`DirAccess`、`InputEvent` 入手。" },
          { title: "再找注册", text: "看平台初始化时把哪个实现注册为默认值或 create function。" },
          { title: "确认构建开关", text: "检查 `target`、`platform`、driver 开关、module 开关和 `builtin_*` 选项。" },
          { title: "跟到平台实现", text: "Windows、LinuxBSD、macOS、Android、Web 各自处理入口、事件循环和系统 API。" },
          { title: "回到上层边界", text: "平台层把事件、音频、文件、窗口能力交回 core/servers/scene；不要让平台实现侵入场景逻辑。" }
        ]
      },
      {
        type: "heading",
        title: "核心抽象分工"
      },
      {
        type: "table",
        title: "从 Godot 抽象到真实系统",
        headers: ["抽象/目录", "源码入口", "负责什么", "真实系统例子"],
        rows: [
          ["OS", "`core/os/os.h:46`", "进程级能力、命令行、路径、时间、熵、渲染驱动选择、Main 初始化钩子。", "`OS_Windows`、`OS_LinuxBSD`、`OS_Android`、`OS_Web`。"],
          ["DisplayServer", "`servers/display/display_server.h:62`", "窗口、屏幕、输入回调、剪贴板、surface、事件抽取、present 边界。", "Win32、X11、Wayland、Cocoa、Android Surface、Web canvas。"],
          ["AudioDriver", "`servers/audio/audio_server.h:48`、`:159`", "设备线程、输出/输入缓冲、采样率、声道、延迟，向 AudioServer 请求混音。", "WASAPI、XAudio2、PulseAudio、ALSA、CoreAudio、OpenSL、Web AudioWorklet。"],
          ["FileAccess / DirAccess", "`core/io/file_access.h`、`core/io/dir_access.h`", "`res://`、`user://`、普通文件系统、pipe 的默认实现。", "Windows 文件 API、Unix 文件 API、Android APK/Java handler、Web IndexedDB。"],
          ["platform", "`platform/<platform>`", "目标系统专属入口、生命周期、窗口/输入/文件/音频/动态库适配。", "Windows、LinuxBSD、macOS、Android、Web、iOS、visionOS。"],
          ["drivers", "`drivers/SCsub:11`", "跨平台或半跨平台驱动、图形/音频/格式支持。", "Vulkan、D3D12、OpenGL/GLES、PNG、SDL、PulseAudio、WASAPI。"],
          ["thirdparty", "`SConstruct:318`", "vendored 外部库源码和内置库选项。", "freetype、harfbuzz、libpng、mbedtls、glslang、spirv-cross、zstd。"],
          ["modules", "`modules/SCsub:18`", "可选功能模块，常包装 thirdparty 或注册完整子系统。", "gdscript、mono、gltf、openxr、websocket、mbedtls、freetype。"]
        ]
      },
      {
        type: "heading",
        title: "桌面启动链"
      },
      {
        type: "paragraph",
        text: "Windows 是最直观样本：`platform/windows/godot_windows.cpp:68` 的 `widechar_main()` 创建 `OS_Windows`，`83` 调 `Main::setup()`，`97` 调 `Main::start()`，成功后 `98` 调 `os.run()`，最后 `102` 调 `Main::cleanup()`。LinuxBSD 的入口在 `platform/linuxbsd/godot_linuxbsd.cpp:70`，形状相同。"
      },
      {
        type: "code",
        language: "cpp",
        title: "桌面平台入口的简化形状",
        code: `int widechar_main(int argc, wchar_t **argv) {
    OS_Windows os(nullptr);

    Error err = Main::setup(exec_path, argc - 1, argv_utf8);
    if (err != OK) {
        return EXIT_FAILURE;
    }

    if (Main::start() == EXIT_SUCCESS) {
        os.run();
    }

    Main::cleanup();
    return os.get_exit_code();
}`
      },
      {
        type: "paragraph",
        text: "一个源码细节要注意：`core/os/os.h` 定义了 OS 抽象和 Main 会调用的初始化钩子，但没有统一的虚函数 `OS::run()`。`run()` 是桌面平台 OS 子类提供的惯用主循环方法；Android/Web 则由宿主平台每帧回调 Godot。读文档里“OS::run”时，要理解成“平台 OS 的主循环入口”，不要误以为 core/os/os.h 里有同名虚函数。"
      },
      {
        type: "flow",
        title: "Windows / LinuxBSD 桌面主循环",
        steps: [
          { title: "平台 main", text: "创建平台 OS 子类，把命令行整理成 Godot 格式。" },
          { title: "Main::setup", text: "初始化 core、项目设置、Server、资源系统、命令行等。" },
          { title: "Main::start", text: "选择 editor、project manager、导出命令或游戏主场景。" },
          { title: "main_loop->initialize", text: "`OS_Windows::run()` 在 `os_windows.cpp:2348`，LinuxBSD 在 `os_linuxbsd.cpp:990`。" },
          { title: "process_events", text: "每轮先 `DisplayServer::get_singleton()->process_events()`。" },
          { title: "Main::iteration", text: "再跑一帧引擎主循环；返回 true 时退出。" },
          { title: "finalize/cleanup", text: "main loop finalize，平台入口再调用 `Main::cleanup()`。" }
        ]
      },
      {
        type: "table",
        title: "桌面平台关键锚点",
        headers: ["平台", "入口", "文件系统默认值", "音频/显示注册", "主循环"],
        rows: [
          ["Windows", "`godot_windows.cpp:68`", "`OS_Windows::initialize()` 在 `os_windows.cpp:268`，`281` 到 `287` 注册 `FileAccessWindows`/`DirAccessWindows`。", "`os_windows.cpp:2907` 注册 WASAPI，`2910` 注册 XAudio2，`2913` 注册 `DisplayServerWindows`。", "`OS_Windows::run()` 在 `os_windows.cpp:2343`。"],
          ["LinuxBSD", "`godot_linuxbsd.cpp:70`", "Unix 默认值在 `drivers/unix/os_unix.cpp:171` 到 `177` 注册 `FileAccessUnix`/`DirAccessUnix`。", "`os_linuxbsd.cpp:1299` 注册 PulseAudio，`1303` 注册 ALSA，X11/Wayland 由平台显示代码注册。", "`OS_LinuxBSD::run()` 在 `os_linuxbsd.cpp:985`。"],
          ["macOS", "`godot_main_macos.mm:41`", "Objective-C++ 平台层处理 bundle、路径、菜单、窗口和系统集成。", "CoreAudio 由 `drivers/coreaudio`，DisplayServerMacOS 在 `display_server_macos.mm:3523` 注册。", "Cocoa/NSApplication 与 Godot 主循环协作。"]
        ]
      },
      {
        type: "heading",
        title: "Android 和 Web"
      },
      {
        type: "table",
        title: "宿主平台驱动的主循环",
        headers: ["平台", "源码入口", "主循环模式", "读源码重点"],
        rows: [
          ["Android setup", "`platform/android/java_godot_lib_jni.cpp:192`", "Java/Kotlin 层通过 JNI 调 `GodotLib_setup()`，内部 `217` 调 `Main::setup()`。", "Activity、Surface、权限、传感器和输入从 Java/JNI 进入。"],
          ["Android step", "`java_godot_lib_jni.cpp:284`", "`GodotLib_step()` 分阶段调用 `Main::setup2()`、`Main::start()`，然后每帧 `os_android->main_loop_iterate()`。", "看 `STEP_SETUP`、`STEP_SHOW_LOGO`、`STEP_STARTED` 状态，而不是找桌面 while 循环。"],
          ["Web start", "`platform/web/web_main.cpp:165`、`:175`", "Web 调 `Main::start()` 后用 `emscripten_set_main_loop()` 注册浏览器回调。", "浏览器 animation frame 调回 Godot，不能阻塞主线程。"],
          ["Web iteration", "`platform/web/os_web.cpp:87`", "`OS_Web` 每帧先 `DisplayServer::process_events()`，再 `Main::iteration()`。", "输入、剪贴板、fullscreen、gamepad、drop files 要看 C++ 和 JS glue。"]
        ]
      },
      {
        type: "heading",
        title: "DisplayServer"
      },
      {
        type: "paragraph",
        text: "`DisplayServer` 是窗口和输入的集中边界。抽象类入口在 `servers/display/display_server.h:62`，`process_events()` 是 `:119`，`swap_buffers()` 是 `:123`，`window_set_input_event_callback()` 是 `:394`。平台实现把系统事件转成 `InputEvent`，再通过 `Input::parse_input_event()` 和 Window/Viewport 的回调链进入场景层。"
      },
      {
        type: "flow",
        title: "DisplayServer 注册和创建",
        steps: [
          { title: "平台注册", text: "例如 `DisplayServerWindows::register_windows_driver()` 在 `display_server_windows.cpp:8278`。" },
          { title: "写入注册表", text: "`DisplayServer::register_create_function()` 在 `display_server.cpp:2027` 保存 name、create function、rendering driver getter。" },
          { title: "Main 选择", text: "`main/main.cpp:3343` 用当前 display driver index 和 rendering driver 调 `DisplayServer::create()`。" },
          { title: "失败 fallback", text: "`main/main.cpp:3344` 到 `3358` 会尝试其他 display server，跳过 headless 默认末尾项。" },
          { title: "事件抽取", text: "平台 `process_events()` 把系统事件变成 Godot 输入事件。" },
          { title: "上层分发", text: "Input、Window、Viewport 再决定 GUI、shortcut、unhandled input、脚本回调。" }
        ]
      },
      {
        type: "table",
        title: "DisplayServer 平台实现",
        headers: ["实现", "注册入口", "事件入口", "输入转换例子"],
        rows: [
          ["Windows", "`platform/windows/display_server_windows.cpp:8278`", "`process_events()` 在 `:4224`", "鼠标在 `:4979`，键盘在 `:6865` 调 `Input::parse_input_event()`。"],
          ["X11", "`platform/linuxbsd/x11/display_server_x11.cpp:7588`", "`process_events()` 在 `:4886`", "键盘、鼠标、触摸事件在 X11 分支中转为 InputEvent。"],
          ["Wayland", "`platform/linuxbsd/wayland/display_server_wayland.cpp:2603`", "`process_events()` 在 `:1892`", "`:2039`、`:2077` 可见输入事件进入 Input。"],
          ["macOS", "`platform/macos/display_server_macos.mm:3523`", "`process_events()` 在 `:3180`", "Objective-C++ 事件处理在 `:506`、`:659` 等位置调 Input。"],
          ["Android", "`platform/android/display_server_android.cpp:681`", "`process_events()` 在 `:647`", "Java/JNI 输入和传感器最终进入 DisplayServerAndroid/Input。"],
          ["Web", "`platform/web/display_server_web.cpp:1238`", "`process_events()` 在 `:1469`", "JS 回调在 `:304`、`:350`、`:1497` 等位置转成 InputEvent。"]
        ]
      },
      {
        type: "code",
        language: "cpp",
        title: "DisplayServer 注册表的核心形状",
        code: `void DisplayServer::register_create_function(const char *p_name, CreateFunction p_function, GetRenderingDriversFunction p_get_drivers) {
    server_create_functions[server_create_count - 1].name = p_name;
    server_create_functions[server_create_count - 1].create_function = p_function;
    server_create_functions[server_create_count - 1].get_rendering_drivers_function = p_get_drivers;
    server_create_count++;
}

DisplayServer *DisplayServer::create(int p_index, const String &p_rendering_driver, ..., Error &r_error) {
    return server_create_functions[p_index].create_function(p_rendering_driver, ..., r_error);
}`
      },
      {
        type: "heading",
        title: "AudioDriver 和文件系统"
      },
      {
        type: "paragraph",
        text: "`AudioServer` 负责混音，`AudioDriver` 负责设备。`AudioDriverManager` 的驱动数组在 `servers/audio/audio_server.cpp:210`，默认有 dummy driver；`add_driver()` 在 `:215` 会把新驱动插到 dummy 前面，保证 dummy 永远是 fallback；`initialize()` 在 `:227` 先试用户指定驱动，再按顺序试所有驱动，全部失败时用 dummy 并警告。"
      },
      {
        type: "flow",
        title: "声音从 Godot 到设备",
        steps: [
          { title: "平台添加驱动", text: "Windows 添加 WASAPI/XAudio2；LinuxBSD 添加 PulseAudio/ALSA；macOS/iOS 添加 CoreAudio；Web 添加 AudioWorklet/ScriptProcessor。" },
          { title: "AudioDriverManager", text: "初始化时选择可用驱动，失败落到 Dummy。" },
          { title: "设备线程/回调", text: "驱动按设备缓冲请求采样。" },
          { title: "AudioServer 混音", text: "`AudioDriver::audio_server_process()` 调 `AudioServer::_driver_process()`。" },
          { title: "提交缓冲", text: "驱动把混音结果写入 WASAPI、PulseAudio、CoreAudio、Web Audio 等后端。" }
        ]
      },
      {
        type: "table",
        title: "文件系统默认实现",
        headers: ["平台/层", "源码锚点", "映射含义"],
        rows: [
          ["Windows", "`platform/windows/os_windows.cpp:281` 到 `287`", "`ACCESS_RESOURCES`、`ACCESS_USERDATA`、`ACCESS_FILESYSTEM` 用 `FileAccessWindows`/`DirAccessWindows`，pipe 用 `FileAccessWindowsPipe`。"],
          ["Unix/LinuxBSD", "`drivers/unix/os_unix.cpp:171` 到 `177`", "多个 Unix-like 平台共用 `FileAccessUnix`/`DirAccessUnix`，pipe 用 `FileAccessUnixPipe`。"],
          ["Android", "`platform/android/os_android.cpp`", "资源路径、APK、Java file access、用户目录和外部存储权限会影响 `res://`/`user://`。"],
          ["Web", "`platform/web`", "虚拟文件系统、preload zip、IndexedDB 和浏览器安全模型影响文件访问。"],
          ["导出包", "ResourceLoader / FileAccess packed", "导出后 `res://` 可能来自 pack/pck，不再是普通磁盘目录。"]
        ]
      },
      {
        type: "heading",
        title: "drivers、modules、thirdparty"
      },
      {
        type: "paragraph",
        text: "`drivers/SCsub` 决定哪些共享驱动进入构建：`11` 到 `13` 包含 Unix/Windows 基础驱动，`16` 到 `26` 处理 ALSA/PulseAudio/WASAPI/XAudio2，`46` 到 `61` 按 `vulkan`、`d3d12`、`opengl3`、`metal` 开关加入图形驱动，`69` 加入 PNG。`drivers/register_driver_types.cpp:44` 则把 PNG loader/saver 等核心驱动类型注册进资源系统。"
      },
      {
        type: "paragraph",
        text: "`modules/SCsub:18` 生成 `modules_enabled.gen.h`，`:24` 生成 `register_module_types.gen.cpp`，`:33` 遍历 `env.module_list` 构建每个启用模块。`SConstruct:485` 生成 `module_<name>_enabled` 构建选项；`SConstruct:318` 到 `348` 定义大量 `builtin_*` thirdparty 选项；`SConstruct:1243` 只构建当前选中的 `platform/<platform>/SCsub`。"
      },
      {
        type: "table",
        title: "三层不要混读",
        headers: ["层", "你应该先看", "什么时候深入下一层"],
        rows: [
          ["platform", "目标平台入口、OS 子类、DisplayServer、平台文件/动态库/生命周期。", "只有当抽象调用已经确认落到平台系统 API 时。"],
          ["drivers", "Godot 的驱动包装层和注册代码。", "需要确认某个驱动是否编译进当前二进制、是否注册、是否初始化失败。"],
          ["modules", "模块注册、资源 loader/saver、ClassDB 注册、Server/Resource 包装。", "功能由模块提供，例如 glTF、mbedtls、freetype、websocket、openxr。"],
          ["thirdparty", "尽量先看 Godot 包装层。", "确认问题来自外部库行为、升级补丁、平台兼容或许可证时再深入。"],
          ["SConstruct/SCsub", "构建选项、平台选择、driver/module/thirdparty 开关。", "源码存在但运行时找不到，或某类代码未参与当前构建时。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "table",
        title: "常见问题应该沿哪条链查",
        headers: ["问题", "排查链路", "常见误判"],
        rows: [
          ["窗口没创建或显示后端 fallback", "平台注册 DisplayServer -> `DisplayServer::create()` -> `main/main.cpp:3343` fallback -> 渲染 driver 可用性。", "只看 RenderingServer，而忽略 display driver 注册和平台窗口创建失败。"],
          ["输入事件进不了脚本", "平台事件 -> DisplayServer::process_events -> `Input::parse_input_event()` -> Window/Viewport -> `_input`/GUI/unhandled。", "以为平台层直接调用脚本。"],
          ["`res://` 或 `user://` 在某平台路径不对", "FileAccess/DirAccess 默认实现 -> pack/APK/IndexedDB/用户目录 -> ProjectSettings 路径本地化。", "把导出包里的 `res://` 当成普通文件夹。"],
          ["音频无输出", "平台是否添加 AudioDriver -> AudioDriverManager 初始化 -> 设备线程/回调 -> AudioServer 混音 -> dummy fallback。", "只看 AudioStreamPlayer 节点。"],
          ["Web 版本卡住或没进帧", "Emscripten main loop -> `OS_Web` iteration -> browser event/canvas state。", "在 Web 平台寻找桌面式 `while(true)`。"],
          ["第三方库看起来有 bug", "先看 Godot wrapper、模块选项、资源导入/转换参数，再确认 thirdparty 调用。", "直接改 thirdparty，导致升级冲突或误改外部库默认行为。"]
        ]
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "边界判断",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["OS", "进程级平台能力和 Main 初始化协作。", "不分发场景树输入，也不直接调用用户脚本。"],
          ["DisplayServer", "窗口、屏幕、输入事件抽取、系统窗口能力。", "不决定 Node 生命周期，不处理游戏逻辑。"],
          ["Input", "统一输入状态和事件解析入口。", "不负责 Win32/X11/Cocoa/JS 原始事件采集。"],
          ["AudioServer", "混音、bus、effect、播放状态。", "不直接打开 WASAPI/PulseAudio/CoreAudio 设备。"],
          ["AudioDriver", "设备缓冲和平台音频 API。", "不理解场景里的 AudioStreamPlayer 节点树。"],
          ["drivers", "Godot 维护的驱动/格式适配。", "不是所有外部库源码。"],
          ["thirdparty", "外部库源码副本。", "不是 Godot 的业务语义层。"],
          ["modules", "功能模块注册和封装。", "不是目标平台入口。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：平台层可以直接处理场景逻辑。健康边界是平台产出窗口、输入、文件、音频、时间等能力，上层再分发到场景。",
          "误区二：源码存在就一定参与当前构建。必须看 `platform`、`target`、driver 开关、module 开关和 `builtin_*` 选项。",
          "误区三：DisplayServer 等于渲染后端。DisplayServer 管窗口和事件；RenderingDevice/RenderingServer 管渲染资源和绘制命令。",
          "误区四：桌面、Android、Web 都有同一种主循环。Android/Web 由宿主平台驱动每帧回调。",
          "误区五：thirdparty 代码就是 Godot 行为。Godot 通常通过 modules/drivers 包装外部库，用户可见语义多在包装层。",
          "误区六：音频问题只看 AudioServer。设备初始化、driver 选择、dummy fallback、线程回调都可能是原因。",
          "误区七：`res://` 永远对应磁盘文件夹。导出后可能是 pack/APK/preload zip/虚拟文件系统。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/os/os.h:46`，理解 OS 抽象保存哪些进程级状态和 Main 初始化钩子。",
          "读 Windows 或 LinuxBSD 的平台入口，确认 `Main::setup()`、`Main::start()`、平台 `run()`、`Main::cleanup()` 的顺序。",
          "读 `OS_Windows::initialize()` 或 `drivers/unix/os_unix.cpp`，看 FileAccess/DirAccess 默认实现如何注册。",
          "读 `servers/display/display_server.h` 和 `display_server.cpp:2027`、`:2051`，理解 DisplayServer 注册表和创建流程。",
          "选一个平台 DisplayServer，看 `process_events()` 如何把原生事件转成 `InputEvent` 并交给 Input。",
          "读 `servers/audio/audio_server.h/cpp` 的 AudioDriverManager，再看一个平台怎样 `add_driver()`。",
          "读 `drivers/SCsub`、`modules/SCsub`、`SConstruct:318` 到 `348`、`SConstruct:1243`，把构建开关和源码目录对应起来。",
          "最后再进入 thirdparty，只验证外部库调用、升级补丁或平台兼容问题，不要用 thirdparty 反推 Godot 用户 API。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "平台层的本质是把 Godot 的统一抽象接到真实系统：平台入口和 OS 驱动生命周期，DisplayServer 接窗口和输入，AudioDriver 接设备，FileAccess/DirAccess 接文件系统，drivers/modules 包装功能，thirdparty 只在更底层提供外部库能力。"
      }
    ]
  },
  {
    id: "moduleatlas",
    title: "Modules / ModuleInitializationLevel / register_types",
    aliases: ["modules", "module atlas", "ModuleInitializationLevel", "initialize_modules", "uninitialize_modules", "register_module_types.gen.cpp", "modules_enabled.gen.h", "modules_builders.py", "register_types.cpp", "register_types.h", "config.py", "SCsub", "module_*_enabled", "env.module_list", "can_build", "GDREGISTER_CLASS", "GDScript", "GLTFDocument", "OpenXRInterface", "PhysicsServer3DManager", "Jolt Physics", "GodotPhysics3D", "EditorPlugins", "ScriptServer", "ResourceLoader", "ResourceSaver"],
    summary: "modules 是 Godot 可裁剪能力的集中接入区：构建系统先按 config.py、SCsub、module_*_enabled 选择模块并生成总入口，运行时再按 CORE、SERVERS、SCENE、EDITOR 初始化级别调用每个模块的 register_types，让资源格式、脚本语言、Server 后端、节点、编辑器插件等能力出现。",
    article: [
      {
        type: "lead",
        text: "模块不是一个固定层级，而是一套“构建时选择、运行时按级别接入”的机制。同一个模块可以在 CORE 注册底层库，在 SERVERS 注册后端，在 SCENE 注册类和资源，在 EDITOR 注册导入器或插件。读模块时不要先扎进实现文件，先问：它是否参与当前构建？它在哪个初始化级别注册？它注册的是类、资源 loader、Server 后端、脚本语言，还是编辑器工具？"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 modules 想成 Godot 的“可选功能包”。比如 GDScript、glTF、OpenXR、Jolt Physics、WebSocket、mbedTLS、freetype、各种图片和音频格式，都不是随便散落在引擎里的，而是通过模块目录把源码、构建脚本、注册入口和文档类集中起来。"
      },
      {
        type: "paragraph",
        text: "一个模块要先被构建系统选中，才会编译进二进制；编译进去以后，还要在正确的初始化阶段把自己的功能注册给引擎。没有注册，类不会进入 ClassDB，资源 loader 不会被 ResourceLoader 看见，物理后端不会出现在 PhysicsServer manager 里，编辑器导入器也不会出现。"
      },
      {
        type: "paragraph",
        text: "当前源码树里 `modules` 除 `__pycache__` 外共有 57 个模块。主文档里的模块全表是定位地图：你用它判断某个格式、脚本语言、物理后端、文本后端、网络能力或 XR 能力在哪个模块，再回到该模块的 `config.py`、`SCsub`、`register_types.*` 查真实入口。"
      },
      {
        type: "flow",
        title: "一个模块从源码到可用能力",
        steps: [
          { title: "config.py", text: "声明能否构建、默认是否启用、平台条件、文档类和依赖。" },
          { title: "SConstruct", text: "生成 `module_<name>_enabled` 构建选项，并建立 `env.module_list`。" },
          { title: "modules/SCsub", text: "生成 `modules_enabled.gen.h` 和 `register_module_types.gen.cpp`，并编译每个启用模块的源码。" },
          { title: "register_types", text: "每个模块实现 `initialize_<name>_module(p_level)` 和 `uninitialize_<name>_module(p_level)`。" },
          { title: "Main 初始化", text: "`Main` 按 CORE、SERVERS、SCENE、EDITOR 调 `initialize_modules()`。" },
          { title: "能力出现", text: "类、资源格式、脚本语言、Server 后端、导入器、编辑器插件被注册到对应系统。" }
        ]
      },
      {
        type: "heading",
        title: "四个入口文件"
      },
      {
        type: "table",
        title: "读模块先看这里",
        headers: ["文件", "负责什么", "判断问题"],
        rows: [
          ["`modules/<name>/config.py`", "构建条件、默认启用状态、平台支持、模块依赖、文档类路径。", "为什么这个模块没编进当前二进制？为什么文档类缺失？"],
          ["`modules/<name>/SCsub`", "把模块 cpp、第三方源码、include path、生成文件加入 SCons。", "某个源码文件到底有没有参与编译？thirdparty 是从哪里接进来的？"],
          ["`modules/<name>/register_types.h/cpp`", "运行时注册入口，按 `ModuleInitializationLevel` 接入引擎。", "功能在哪个阶段出现？注册了类、loader、后端、插件还是 singleton？"],
          ["模块内部实现", "parser、loader、server backend、resource、editor plugin 等真实逻辑。", "只有确认注册路径后再读，否则容易在细节里迷路。"]
        ]
      },
      {
        type: "table",
        title: "config.py 常见信息",
        headers: ["函数/字段", "例子", "含义"],
        rows: [
          ["`can_build(env, platform)`", "`modules/openxr/config.py:1` 按平台和 `disable_xr` 判断。", "模块是否能在目标平台构建。"],
          ["`is_enabled()`", "`modules/mono/config.py:31` 默认返回 false。", "默认是否启用，用户可用 `module_mono_enabled=yes` 打开。"],
          ["`env.module_add_dependencies()`", "`modules/mono/config.py:2` 在 editor build 下依赖 regex。", "声明模块之间的构建依赖。"],
          ["`get_doc_classes()`", "`modules/openxr/config.py:13` 列 OpenXR 文档类。", "文档生成和 API 文档入口。"],
          ["`configure(env)`", "`modules/mono/config.py:8` 检查平台是否支持 mono。", "模块可修改构建环境或失败退出。"]
        ]
      },
      {
        type: "heading",
        title: "生成总入口"
      },
      {
        type: "paragraph",
        text: "`modules/SCsub:19` 生成 `modules_enabled.gen.h`，`:24` 生成 `register_module_types.gen.cpp`，`:33` 遍历 `env.module_list` 进入每个模块的 `SCsub`。所以不要手改生成文件；真正源头是 SCons 参数、模块 config、模块 SCsub 和 `modules/modules_builders.py`。"
      },
      {
        type: "paragraph",
        text: "`modules/modules_builders.py:8` 的 `modules_enabled_builder()` 会为每个启用模块写出 `#define MODULE_<NAME>_ENABLED`。`register_module_types_builder()` 在 `:15` 读取模块表，`:17` 生成所有 `register_types.h` include，`:18` 到 `:24` 生成 `initialize_<name>_module(p_level)` 调用，`:25` 到 `:31` 生成卸载调用，`:43` 和 `:47` 写出统一的 `initialize_modules()` / `uninitialize_modules()`。"
      },
      {
        type: "code",
        language: "cpp",
        title: "生成出来的总入口形状",
        code: `// modules/register_module_types.gen.cpp 的核心形状。
#include "modules/modules_enabled.gen.h"
#include "modules/gdscript/register_types.h"
#include "modules/gltf/register_types.h"

void initialize_modules(ModuleInitializationLevel p_level) {
#ifdef MODULE_GDSCRIPT_ENABLED
    initialize_gdscript_module(p_level);
#endif
#ifdef MODULE_GLTF_ENABLED
    initialize_gltf_module(p_level);
#endif
}

void uninitialize_modules(ModuleInitializationLevel p_level) {
#ifdef MODULE_GDSCRIPT_ENABLED
    uninitialize_gdscript_module(p_level);
#endif
}`
      },
      {
        type: "table",
        title: "生成文件和源码源头",
        headers: ["产物", "生成源头", "用途"],
        rows: [
          ["`modules_enabled.gen.h`", "`modules/SCsub:19` + `modules_builders.py:8`", "把启用模块变成 `MODULE_*_ENABLED` 宏。"],
          ["`register_module_types.gen.cpp`", "`modules/SCsub:24` + `modules_builders.py:15`", "统一 include 和调用每个模块的 initialize/uninitialize。"],
          ["`register_module_types.h`", "源码文件，`modules/register_module_types.h:35`", "定义 `ModuleInitializationLevel` 和统一函数声明。"],
          ["模块静态库", "`modules/SCsub:33` 到 `42`", "把每个有源码的模块编成 `module_<name>` 并加入链接。"],
          ["模块测试 include", "`modules/SCsub:44` 以后", "测试构建时收集模块 tests 目录。"]
        ]
      },
      {
        type: "heading",
        title: "初始化级别"
      },
      {
        type: "paragraph",
        text: "`ModuleInitializationLevel` 在 `modules/register_module_types.h:35`，它直接映射到 GDExtension 的 CORE、SERVERS、SCENE、EDITOR 四个级别。`Main` 会在初始化过程中多次调用 `initialize_modules(p_level)`：当前源码在 `main/main.cpp:787`、`:797`、`:830`、`:837` 的早期路径，以及 `:3201`、`:3767`、`:3800` 的正常启动路径都能看到对应调用。清理时反向调用 `uninitialize_modules()`，锚点包括 `main/main.cpp:5261`、`:5269`、`:5289`、`:5365`。"
      },
      {
        type: "table",
        title: "四个初始化级别",
        headers: ["级别", "适合注册什么", "不该依赖什么", "例子"],
        rows: [
          ["CORE", "底层格式、加密、压缩、shader 编译、基础资源支持。", "不要依赖 SceneTree、Node、编辑器 UI。", "`mbedtls` 在 CORE，`glslang` 在 CORE。"],
          ["SERVERS", "脚本语言、物理/文本/导航/XR 后端、Server manager。", "不要依赖场景节点已注册。", "`gdscript` 在 SERVERS 注册 ScriptLanguage；`jolt_physics` 注册 PhysicsServer3D 后端。"],
          ["SCENE", "Resource、Node、运行时 API、资源 loader/saver、类绑定。", "不要依赖 editor-only 类型。", "`gltf` 在 SCENE 注册 `GLTFDocument` 等运行时类；`mono` 注册 C# 脚本资源。"],
          ["EDITOR", "导入器、导出器、Inspector 插件、主屏/dock、语言服务等工具能力。", "导出模板通常没有这层。", "`gltf` 在 EDITOR 注册 `EditorSceneFormatImporterGLTF`；GDScript 在 EDITOR 注册 LSP/语法高亮。"]
        ]
      },
      {
        type: "flow",
        title: "Main 和模块初始化顺序",
        steps: [
          { title: "CORE", text: "core 类型和早期 singleton 准备后，模块可注册底层能力。" },
          { title: "SERVERS", text: "Server 层可用，模块接入脚本语言、物理、文本、导航、XR 等后端。" },
          { title: "SCENE", text: "scene 类型注册后，模块可暴露 Node、Resource 和运行时 API。" },
          { title: "EDITOR", text: "`TOOLS_ENABLED` 下进入编辑器层，模块注册导入导出、插件、语言服务和工具 UI。" },
          { title: "cleanup", text: "退出时反向卸载：EDITOR -> SCENE -> SERVERS -> CORE。" }
        ]
      },
      {
        type: "heading",
        title: "模块注册案例"
      },
      {
        type: "table",
        title: "同一套机制接入不同能力",
        headers: ["模块", "源码入口", "初始化级别", "注册结果"],
        rows: [
          ["gdscript", "`modules/gdscript/register_types.cpp:137`", "SERVERS + EDITOR", "SERVERS 注册 `GDScript`、`GDScriptLanguage`、`.gd` loader/saver 和 cache；EDITOR 注册语法高亮、LSP、语言服务插件。"],
          ["gltf", "`modules/gltf/register_types.cpp:111`", "SCENE + EDITOR", "SCENE 注册 `GLTFDocument`、GLTF 资源类和扩展；EDITOR 注册 `EditorSceneFormatImporterGLTF`、导出插件和 Blender 导入设置。"],
          ["openxr", "`modules/openxr/register_types.cpp:134`", "CORE + SERVERS + SCENE + EDITOR", "CORE 注册 extension wrapper 类；SERVERS 注册 OpenXR API wrapper；SCENE 注册 `OpenXRInterface` 并加入 XRServer；EDITOR 注册交互配置编辑器。"],
          ["godot_physics_3d", "`modules/godot_physics_3d/register_types.cpp:52`", "SERVERS", "注册 `GodotPhysics3D`，并设置为 PhysicsServer3D 默认后端。"],
          ["jolt_physics", "`modules/jolt_physics/register_types.cpp:53`", "SERVERS", "初始化 Jolt，向 PhysicsServer3DManager 注册 `Jolt Physics` 后端和项目设置。"],
          ["mono", "`modules/mono/register_types.cpp:46`", "SCENE", "注册 C# 脚本语言和 C# Script 资源，config.py 默认不启用。"]
        ]
      },
      {
        type: "code",
        language: "cpp",
        title: "典型 register_types 写法",
        code: `void initialize_example_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
        // 适合注册 Server 后端、脚本语言、底层服务。
    }

    if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        GDREGISTER_CLASS(MyResource);
        ResourceLoader::add_resource_format_loader(my_loader);
    }

#ifdef TOOLS_ENABLED
    if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
        EditorPlugins::add_by_type<MyEditorPlugin>();
    }
#endif
}`
      },
      {
        type: "heading",
        title: "模块分类地图"
      },
      {
        type: "table",
        title: "57 个模块可以先按用途分组",
        headers: ["类别", "常见模块", "先看什么"],
        rows: [
          ["脚本/语言", "`gdscript`、`mono`", "ScriptLanguage、Script、ScriptInstance、ResourceLoader、编辑器语言服务。"],
          ["物理/导航/XR", "`godot_physics_2d`、`godot_physics_3d`、`jolt_physics`、`navigation_2d`、`navigation_3d`、`openxr`、`webxr`", "Server manager、后端注册、接口资源、平台能力。"],
          ["导入/资源格式", "`gltf`、`fbx`、`bmp`、`jpg`、`webp`、`svg`、`dds`、`ktx`、`tinyexr`、`hdr`、`tga`", "ResourceFormatLoader/Saver、Importer、Editor import 插件。"],
          ["音频/视频/容器", "`ogg`、`vorbis`、`mp3`、`theora`、`interactive_music`", "AudioStream/VideoStream 资源、解码器、导入器。"],
          ["文本/字体/图形工具链", "`freetype`、`msdfgen`、`text_server_adv`、`text_server_fb`、`glslang`、`basis_universal`、`astcenc`、`xatlas_unwrap`", "TextServer、shader 编译、纹理压缩、UV/mesh 工具。"],
          ["网络/安全", "`mbedtls`、`enet`、`websocket`、`webrtc`、`upnp`、`jsonrpc`", "协议资源、PacketPeer/Multiplayer、TLS/Crypto 封装。"]
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "table",
        title: "用模块链排查问题",
        headers: ["问题", "排查路径", "常见误判"],
        rows: [
          ["某个类在文档里有但当前二进制找不到", "`config.py` -> `module_*_enabled` -> `modules_enabled.gen.h` -> `register_types` 的初始化级别。", "只搜索类名，不确认模块是否启用。"],
          ["GLTF 导入器没出现", "`gltf/config.py` -> `gltf/SCsub` -> `initialize_gltf_module(EDITOR)` -> `EditorSceneFormatImporterGLTF`。", "只看 `GLTFDocument` 运行时类，忽略 EDITOR 级别导入器。"],
          ["Jolt 后端无法选择", "`module_jolt_physics_enabled` -> `initialize_jolt_physics_module(SERVERS)` -> `PhysicsServer3DManager::register_server()`。", "去改 Node3D 或 PhysicsBody3D API。"],
          ["C# 模块没有编译", "`modules/mono/config.py:is_enabled()` 默认 false，`configure()` 还检查平台 supported。", "以为 `modules/mono` 目录存在就代表已启用。"],
          ["新增模块编译了但运行时没效果", "确认 `register_types.h` 被生成总入口 include，`initialize_<name>_module()` 在正确 level 执行，注册对象不是局部临时变量。", "只把 cpp 加进 SCsub，却没有注册运行时入口。"],
          ["导出模板里编辑器插件不存在", "确认代码是否包在 `TOOLS_ENABLED` 和 EDITOR level 下。", "把 editor-only 模块能力当成运行时功能。"]
        ]
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        title: "modules 和其他层的边界",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["modules", "可选功能接入、构建开关、运行时注册入口。", "不是平台入口，也不是统一业务层。"],
          ["drivers", "Godot 维护的底层驱动和共享格式支持。", "不负责模块级 API 暴露。"],
          ["thirdparty", "外部库源码副本。", "不直接定义 Godot 用户 API。"],
          ["ClassDB", "保存已注册类、方法、属性、信号。", "不决定模块是否编译。"],
          ["ResourceLoader", "加载已注册的资源格式。", "不自动发现未注册模块 loader。"],
          ["Server manager", "保存可选后端，如 PhysicsServer3DManager。", "不负责模块构建开关。"],
          ["GDExtension", "外部动态库按同样初始化级别接入。", "不是内置 modules 的构建系统。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：modules 是一个运行时层级。实际它是可选功能接入区，能跨 CORE、SERVERS、SCENE、EDITOR。",
          "误区二：目录存在就代表模块启用。必须看构建选项、config.py、modules_enabled.gen.h 和 register_module_types.gen.cpp。",
          "误区三：所有模块都在 SCENE 注册。脚本语言、物理后端、文本后端常在 SERVERS；导入器和语言服务常在 EDITOR。",
          "误区四：手改生成文件能解决问题。生成文件来自 modules/SCsub 和 modules_builders.py，下一次构建会重写。",
          "误区五：模块等于 thirdparty。模块可能包装 thirdparty，也可能完全是 Godot 自己的功能。",
          "误区六：编辑器插件可以在导出模板使用。EDITOR level 和 `TOOLS_ENABLED` 代码通常不进入普通运行时。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先用主文档模块表或 `Get-ChildItem modules -Directory` 找到目标模块目录。",
          "读 `modules/<name>/config.py`，确认平台支持、默认启用、依赖和文档类。",
          "读 `modules/<name>/SCsub`，确认源码、thirdparty、include path 和生成文件是否进入构建。",
          "读 `modules/SCsub` 和 `modules/modules_builders.py`，理解生成总入口的机制。",
          "读 `modules/register_module_types.h`，记住 CORE、SERVERS、SCENE、EDITOR 四个级别。",
          "读目标模块 `register_types.cpp`，按 `p_level` 分支列出它注册的所有类、loader、saver、server、插件。",
          "沿注册结果进入真实实现：ResourceLoader、ClassDB、ScriptServer、PhysicsServer manager、EditorPlugins 或对应 Server。",
          "最后回到构建参数和 `TOOLS_ENABLED`，确认当前二进制是否真的会包含这条路径。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "模块系统的主线是：构建时选择模块并生成总入口，运行时按 CORE/SERVERS/SCENE/EDITOR 调 register_types；功能是否存在，取决于模块是否启用、是否编译、是否在正确初始化级别完成注册。"
      }
    ]
  },
  {
    id: "projectsettings",
    title: "ProjectSettings / project.godot / autoload",
    aliases: ["ProjectSettings", "project.godot", "autoload", "main_scene", "GLOBAL_GET", "ProjectSettings::AutoloadInfo"],
    summary: "项目配置进入运行时的中间层：它把 project.godot、资源包、feature override、autoload 和主场景设置提供给 Main、InputMap、ResourceLoader 和脚本系统。",
    article: [
      {
        type: "lead",
        text: "ProjectSettings 是 Godot 把项目文件变成运行时决策的中心。主场景、autoload、输入映射、资源包、feature override、窗口和语言设置都可能先进入 ProjectSettings，再被 Main、InputMap、ResourceLoader、GDScript 或编辑器读取。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 ProjectSettings 想成项目启动前要看的总配置表。Godot 不是在源码里写死“加载哪个场景、有哪些全局单例、有哪些输入 action”，而是在启动时读取项目配置，再按这张表搭出游戏运行环境。"
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "类入口是 `core/config/project_settings.h:40`。它继承 Object，因此也能通过 ClassDB 暴露给脚本和编辑器。内部保存设置表、autoload 列表、资源路径、project data 目录和 feature override 相关状态。动态写入入口是 `core/config/project_settings.cpp:289` 的 `_set()`，这里会特别处理 `autoload/` 和 `autoload_prepend/`。"
      },
      {
        type: "paragraph",
        text: "普通游戏路径里，`Main::start()` 在 `main/main.cpp:4311` 读取 `application/run/main_scene`，决定没有命令行场景时要加载哪个主场景。autoload 则在 `main/main.cpp:4476` 以后从 `ProjectSettings::get_autoload_list()` 取出，先于主场景实例化并加入 SceneTree root。"
      },
      {
        type: "table",
        title: "ProjectSettings 影响哪些启动行为",
        headers: ["配置", "运行时影响", "源码入口"],
        rows: [
          ["`application/run/main_scene`", "决定普通游戏默认加载的主场景。", "`main/main.cpp:4311`"],
          ["`autoload/*`", "创建全局节点或脚本单例，并在主场景前加入 SceneTree。", "`main/main.cpp:4476`"],
          ["`input/*`", "生成 InputMap 的 action 和按键映射。", "`main/main.cpp:2334`"],
          ["资源包和路径", "影响 `res://`、UID cache、remap 和打包资源可见性。", "`project_settings.cpp:581`、`614`"],
          ["feature override", "按平台、构建特性或自定义 feature 选择不同设置值。", "`project_settings.cpp:392`、`415`"]
        ]
      },
      {
        type: "heading",
        title: "源码锚点"
      },
      {
        type: "list",
        items: [
          "`core/config/project_settings.h:40`：ProjectSettings 类定义。",
          "`core/config/project_settings.cpp:289`：动态设置入口，包含 autoload 特殊处理。",
          "`main/main.cpp:4311`：读取主场景设置。",
          "`main/main.cpp:4476`：读取并实例化 autoload 列表。"
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "code",
        code: [
          "# project.godot 中的典型片段",
          "application/run/main_scene=\"res://main.tscn\"",
          "",
          "[autoload]",
          "GameState=\"*res://game_state.gd\"",
          "",
          "[input]",
          "jump={",
          "\"deadzone\": 0.5,",
          "\"events\": [Object(InputEventKey,\"keycode\":32)]",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这段配置最终会影响三条路径：Main 读取主场景，Main 在主场景前实例化 GameState，InputMap 从项目设置加载 jump action。排查运行结果时要把这三条路径分开看。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["ProjectSettings", "保存和查询项目级设置、autoload、资源路径、feature override。", "不负责加载 PackedScene 的具体格式。"],
          ["ResourceLoader", "按路径和 loader 把资源变成 Resource。", "不决定项目默认主场景。"],
          ["SceneTree", "运行和调度节点树。", "不保存 project.godot 的原始配置表。"],
          ["EditorSettings", "保存编辑器个人偏好。", "不等同于项目运行时设置。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：主场景是 Main 写死的。实际来自命令行或 ProjectSettings。",
          "误区二：autoload 是脚本语言特性。实际 Main 会按项目设置把脚本或场景实例化成节点。",
          "误区三：InputMap 默认总一样。编辑器和游戏加载的输入映射路径不同。",
          "误区四：资源路径只由文件系统决定。资源包、UID、remap 和 ProjectSettings 都会影响路径解析。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `core/config/project_settings.h:40`，看它保存哪些项目级状态。",
          "读 `_set()` 对 `autoload/` 和普通设置的分支。",
          "到 `Main::setup()` 看 ProjectSettings 何时创建和 setup。",
          "到 `Main::start()` 看主场景、autoload、输入映射如何消费这些设置。",
          "最后按具体问题进入 ResourceUID、InputMap、GDScript analyzer 或编辑器 Project Settings UI。"
        ]
      }
    ]
  },
  {
    id: "viewportwindowworld",
    title: "Viewport / Window / World2D / World3D",
    aliases: ["Viewport", "SubViewport", "Godot Window", "World2D", "World3D", "RendererViewport", "scenario", "render target"],
    summary: "场景树和渲染/输入之间的关键枢纽：Viewport 管输入、GUI、canvas、world 和 render target；Window 把 Viewport 接到 DisplayServer；World2D/World3D 保存空间上下文。",
    article: [
      {
        type: "lead",
        text: "Viewport 是很多显示和输入问题的中间答案。Node 在 SceneTree 里运行，但输入如何分发、GUI 根在哪里、2D canvas 和 3D scenario 属于谁、画面最终渲染到哪个目标，都要看 Viewport、Window 和 World 这一层。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "如果 SceneTree 是整栋楼的管理表，Viewport 就像一个带摄像机和门禁的房间：输入先到这里，UI 在这里排队，2D/3D 世界从这里接入，最后画面也从这里输出到窗口或纹理。"
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`Viewport` 定义在 `scene/main/viewport.h:96`，继承 Node。它保存 GUI、canvas、world、render target、输入 picking、debug draw、MSAA、scaling、transparent background 等状态。`Window` 定义在 `scene/main/window.h:42`，继承 Viewport，并额外连接 DisplayServer 的窗口 ID、native window 状态和焦点/大小事件。"
      },
      {
        type: "paragraph",
        text: "`World2D` 在 `scene/resources/world_2d.h:43`，`World3D` 在 `scene/resources/3d/world_3d.h:46`。它们是 Resource，而不是 Node；负责保存对应空间里的 canvas、physics space、navigation map、3D scenario、environment 等。渲染后端按 Viewport 汇总绘制，入口可从 `servers/rendering/renderer_viewport.cpp:782` 的 `draw_viewports()` 开始读。"
      },
      {
        type: "table",
        title: "四个对象的职责边界",
        headers: ["对象", "负责什么", "不负责什么"],
        rows: [
          ["Viewport", "输入分发、GUI root、canvas/world 归属、render target。", "不直接实现平台窗口 API。"],
          ["Window", "把 Viewport 接到 DisplayServer 窗口。", "不替代 SceneTree 主循环。"],
          ["World2D", "2D canvas、physics space、navigation map。", "不是 Node，不参与树遍历。"],
          ["World3D", "3D scenario、physics space、environment、camera effects。", "不保存节点父子关系。"],
          ["RendererViewport", "后端按 Viewport 组织绘制和 render target。", "不保存用户节点语义。"]
        ]
      },
      {
        type: "heading",
        title: "源码锚点"
      },
      {
        type: "list",
        items: [
          "`scene/main/viewport.h:96`：Viewport 类入口。",
          "`scene/main/window.h:42`：Window 继承 Viewport。",
          "`scene/resources/world_2d.h:43`：World2D 资源入口。",
          "`scene/resources/3d/world_3d.h:46`：World3D 资源入口。",
          "`servers/rendering/renderer_viewport.cpp:782`：渲染后端绘制 Viewport 的入口。"
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "code",
        code: [
          "# GDScript：确认节点最终属于哪个 Viewport 和 World",
          "func _ready():",
          "    print(get_viewport())",
          "    print(get_window())",
          "    if self is Node3D:",
          "        print(get_world_3d())",
          "    if self is CanvasItem:",
          "        print(get_world_2d())"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个检查能快速区分“节点不在树里”“在树里但不在预期 Viewport”“3D 对象进了错误 World3D”“SubViewport 没有被显示或更新”等问题。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        headers: ["概念", "边界"],
        rows: [
          ["SceneTree", "调度整棵树和主场景；不等同于某个渲染目标。"],
          ["Node", "保存父子关系和生命周期；是否显示还要看 Viewport/World/Server 状态。"],
          ["CanvasItem / VisualInstance3D", "提交可见对象状态；真正归属通过 Viewport 和 World 确定。"],
          ["DisplayServer", "处理平台窗口和输入事件；不保存场景 GUI 树。"],
          ["RenderingServer", "保存渲染后端对象；按 Viewport 组织最终绘制。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：Node 在 SceneTree 里就一定会显示。还要看它所属 Viewport、World、可见性和 Server 状态。",
          "误区二：Window 只是平台窗口包装。它也是 Viewport，会参与场景树、输入和渲染目标逻辑。",
          "误区三：World3D 是节点。实际它是 Resource，用来保存 3D 空间上下文。",
          "误区四：SubViewport 会自动显示。它通常要被 ViewportTexture、SubViewportContainer 或自定义渲染路径消费。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `Viewport` 头文件，圈出 GUI、world、render target、input 相关字段。",
          "读 `Window`，看它如何把 Viewport 接到 DisplayServer window id。",
          "读 `World2D` / `World3D`，理解 canvas、physics space、scenario 的保存位置。",
          "沿 CanvasItem 或 VisualInstance3D 进入树的通知，看它们如何拿 Viewport/World 并提交 Server 状态。",
          "最后读 `RendererViewport::draw_viewports()`，确认后端按 Viewport 组织绘制。"
        ]
      }
    ]
  },
  {
    id: "resourceimportpipeline",
    title: "Resource Import Pipeline / ResourceUID / .import",
    aliases: ["Resource Import Pipeline", "ResourceImporter", "ResourceFormatImporter", "ResourceUID", ".import", "EditorFileSystem", "uid://", "imported resource"],
    summary: "编辑器资源管线：扫描源文件和 .import，生成内部资源，用 UID 维持稳定引用，再让 ResourceLoader 和导出流程消费导入产物。",
    article: [
      {
        type: "lead",
        text: "资源导入管线解释了为什么 Godot 项目里同一个素材会同时出现源文件、`.import` 元数据、`.godot/imported` 产物、UID 和导出包内文件。运行时加载的常常不是原始 png/fbx/wav，而是编辑器导入后的内部资源路径。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把导入系统想成厨房备菜。你放进项目的是原材料，编辑器扫描后会按导入设置切好、压缩、转换并登记编号；游戏运行时通常直接拿备好的成品，而不是每次从原材料重新加工。"
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`ResourceUID` 定义在 `core/io/resource_uid.h:41`，维护 UID 与资源路径的映射。`ResourceFormatImporter` 定义在 `core/io/resource_importer.h:41`，继承 ResourceFormatLoader，负责读取 `.import` 元数据并把原始路径导向导入后的内部资源。具体导入器继承 `ResourceImporter`，定义入口在 `core/io/resource_importer.h:110`。"
      },
      {
        type: "paragraph",
        text: "编辑器侧的项目扫描由 `EditorFileSystem` 管，类入口在 `editor/file_system/editor_file_system.h:145`。扫描线程入口在 `editor/file_system/editor_file_system.cpp:565`，重导入判断入口在 `editor/file_system/editor_file_system.cpp:570`。它会比较源文件时间、`.import` 内容、md5、导入产物和 importer 支持状态，决定是否重新导入。"
      },
      {
        type: "flow",
        title: "从源文件到运行时加载",
        steps: [
          { title: "源文件进入项目", text: "例如 `res://art/hero.png`。" },
          { title: "EditorFileSystem 扫描", text: "记录文件、UID、依赖和导入状态。" },
          { title: "ResourceImporter 转换", text: "按导入设置生成内部资源和 `.import` 元数据。" },
          { title: "ResourceLoader 加载", text: "ResourceFormatImporter 读取 `.import`，找到真正的导入产物。" },
          { title: "导出打包", text: "ExportPlatform 收集导入后运行所需资源，而不只是原始文件。" }
        ]
      },
      {
        type: "heading",
        title: "源码锚点"
      },
      {
        type: "list",
        items: [
          "`core/io/resource_uid.h:41`：ResourceUID 类定义。",
          "`core/io/resource_importer.h:41`：ResourceFormatImporter 类定义。",
          "`editor/file_system/editor_file_system.h:145`：EditorFileSystem 类定义。",
          "`editor/file_system/editor_file_system.cpp:570`：判断是否需要重导入的关键入口。"
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "table",
        headers: ["现象", "先看哪里", "原因"],
        rows: [
          ["改了图片但游戏仍显示旧图", "`.import`、导入产物、EditorFileSystem reimport。", "源文件变了不代表内部资源已重新生成。"],
          ["移动资源后场景引用没断", "ResourceUID 和 uid path。", "文本资源可能通过 UID 还原路径。"],
          ["导出包缺模型贴图", "ExportPreset、导入产物、依赖列表。", "导出收集的是运行所需文件集合。"],
          ["运行时加载原始路径失败", "ResourceFormatImporter 和 `.import`。", "该类型可能只在编辑器导入后有内部资源可用。"]
        ]
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        headers: ["概念", "负责什么", "不负责什么"],
        rows: [
          ["ResourceLoader", "运行时按路径和 loader 加载 Resource。", "不负责编辑器扫描项目文件。"],
          ["ResourceImporter", "编辑器把源文件转换成内部资源。", "不直接调度 SceneTree。"],
          ["EditorFileSystem", "扫描项目、判断变化、安排重导入。", "不代表运行时文件系统 API。"],
          ["ResourceUID", "维护稳定资源身份。", "不保存资源对象内容。"],
          ["ExportPlatform", "决定哪些资源进入导出包。", "不重新解释所有导入格式语义。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：运行时总是直接加载原始素材。很多资源会先导入成内部产物。",
          "误区二：删除 `.import` 不会影响项目。它是 ResourceFormatImporter 找内部资源的重要元数据。",
          "误区三：UID 是文件内容 hash。UID 是稳定标识和路径映射，不等于内容校验。",
          "误区四：导出包应该包含所有源文件。导出通常只包含运行需要的文件和导入产物。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `ResourceUID`，理解 uid/path 映射。",
          "读 `ResourceFormatImporter`，确认 `.import` 如何影响 ResourceLoader。",
          "读 `EditorFileSystem` 的扫描和重导入判断。",
          "进入具体 importer，例如 texture、scene、font、wav，看它输出什么内部资源。",
          "最后读 export 平台收集文件的逻辑，确认导入产物如何进入包。"
        ]
      }
    ]
  },
  {
    id: "stringnamenodepath",
    title: "StringName / NodePath",
    aliases: ["StringName", "NodePath", "SNAME", "UNIQUE_NODE_PREFIX", "property path"],
    summary: "Godot 高频名字和路径基础设施：StringName 用于稳定高效的名字键，NodePath 用于节点路径和属性子路径引用。",
    article: [
      {
        type: "lead",
        text: "StringName 和 NodePath 很小，但几乎贯穿整个引擎。方法名、信号名、属性名、组名、主题名常用 StringName；动画轨道、场景引用、Inspector 属性路径和脚本导出 NodePath 都依赖 NodePath。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "StringName 像一张已经登记好的名字卡，比较时不用每次逐字比较字符串；NodePath 像一条带上下文的路线，既可以指向节点，也可以指向节点上的属性或子属性。"
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "`StringName` 定义在 `core/string/string_name.h:38`。它内部持有 `_Data`，保存字符串、hash、引用计数和链表指针。`SNAME()` 宏在 `core/string/string_name.h:212`，用静态局部变量缓存常用名字，避免热路径反复构造。"
      },
      {
        type: "paragraph",
        text: "`NodePath` 定义在 `core/string/node_path.h:34`。内部 Data 保存 `Vector<StringName> path`、`Vector<StringName> subpath`、拼接后的 StringName、absolute 标记和 hash cache。它不直接保存 Node 指针，解析必须结合当前节点或 SceneTree 上下文。"
      },
      {
        type: "table",
        title: "常见用途",
        headers: ["类型", "常见位置", "读法"],
        rows: [
          ["StringName", "方法、属性、信号、组、主题、ClassDB key。", "看它是不是稳定名字键，不要当普通可变字符串。"],
          ["SNAME", "热路径常量名字。", "表示这个名字会被重复使用，适合静态缓存。"],
          ["NodePath path", "`../Player/Camera` 这类节点路径。", "要结合当前节点解析。"],
          ["NodePath subpath", "`material:shader_parameter/color` 这类属性路径。", "可能指向资源属性或嵌套属性，不只是节点。"]
        ]
      },
      {
        type: "heading",
        title: "源码锚点"
      },
      {
        type: "list",
        items: [
          "`core/string/string_name.h:38`：StringName 类定义。",
          "`core/string/string_name.h:212`：SNAME 宏。",
          "`core/string/node_path.h:34`：NodePath 类定义。"
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "code",
        code: [
          "const StringName pressed = SNAME(\"pressed\");",
          "button->connect(pressed, callable_mp(this, &MyPanel::_on_pressed));",
          "",
          "NodePath target_path(\"../Player:position\");",
          "// path 部分找节点，subpath 部分可以继续指向属性。"
        ].join("\n")
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        headers: ["概念", "区别"],
        rows: [
          ["String", "普通字符串值，适合文本内容和可变字符串处理。"],
          ["StringName", "interned 名字键，适合重复比较和查找。"],
          ["NodePath", "路径值，不持有目标对象。"],
          ["ObjectID", "对象弱句柄，和路径解析不是一回事。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：StringName 只是 String 的别名。实际它有名字表和引用计数，比较语义不同。",
          "误区二：NodePath 保存 Node 指针。实际它只是路径，目标可能不存在或解析到不同对象。",
          "误区三：NodePath 只能指节点。它还能带 subpath 指向属性。",
          "误区四：所有字符串都该用 StringName。用户文本和临时字符串仍应使用 String。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `string_name.h` 的 `_Data` 字段，理解名字表和 hash。",
          "看 `SNAME()` 宏，理解为什么热路径喜欢静态 StringName。",
          "读 `node_path.h` 的 Data，区分 path 和 subpath。",
          "找一个动画轨道或 exported NodePath，追它如何解析成实际对象或属性。"
        ]
      }
    ]
  },
  {
    id: "errorandmemorymacros",
    title: "Error Macros / memnew / memdelete",
    aliases: ["Error Macros", "ERR_FAIL", "ERR_FAIL_COND", "ERR_FAIL_NULL", "ERR_FAIL_INDEX", "CRASH_COND", "memnew", "memdelete", "Memory::alloc_static", "Memory::free_static"],
    summary: "Godot C++ 基础读法：错误宏统一早返回、日志和崩溃语义；memnew/memdelete 接入内存统计和 Object 生命周期钩子。",
    article: [
      {
        type: "lead",
        text: "读 Godot C++ 时，错误宏和内存宏不是噪音。`ERR_FAIL_*` 决定非法输入时函数如何返回，`CRASH_*` 表达不可恢复错误，`memnew` / `memdelete` 则把分配释放接入 Godot 的内存统计、Object postinitialize 和 predelete 链路。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "错误宏像统一的门卫：条件不合法就记录原因并按约定返回。memnew/memdelete 像带登记流程的 new/delete：不只是申请和释放内存，还会让 Godot 的对象系统和调试工具知道发生了什么。"
      },
      {
        type: "heading",
        title: "深入解释"
      },
      {
        type: "paragraph",
        text: "错误宏集中在 `core/error/error_macros.h`。例如 `ERR_FAIL_INDEX` 从 `core/error/error_macros.h:134` 开始，`ERR_FAIL_COND` 在 `core/error/error_macros.h:412`。带 `_V` 的宏会返回指定值，不带 `_V` 的宏通常用于 void 函数；`CRASH_*` 更偏向内部不变量被破坏。"
      },
      {
        type: "paragraph",
        text: "`memnew` 宏定义在 `core/os/memory.h:150`，`memdelete` 在 `core/os/memory.h:160`。对 Object 子类，创建后会经过 `_post_initialize`，释放前会经过 predelete handler，这和 ObjectDB、信号断开、通知、调试统计都有关系。"
      },
      {
        type: "table",
        title: "宏读法",
        headers: ["宏", "代表什么", "读源码时注意"],
        rows: [
          ["`ERR_FAIL_COND(cond)`", "条件为真就报错并提前返回。", "看当前函数返回类型，确认后续逻辑不会执行。"],
          ["`ERR_FAIL_COND_V(cond, value)`", "条件为真就返回指定值。", "调用者必须能处理这个失败值。"],
          ["`ERR_FAIL_NULL(ptr)`", "空指针防御。", "不要把后续解引用当作必然安全，检查调用来源。"],
          ["`CRASH_COND(cond)`", "内部不变量失败，通常是严重 bug。", "比普通输入校验更强。"],
          ["`memnew(T)`", "用 Godot allocator 构造对象并 post-initialize。", "Object 子类不要随意绕开。"],
          ["`memdelete(ptr)`", "predelete、析构并释放内存。", "Node/RefCounted/RID 仍有各自正确释放通道。"]
        ]
      },
      {
        type: "heading",
        title: "源码锚点"
      },
      {
        type: "list",
        items: [
          "`core/error/error_macros.h:134`：索引错误宏入口。",
          "`core/error/error_macros.h:412`：条件错误宏入口。",
          "`core/os/memory.h:150`：memnew 宏定义。",
          "`core/os/memory.h:160`：memdelete 模板定义。"
        ]
      },
      {
        type: "heading",
        title: "案例"
      },
      {
        type: "code",
        code: [
          "Ref<Resource> load_checked(const String &path) {",
          "    ERR_FAIL_COND_V(path.is_empty(), Ref<Resource>());",
          "    Ref<Resource> res = ResourceLoader::load(path);",
          "    ERR_FAIL_COND_V(res.is_null(), Ref<Resource>());",
          "    return res;",
          "}"
        ].join("\n")
      },
      {
        type: "paragraph",
        text: "这个函数的失败路径不是异常，而是空 Ref 返回值。读调用者时要确认它是否检查了返回资源是否为空。"
      },
      {
        type: "heading",
        title: "相邻概念边界"
      },
      {
        type: "table",
        headers: ["概念", "边界"],
        rows: [
          ["ERR_FAIL_*", "防御性错误处理和早返回，不等同于测试断言。"],
          ["CRASH_*", "严重内部错误，不适合普通用户输入校验。"],
          ["memnew/memdelete", "内存和 Object 生命周期入口，不等同于 Node 的 queue_free。"],
          ["RefCounted", "引用计数释放路径，不能简单用 memdelete 代替。"],
          ["RID", "Server 内部对象释放要走 Server API，不靠 memdelete RID。"]
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：ERR_FAIL_* 只是日志。实际它通常会提前返回。",
          "误区二：带 `_V` 和不带 `_V` 没区别。带 `_V` 会返回指定值。",
          "误区三：memdelete 可以释放所有东西。Node、RefCounted、RID 都有更具体的生命周期规则。",
          "误区四：CRASH_* 可以用于普通用户错误。它更适合内部不变量损坏。"
        ]
      },
      {
        type: "heading",
        title: "建议阅读路线"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "读 `error_macros.h` 中 `_V`、`_MSG`、`CRASH` 系列的差异。",
          "遇到宏时先展开成“检查条件 -> 打印 -> 返回/崩溃”的控制流。",
          "读 `memory.h` 的 memnew/memdelete，理解 allocator 和 post/pre handler。",
          "回到 Object、Node、RefCounted、RID 的释放路径，区分不同生命周期通道。"
        ]
      }
    ]
  },
  {
    id: "navigationserver",
    title: "NavigationServer2D / NavigationServer3D",
    aliases: ["NavigationServer2D", "NavigationServer3D", "NavigationServer", "navigation map", "navigation region", "navigation agent"],
    summary: "导航不是物理的一部分：NavigationServer 管 map、region、agent、路径查询和避障，和 PhysicsServer 只在空间与时间轴上相邻。",
    article: [
      { type: "lead", text: "NavigationServer 负责路径搜索、导航网格、region、agent 和避障。它和 PhysicsServer 都处理空间，但目标不同：物理回答碰撞和运动约束，导航回答从哪里走到哪里。" },
      { type: "heading", title: "小白版解释" },
      { type: "paragraph", text: "物理像判断你有没有撞墙，导航像帮你规划绕过墙的路线。它们都和空间有关，但不是同一套系统。" },
      { type: "heading", title: "深入解释" },
      { type: "paragraph", text: "`NavigationServer2D` 定义在 `servers/navigation_2d/navigation_server_2d.h:50`，`NavigationServer3D` 定义在 `servers/navigation_3d/navigation_server_3d.h:46`。它们通过 RID 管理 map、region、link、agent 等对象，并提供路径查询和避障更新。" },
      { type: "heading", title: "源码锚点" },
      { type: "list", items: ["`servers/navigation_2d/navigation_server_2d.h:50`", "`servers/navigation_3d/navigation_server_3d.h:46`", "`servers/navigation_3d/navigation_server_3d.h:523`：NavigationServer3DManager。"] },
      { type: "heading", title: "案例" },
      { type: "paragraph", text: "NavigationAgent3D 请求路径时，场景节点保存用户 API 和目标点，底层查询会落到 NavigationServer 的 map/region 数据。角色真正移动仍由脚本、CharacterBody 或物理逻辑执行，导航只给路线和避障建议。" },
      { type: "heading", title: "相邻概念边界" },
      { type: "table", headers: ["概念", "区别"], rows: [["PhysicsServer", "碰撞、刚体、shape、空间查询。"], ["NavigationServer", "路径、网格、region、agent、避障。"], ["SceneTree", "调度节点和每帧阶段，不保存导航网格实现。"]] },
      { type: "heading", title: "常见误区" },
      { type: "list", items: ["误区一：导航会自动移动角色。实际它通常只提供路径和避障信息。", "误区二：物理碰撞体等于导航网格。导航网格需要烘焙或手动提供。", "误区三：导航查询就是物理 raycast。两者数据结构和结果语义不同。"] },
      { type: "heading", title: "建议阅读路线" },
      { type: "list", ordered: true, items: ["先看 NavigationServer 接口。", "读 map、region、agent 的 RID API。", "回到 scene 节点看 NavigationRegion/Agent 如何提交数据。", "最后看每帧更新和异步烘焙路径。"] }
    ]
  },
  {
    id: "multiplayerapi",
    title: "MultiplayerAPI / RPC",
    aliases: ["MultiplayerAPI", "MultiplayerAPIExtension", "RPC", "SceneMultiplayer", "MultiplayerPeer"],
    summary: "Godot 高层多人 API：Node/SceneTree 提供用户入口，MultiplayerAPI 负责 RPC、peer 和场景复制，传输层由模块或平台 peer 实现。",
    article: [
      { type: "lead", text: "MultiplayerAPI 是 Godot 高层网络游戏能力的入口。它让 Node 能发 RPC、按 peer 同步状态，并把底层 ENet/WebRTC/WebSocket 等传输差异藏在 MultiplayerPeer 后面。" },
      { type: "heading", title: "小白版解释" },
      { type: "paragraph", text: "可以把 MultiplayerAPI 看成游戏对象和网络连接之间的调度员：脚本说“调用远端这个方法”，它负责检查权限、目标 peer、路径和底层传输。" },
      { type: "heading", title: "深入解释" },
      { type: "paragraph", text: "`MultiplayerAPI` 定义在 `scene/main/multiplayer_api.h:36`，继承 RefCounted。SceneTree 和 Node 都持有或访问 multiplayer 相关入口，具体实现可以由模块提供，例如 multiplayer 模块、ENet、WebRTC 或扩展 peer。" },
      { type: "heading", title: "源码锚点" },
      { type: "list", items: ["`scene/main/multiplayer_api.h:36`：MultiplayerAPI。", "`scene/main/multiplayer_api.h:82`：MultiplayerAPIExtension。", "`modules/multiplayer`：高层 multiplayer 模块。", "`modules/enet`、`modules/webrtc`、`modules/websocket`：常见传输模块。"] },
      { type: "heading", title: "案例" },
      { type: "paragraph", text: "脚本里调用 RPC 时，用户看到的是 Node 方法；运行时要把 NodePath、方法名、peer、可靠性和权限转成 MultiplayerAPI 调用，再交给 MultiplayerPeer 发送。接收端反过来按路径找到节点并调用方法。" },
      { type: "heading", title: "相邻概念边界" },
      { type: "table", headers: ["概念", "边界"], rows: [["MultiplayerAPI", "高层 RPC 和复制语义。"], ["MultiplayerPeer", "底层连接和包传输。"], ["SceneTree/Node", "提供节点路径、权限和用户脚本入口。"], ["ENet/WebRTC/WebSocket", "具体传输实现或协议模块。"]] },
      { type: "heading", title: "常见误区" },
      { type: "list", items: ["误区一：RPC 是语言特性。实际它通过 Node、SceneTree 和 MultiplayerAPI 接入。", "误区二：MultiplayerAPI 等于 ENet。ENet 只是可能的传输实现之一。", "误区三：网络调用等同本地 call。它还要处理 peer、权限、路径和序列化。"] },
      { type: "heading", title: "建议阅读路线" },
      { type: "list", ordered: true, items: ["先读 `scene/main/multiplayer_api.h` 的接口。", "再看 Node/SceneTree 如何暴露 multiplayer 入口。", "读 modules/multiplayer 的 SceneMultiplayer。", "最后按传输选择 ENet/WebRTC/WebSocket。"] }
    ]
  },
  {
    id: "translationserver",
    title: "TranslationServer / Localization",
    aliases: ["TranslationServer", "Localization", "Translation", "locale", "CSVTranslation"],
    summary: "Godot 本地化中心：TranslationServer 管语言环境、翻译资源和文本查找，GUI、脚本和导入器都围绕它协作。",
    article: [
      { type: "lead", text: "TranslationServer 是 Godot 的本地化入口。它保存当前 locale、翻译资源列表和翻译查找逻辑，让脚本、Control、导入器和编辑器能用统一方式处理多语言文本。" },
      { type: "heading", title: "小白版解释" },
      { type: "paragraph", text: "可以把 TranslationServer 想成语言词典管理员。游戏问一个 key 在当前语言里怎么说，它按当前 locale 和已加载的翻译资源查找结果。" },
      { type: "heading", title: "深入解释" },
      { type: "paragraph", text: "`TranslationServer` 定义在 `core/string/translation_server.h:36`，继承 Object。它属于 core 字符串系统的一部分，而不是 GUI 专属功能。CSV、PO 等资源或导入器会把翻译内容注册进这个中心。" },
      { type: "heading", title: "源码锚点" },
      { type: "list", items: ["`core/string/translation_server.h:36`：TranslationServer 类入口。", "`core/register_core_types.cpp`：核心单例注册路径。", "`editor/import/resource_importer_csv_translation.h:35`：CSV 翻译导入器示例。"] },
      { type: "heading", title: "案例" },
      { type: "paragraph", text: "Label 显示本地化文本时，Control/Theme/TextServer 负责布局和绘制，但文本 key 到当前语言字符串的查找应回到 TranslationServer 和 Translation 资源。" },
      { type: "heading", title: "相邻概念边界" },
      { type: "table", headers: ["概念", "边界"], rows: [["TranslationServer", "翻译资源和 locale 查找。"], ["TextServer", "字体 shaping、断行和字形布局。"], ["Control/Label", "显示和布局 UI 文本。"], ["ResourceImporter", "把 CSV/PO 等源文件导入成 Translation 资源。"]] },
      { type: "heading", title: "常见误区" },
      { type: "list", items: ["误区一：本地化等于字体渲染。翻译查找和字形排版是两套系统。", "误区二：Label 自己保存所有语言。它通常拿 key 或文本交给翻译系统。", "误区三：导入翻译文件后就一定生效。还要确认资源加载、locale 和 TranslationServer 注册状态。"] },
      { type: "heading", title: "建议阅读路线" },
      { type: "list", ordered: true, items: ["先读 TranslationServer 接口。", "看 Translation 资源和导入器如何生成数据。", "回到 Control/Label 或脚本 tr() 调用路径。", "最后区分 TranslationServer 和 TextServer 的边界。"] }
    ]
  },
  {
    id: "deepdesignpatterns",
    title: "Ownership / queue_free / Thread Boundary / Registration",
    aliases: [
      "Ownership",
      "Lifecycle",
      "memnew",
      "memdelete",
      "ObjectDB",
      "ObjectID",
      "queue_free",
      "SceneTree::queue_delete",
      "MessageQueue",
      "call_deferred",
      "set_deferred",
      "ResourceCache",
      "RID_Owner",
      "WorkerThreadPool",
      "process thread group",
      "call_deferred_thread_group",
      "wrap_mt",
      "register_core_types",
      "register_server_types",
      "register_scene_types",
      "register_editor_types",
      "initialize_modules",
      "GDExtensionManager",
      "ERR_FAIL_COND"
    ],
    summary: "Godot 底层设计模式总图：谁拥有对象、什么时候删除、怎么跨线程、注册顺序为什么不能乱。",
    article: [
      {
        type: "lead",
        text: "这一节不是单个类，而是一组读 Godot 源码时反复遇到的底层规则：ObjectDB 只登记对象，RefCounted 才负责引用计数，Node 要通过 SceneTree 安全删除，RID 把场景层和 Server 内部数据隔开，MessageQueue 把危险操作推迟到安全点，初始化注册顺序决定一个类型什么时候能被脚本、Inspector、ResourceLoader 或 GDExtension 看见。"
      },
      {
        type: "heading",
        title: "小白版解释"
      },
      {
        type: "paragraph",
        text: "可以把 Godot 引擎想成一座大楼。ObjectDB 像前台登记簿，只记录谁现在还在楼里；它不替任何人付房租，也不负责把人送回家。Node 像房间里的家具，运行中要搬走最好先贴一个“稍后搬走”的标签，也就是 queue_free；Resource 像共享资料，谁还拿着 Ref，资料就不能销毁；RID 像取物牌，真正的物品放在 Server 仓库里；MessageQueue 像待办箱，把现在不适合做的事放到下一轮统一处理。"
      },
      {
        type: "flow",
        title: "一眼看懂这些规则怎么配合",
        steps: [
          { title: "登记", text: "Object 构造后进入 ObjectDB，获得 ObjectID。这个 ID 只能用来查对象是否还活着。" },
          { title: "拥有", text: "Node、Resource、RID 背后有不同所有权规则：场景树、Ref 引用计数、Server 内部 RID_Owner。" },
          { title: "延迟", text: "call_deferred、set_deferred、queue_free 把危险操作放进队列，在安全点统一执行。" },
          { title: "线程", text: "WorkerThreadPool、process thread group 和 wrap_mt 把并行执行限制在明确边界里。" },
          { title: "注册", text: "core -> servers -> scene -> editor 的注册顺序保证高层代码只使用已经准备好的底层能力。" }
        ]
      },
      {
        type: "heading",
        title: "为什么这组概念重要"
      },
      {
        type: "paragraph",
        text: "Godot 源码里很多难 bug 都不是“函数不会写”，而是边界没分清：把 ObjectID 当强引用、在树遍历中直接删除 Node、忘记释放 RID、在子线程直接改场景树、把 editor-only 注册放进运行时路径、手改生成文件。理解这组规则后，读源码时会先问“谁拥有它、它在哪个阶段释放、它能不能跨线程、它什么时候注册”。"
      },
      {
        type: "table",
        title: "底层设计模式速查表",
        headers: ["模式", "一句话", "源码入口", "常见错误"],
        rows: [
          ["ObjectDB / ObjectID", "登记活对象，提供弱查找；不拥有对象。", "`core/object/object.cpp:2450`、`core/object/object.h:908`", "保存 ObjectID 后以为对象不会被释放。"],
          ["memnew / memdelete", "Godot 的统一分配释放入口，接入统计、通知和调试检查。", "`core/os/memory.h:150`、`core/os/memory.h:160`", "绕过 memdelete，导致 Object predelete/postinitialize 链路不完整。"],
          ["RefCounted / Ref", "引用计数所有权，最后一个 Ref 释放后对象销毁。", "`core/object/ref_counted.h:36`、`core/object/ref_counted.cpp:74`", "裸指针传 RefCounted，或制造循环引用。"],
          ["Node / queue_free", "运行中删除节点先排队，等 SceneTree 安全点再释放。", "`scene/main/node.cpp:3461`、`scene/main/scene_tree.cpp:1637`", "信号、遍历、物理回调里立即删除当前节点。"],
          ["RID / RID_Owner", "对外暴露轻量句柄，真实对象由 Server 内部 owner 管理。", "`core/templates/rid.h:38`、`core/templates/rid_owner.h:91`", "丢掉 RID 不调用 Server::free，或把不同 Server 的 RID 混用。"],
          ["MessageQueue", "把 call/set 推迟到安全刷新点。", "`core/object/object.cpp:1998`、`core/object/message_queue.h:150`", "在不允许重入的阶段直接改对象或树结构。"],
          ["ResourceCache", "按路径复用 Resource，避免重复加载，也会带来共享状态。", "`core/io/resource.cpp:804`、`core/io/resource.cpp:852`", "以为 load 两次一定得到两个独立资源。"],
          ["Registration order", "类型、模块、扩展必须按依赖层级注册和反注册。", "`main/main.cpp:1059`、`main/main.cpp:3197`、`main/main.cpp:3759`", "在 CORE 层依赖 Scene 类型，或在导出模板使用 editor-only 类。"]
        ]
      },
      {
        type: "heading",
        title: "深入解释：所有权不是一个规则"
      },
      {
        type: "paragraph",
        text: "Godot 没有用一种 C++ 智能指针统一管理所有对象，因为引擎对象的用途差异很大。Object 是反射、信号、属性和脚本绑定的共同根；Node 要服务场景树和帧循环；Resource 要能被多个场景、材质、脚本或导入器共享；RID 要隐藏 Server 后端对象；Variant 要能在脚本和 C++ 之间搬运值。因此读释放逻辑时必须先识别对象类别。"
      },
      {
        type: "table",
        title: "所有权矩阵",
        headers: ["对象", "谁真正拥有", "怎么释放", "读源码时先看"],
        rows: [
          ["Object", "具体子类或创建者。ObjectDB 只登记。", "通常通过 `memdelete` 或子类生命周期释放。", "`Object::_construct_object()`、`Object::~Object()`、`ObjectDB::add_instance()`、`ObjectDB::remove_instance()`。"],
          ["Node", "场景树保存结构关系；父子关系决定通知和遍历顺序。", "树内运行时优先 `queue_free()`，由 SceneTree 删除队列释放。", "`Node::queue_free()`、`SceneTree::queue_delete()`、`SceneTree::_flush_delete_queue()`。"],
          ["RefCounted", "`Ref<T>` 引用计数。", "`unreference()` 后计数归零时删除。", "`RefCounted::reference()`、`RefCounted::unreference()`。"],
          ["Resource", "`Ref<Resource>` 管生命周期，路径缓存可能保留共享实例。", "最后一个 Ref 释放；缓存模式影响复用。", "`Resource::set_path()`、`ResourceCache::get_ref()`、`ResourceLoader::load()`。"],
          ["RID", "对应 Server 的 `RID_Owner` / `RID_PtrOwner`。", "必须走对应 Server 的 free 或专用释放 API。", "`RID_Alloc::make_rid()`、`get_or_null()`、`free()`。"],
          ["Variant", "值容器本身，内部按类型持有数据、Ref 或 Object 指针。", "Variant 析构时按内部类型释放或减少引用。", "`core/variant/variant.h` 的 Variant 类型分支。"]
        ]
      },
      {
        type: "flow",
        title: "queue_free 的真实路线",
        steps: [
          { title: "调用 Node::queue_free()", text: "`scene/main/node.cpp:3461` 检查节点是否已经排队删除。" },
          { title: "进入 SceneTree", text: "树内节点调用 `SceneTree::queue_delete(this)`，入口在 `scene/main/scene_tree.cpp:1637`。" },
          { title: "保存 ObjectID", text: "SceneTree 不直接保存裸指针，而是把 `get_instance_id()` 放进删除队列。" },
          { title: "安全点刷新", text: "`_flush_delete_queue()` 在主循环安全点取出 ObjectID。" },
          { title: "再次查 ObjectDB", text: "刷新时用 `ObjectDB::get_instance()` 确认对象还活着，再 `memdelete(obj)`。" }
        ]
      },
      {
        type: "code",
        language: "cpp",
        title: "安全删除节点的思路",
        code: [
          "void Enemy::_on_hit_zero_hp() {",
          "    // 当前函数可能来自信号、物理回调或树遍历。",
          "    // 让 SceneTree 在安全点删除，避免马上破坏当前调用链。",
          "    queue_free();",
          "    // 之后不要再假设 this 还能长期可用。",
          "    // 需要通知别人时，传 ObjectID 或提前断开关系，并在使用前重新校验。",
          "}"
        ].join("\n")
      },
      {
        type: "heading",
        title: "ObjectDB：弱索引，不是保险箱"
      },
      {
        type: "paragraph",
        text: "`Object` 内部保存 `_instance_id`。对象构造路径会调用 `ObjectDB::add_instance(this)`，析构时会从 ObjectDB 移除。`ObjectDB::get_instance()` 能把 ObjectID 查回当前仍然活着的 Object 指针，但它不会增加引用计数，也不会阻止对象析构。它适合做跨队列、跨信号、跨编辑器系统的弱查找。"
      },
      {
        type: "table",
        title: "ObjectDB 使用判断",
        headers: ["场景", "可以用 ObjectID 吗", "原因"],
        rows: [
          ["延迟删除队列", "可以", "队列里保存 ObjectID，刷新时再查 ObjectDB，能避免重复删除已经消失的对象。"],
          ["编辑器选择列表", "可以，但每次使用前要校验", "选中对象可能被场景重载、撤销或脚本释放。"],
          ["长期持有 Resource", "不合适", "Resource 应该用 Ref；ObjectID 不能表达拥有关系。"],
          ["保存当前帧局部对象", "通常不需要", "当前调用栈内直接用指针更清楚；跨帧才考虑弱 ID。"]
        ]
      },
      {
        type: "heading",
        title: "RID：句柄把场景层和后端数据隔开"
      },
      {
        type: "paragraph",
        text: "`RID` 类本身很小，重点在 Server 内部的 `RID_Alloc` 和 `RID_Owner`。`make_rid()` 创建句柄，`get_or_null()` 用句柄找真实数据，`free()` 释放槽位。RID 内部还带 validator，避免旧句柄在槽位复用后误指向新对象。RenderingServer、PhysicsServer、TextServer 都依赖这个模式，让场景节点不用知道 GPU buffer、物理 body、字体 shaping cache 的真实布局。"
      },
      {
        type: "flow",
        title: "RID 的生命周期",
        steps: [
          { title: "创建内部对象", text: "Server 在自己的 owner 里分配真实数据，例如 canvas item、mesh、body、shape。" },
          { title: "返回 RID", text: "场景层只保存 RID，后续所有修改都通过 Server API 带着 RID 进入。" },
          { title: "查找和校验", text: "Server 用 `get_or_null()` 根据 RID 找内部数据，并校验 validator。" },
          { title: "显式释放", text: "生命周期结束必须调用对应 Server 的 `free(rid)` 或专用释放函数。" }
        ]
      },
      {
        type: "heading",
        title: "MessageQueue：延迟不是偷懒，是防重入"
      },
      {
        type: "paragraph",
        text: "`Object::call_deferredp()` 会把调用压进 `MessageQueue::push_callp()`；`Object::set_deferred()` 会把属性设置压进 `push_set()`。这类延迟队列的价值是把“现在做会破坏遍历、通知或同步状态”的操作推到安全点。Godot 很多内部流程都在遍历树、广播信号、刷新物理查询、提交渲染命令；在这些阶段立即修改结构容易制造重入和悬空引用。"
      },
      {
        type: "table",
        title: "什么时候应该考虑 deferred",
        headers: ["当前阶段", "风险", "更稳的做法"],
        rows: [
          ["正在发射信号", "回调中删除发射者或监听者，后续监听器看到半销毁状态。", "用 `call_deferred()` 或 `queue_free()`。"],
          ["正在遍历 SceneTree", "删除当前节点会破坏遍历顺序。", "用 `queue_free()`，让 `_flush_delete_queue()` 处理。"],
          ["物理回调或查询刷新中", "直接改 shape/body 可能和 physics sync/flush 冲突。", "延迟到安全阶段，或走 Server 的同步接口。"],
          ["编辑器 Inspector 正在应用属性", "立即触发资源重载、节点替换或撤销动作会造成嵌套状态。", "交给 UndoRedo、deferred set 或编辑器专用队列。"]
        ]
      },
      {
        type: "heading",
        title: "线程边界：并行必须有入口和出口"
      },
      {
        type: "paragraph",
        text: "Godot 支持多线程，但不是“任何对象都能随便在任何线程改”。`WorkerThreadPool` 是通用任务池，`process thread group` 是场景层受控并行，`call_deferred_thread_group` 是线程组消息边界，`wrap_mt` 是 Server 调用的多线程包装。读这类源码时要找清楚提交任务的线程、真正执行任务的线程、同步点、flush 点以及哪些对象允许跨线程访问。"
      },
      {
        type: "table",
        title: "线程相关入口",
        headers: ["机制", "源码锚点", "你要检查什么"],
        rows: [
          ["WorkerThreadPool", "`core/object/worker_thread_pool.h:44`、`worker_thread_pool.cpp:385`、`:706`、`:727`", "任务是否需要等待，是否访问了非线程安全对象。"],
          ["process thread group", "`scene/main/node.cpp:1203`、`scene/main/scene_tree.cpp:1237`、`:1312`", "节点被分到哪个组，组任务何时提交、何时回到主线程。"],
          ["call_deferred_thread_group", "`scene/main/node.cpp:3673`、`:3914`", "消息是发给哪个线程组，刷新点在哪。"],
          ["PhysicsServer wrap_mt", "`servers/server_wrap_mt_common.h`、`physics_server_2d_wrap_mt.cpp:54`、`physics_server_3d_wrap_mt.cpp:54`", "调用是立即执行、排队执行，还是在 sync/flush/step 阶段执行。"]
        ]
      },
      {
        type: "code",
        language: "cpp",
        title: "读多线程代码时的检查清单",
        code: [
          "// 看到 WorkerThreadPool、thread group 或 wrap_mt 时，先写下四件事：",
          "// 1. 谁提交任务？",
          "// 2. 任务在哪个线程真正执行？",
          "// 3. 什么时候等待或 flush？",
          "// 4. 任务内部碰到的 Object/Resource/RID 是否允许跨线程访问？",
          "//",
          "// 如果回答不出来，不要急着改锁；先把生命周期和同步点画出来。"
        ].join("\n")
      },
      {
        type: "heading",
        title: "注册顺序：类型系统的施工顺序"
      },
      {
        type: "paragraph",
        text: "Godot 的注册不是随便把类塞进 ClassDB。`register_core_types` 要先建立基础类型；`register_server_types` 才能建立渲染、物理、音频、输入、文本等 Server；`register_scene_types` 再注册 Node、Resource 子类和用户可见场景 API；`register_editor_types` 只在工具构建中注册编辑器能力。模块和 GDExtension 也按初始化级别插入这些阶段。顺序错了，轻则类找不到，重则初始化时访问未创建的单例。"
      },
      {
        type: "table",
        title: "初始化和反初始化路线",
        headers: ["阶段", "典型入口", "能安全依赖什么"],
        rows: [
          ["CORE", "`main/main.cpp:1059`、`core/register_core_types.cpp:134`、`initialize_modules(MODULE_INITIALIZATION_LEVEL_CORE)`", "Object、Variant、基础 IO、Memory、ProjectSettings 等底层设施。"],
          ["SERVERS", "`main/main.cpp:3197`、`servers/register_server_types.cpp:146`、`initialize_modules(MODULE_INITIALIZATION_LEVEL_SERVERS)`", "Server 抽象和 manager，脚本语言、物理后端、文本后端可在这里接入。"],
          ["SCENE", "`main/main.cpp:3759`、`scene/register_scene_types.cpp:390`、`initialize_modules(MODULE_INITIALIZATION_LEVEL_SCENE)`", "Node、SceneTree、Control、Resource 子类、运行时 API。"],
          ["EDITOR", "`register_editor_types`、`initialize_modules(MODULE_INITIALIZATION_LEVEL_EDITOR)`、`TOOLS_ENABLED`", "EditorNode、EditorPlugin、Inspector、Importer、Exporter、语言服务。"],
          ["GDExtension", "`GDExtensionManager` 按同样级别初始化/卸载扩展", "外部动态库必须尊重当前 level，不能越级触碰还没注册的系统。"],
          ["Cleanup", "`main/main.cpp:5191` 起反向释放", "通常按 EDITOR -> SCENE -> SERVERS -> CORE 反向清理，避免高层对象引用已销毁底层。"]
        ]
      },
      {
        type: "heading",
        title: "ResourceCache：共享是设计，不一定是 bug"
      },
      {
        type: "paragraph",
        text: "`ResourceLoader::load()` 会根据缓存模式查 `ResourceCache`。资源路径写入时，`Resource::set_path()` 也会维护缓存表。同一路径资源被缓存后，多个加载者可能拿到同一个 Ref<Resource> 指向的实例；这对贴图、材质、场景、脚本很重要，因为它减少重复加载，也让编辑器能统一重载资源。但如果你把共享 Resource 当成临时对象修改，就可能影响所有引用它的地方。"
      },
      {
        type: "table",
        title: "ResourceCache 案例",
        headers: ["现象", "底层原因", "排查点"],
        rows: [
          ["加载同一材质后，改 A 影响 B", "两个对象引用了同一路径的同一个 Resource。", "看 `ResourceLoader::load()` 的缓存模式和 Resource path。"],
          ["资源文件变了但运行中没变化", "缓存仍返回旧实例，或导入资源没有重新加载。", "查 ResourceCache、imported resource、editor reload 逻辑。"],
          ["退出时报告 Resource still in use", "还有 Ref 持有资源。", "从 `ResourceCache::clear()` 输出和 Ref 持有路径找泄漏。"],
          ["需要独立可改副本", "共享缓存不是你想要的语义。", "考虑 duplicate、local_to_scene 或改用无路径临时资源。"]
        ]
      },
      {
        type: "heading",
        title: "错误处理和生成代码"
      },
      {
        type: "paragraph",
        text: "Godot 大量使用 `ERR_FAIL_COND*`、`ERR_FAIL_NULL*`、`ERR_PRINT*`。它们通常不是简单 assert，而是带返回值、带日志、在很多构建中仍然保留的早返回机制。读这些宏时要看失败后函数返回什么、调用者会不会继续、状态有没有部分修改。另一个常见陷阱是生成代码：模块总入口、类文档、extension API、图标、shader 等很多文件来自生成器，不要直接把生成文件当作唯一源头。"
      },
      {
        type: "heading",
        title: "案例：排查一个节点延迟删除后又被访问"
      },
      {
        type: "flow",
        title: "排查路线",
        steps: [
          { title: "确认触发点", text: "找谁调用了 `queue_free()`，是在信号、process、physics_process、编辑器操作还是异步回调里。" },
          { title: "查引用保存", text: "搜索裸指针、ObjectID、Callable、lambda 捕获和队列里的回调，确认谁还想访问这个节点。" },
          { title: "看删除刷新", text: "沿 `SceneTree::queue_delete()` 到 `_flush_delete_queue()`，确认对象什么时候真正 `memdelete`。" },
          { title: "校验 ObjectDB", text: "跨帧访问前用 ObjectID 重新查，找不到就停止后续逻辑。" },
          { title: "调整边界", text: "需要保活就重构所有权；不需要保活就断开信号、取消延迟调用或使用弱引用校验。" }
        ]
      },
      {
        type: "table",
        title: "读底层设计时的反问清单",
        headers: ["问题", "为什么要问", "常见入口"],
        rows: [
          ["谁拥有它？", "决定能不能跨帧保存、何时释放。", "ObjectDB、RefCounted、ResourceCache、RID_Owner、SceneTree。"],
          ["什么时候释放？", "决定当前指针、ObjectID、RID 是否还有效。", "`memdelete`、`queue_free`、`free(rid)`、cleanup。"],
          ["在哪个线程执行？", "决定是否能改场景树、访问 Object、提交 Server 命令。", "WorkerThreadPool、process thread group、wrap_mt。"],
          ["注册了吗？", "决定脚本、ClassDB、Inspector、ResourceLoader 是否能看到它。", "`register_*_types`、`initialize_modules`、`GDExtensionManager`。"],
          ["失败后返回什么？", "决定错误路径有没有部分状态残留。", "`ERR_FAIL_COND*`、`ERR_FAIL_NULL*`、`ERR_PRINT*`。"],
          ["这是源文件还是生成文件？", "决定应该改哪里。", "modules builders、doc/classes、extension API、generated headers。"]
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "Godot 底层设计的核心不是背类名，而是先分清所有权、延迟安全点、线程边界和注册层级；这四件事分清后，Object、Node、Resource、RID、Server、模块和编辑器代码就能串成一张稳定的源码地图。"
      }
    ]
  },
  {
    id: "sourcereadingroadmap",
    title: "Source Reading Roadmap / Feature Trace Checklist",
    aliases: [
      "阅读路线",
      "源码阅读路线",
      "学习路径",
      "功能追踪法",
      "入门路线",
      "机制路线",
      "深入路线",
      "中级路线",
      "底层设计",
      "深入层",
      "结束标准",
      "影响面",
      "checklist",
      "Feature Trace",
      "Source Reading Roadmap"
    ],
    summary: "把 Godot 源码阅读拆成三条路线：先建立全局骨架，再追一个功能端到端路径，最后检查生命周期、线程、注册、ABI 和保存格式影响。",
    article: [
      {
        type: "lead",
        text: "源码阅读路线不是“先读哪个文件”的死顺序，而是一套降复杂度的方法：先用入门路线建立地图，再用机制路线追一个真实功能，最后用深入路线检查生命周期、线程、注册、缓存、导出和二进制兼容影响。它的价值是让你读 Godot 时始终知道自己在回答什么问题，而不是在几百万行代码里随机跳转。"
      },
      {
        type: "heading",
        title: "入门解释"
      },
      {
        type: "paragraph",
        text: "如果你第一次读 Godot，不要从渲染后端、物理求解器或模板代码开始。先把引擎当成一座城市：`platform` 是城市入口，`main` 是调度中心，`core` 是基础设施，`scene` 是玩家和编辑器常见的对象世界，`servers` 是专业工厂，`modules` 是可插拔功能区，`editor` 是工具城区。你先学会看地图，再追一辆车从出发到到达。"
      },
      {
        type: "flow",
        title: "三条阅读路线",
        steps: [
          { title: "入门路线", text: "看顶层目录、版本、平台入口、Main::setup、Main::start、SceneTree 和 Node，能讲清引擎如何启动。" },
          { title: "机制路线", text: "选一个可见功能，从用户 API 追到 _bind_methods、内部状态、Resource、Server 和主循环刷新点。" },
          { title: "深入路线", text: "修改前检查所有权、线程边界、注册顺序、保存格式、GDExtension ABI、导入缓存和导出模板。" }
        ]
      },
      {
        type: "table",
        title: "每条路线的目标",
        headers: ["阅读阶段", "要回答的问题", "应该产出什么", "暂时不要做什么"],
        rows: [
          ["入门路线", "Godot 从平台入口到 Main、SceneTree、Node、Server、EditorNode 大概怎么串起来？", "目录依赖图、启动流程图、主循环阶段表。", "不要急着改内存、线程、资源格式、Server 后端。"],
          ["机制路线", "某个 API、节点、资源或编辑器按钮最终落到哪些类和函数？", "一条端到端调用链：用户操作/API -> 绑定 -> 对象状态 -> Resource/Server -> 主循环阶段。", "不要只改可见 bug 点；要确认序列化、Inspector、导入导出和测试。"],
          ["深入路线", "这个改动会不会破坏生命周期、缓存、线程边界、ABI 或生成代码？", "影响面清单、回归测试入口、构建配置差异、失败回滚点。", "不要在没有验证多平台/多构建目标时合并 core、servers、platform 层改动。"]
        ]
      },
      {
        type: "heading",
        title: "入门路线：先建立全局骨架"
      },
      {
        type: "paragraph",
        text: "入门路线的重点不是理解每个类，而是建立定位能力。`README.md:1` 告诉你这是 Godot Engine；`version.py:1` 到 `version.py:5` 给出 short_name、name 和版本号；顶层目录里 `core`、`scene`、`servers`、`modules`、`platform`、`editor` 基本就是第一张地图。启动路径可以从当前平台入口进入，再跟到 `Main::setup()`、`Main::start()` 和 `OS::run`。"
      },
      {
        type: "table",
        title: "入门路线文件地图",
        headers: ["入口", "先看什么", "你要得到的结论"],
        rows: [
          ["`README.md` / `version.py`", "`README.md:1`、`version.py:1`、`version.py:3` 到 `version.py:5`", "确认这是哪个引擎、哪个版本、这棵源码树对应什么构建目标。"],
          ["顶层目录", "`core`、`scene`、`servers`、`modules`、`platform`、`editor`、`drivers`、`thirdparty`、`tests`", "先按职责分块，而不是把所有目录都当平级代码。"],
          ["平台入口", "`platform/windows/godot_windows.cpp` 或当前平台入口", "平台层负责进程入口、OS 子类、原生事件循环和启动参数翻译。"],
          ["Main 初始化", "`main/main.cpp:1027` 的 `Main::setup()`、`:3988` 的 `Main::start()`", "Main 先搭底层，再决定跑编辑器、项目管理器、脚本、导出流程或普通游戏。"],
          ["主循环", "`scene/main/scene_tree.h:89`、`scene/main/scene_tree.cpp:639`、`:688`", "SceneTree 是默认 MainLoop，负责 physics_process、process、消息队列和删除队列。"],
          ["Server 层", "`servers/register_server_types.cpp`、RenderingServer、PhysicsServer、AudioServer", "节点不直接碰底层后端，很多重活交给 Server 和 RID。"]
        ]
      },
      {
        type: "heading",
        title: "机制路线：用功能追踪法读源码"
      },
      {
        type: "paragraph",
        text: "机制路线要从一个用户看得见的功能开始，而不是从最底层开始。例如“Sprite2D 设置 texture 后为什么会重新绘制”：先看 Sprite2D 的公开 API 和绑定，再看内部字段，接着看 CanvasItem 的 draw/RID，最后回到渲染刷新点。这样读出来的是一条可解释的链，而不是一堆孤立函数。"
      },
      {
        type: "flow",
        title: "功能追踪法",
        steps: [
          { title: "用户入口", text: "找类名、方法名、属性名或编辑器按钮，例如 Sprite2D.texture、PackedScene.instantiate、AnimationPlayer.play。" },
          { title: "绑定入口", text: "搜索 `_bind_methods()`，确认哪些方法、属性、信号通过 ClassDB 和 MethodBind 进入公开 API。" },
          { title: "内部状态", text: "读头文件成员和 setter，区分真实状态、缓存、Resource 引用和 Server 转发。" },
          { title: "资源/Server 边界", text: "看到 ResourceLoader、PackedScene、RID、RenderingServer、PhysicsServer 就继续追真实数据在哪里。" },
          { title: "主循环刷新", text: "确认状态在哪个阶段生效：process、physics_process、MessageQueue、flush、sync、draw、import、export。" },
          { title: "影响面", text: "写下脚本 API、Inspector、序列化、导入导出、测试和多线程影响，再决定怎么改。" }
        ]
      },
      {
        type: "table",
        title: "案例一：追 Sprite2D.texture",
        headers: ["步骤", "源码锚点", "读到什么"],
        rows: [
          ["找公开 API", "`scene/2d/sprite_2d.cpp:493` 的 `Sprite2D::_bind_methods()`", "`set_texture`、`get_texture`、`texture` 属性都在这里注册到 ClassDB。"],
          ["找 setter", "`scene/2d/sprite_2d.cpp:177` 的 `Sprite2D::set_texture()`", "setter 保存 `Ref<Texture2D>`，更新 region/filter 状态，并调用 `queue_redraw()`。"],
          ["找属性绑定", "`scene/2d/sprite_2d.cpp:494`、`:537`", "脚本、Inspector、文档都能看到 texture，因为 `ClassDB::bind_method` 和 `ADD_PROPERTY` 暴露了它。"],
          ["找绘制入口", "`scene/main/canvas_item.cpp:481` 的 `CanvasItem::queue_redraw()`、`:902` 的 texture draw", "Sprite2D 不直接画到屏幕；它把 redraw 需求和绘制命令交给 CanvasItem/RenderingServer。"],
          ["找 Server 边界", "`scene/main/canvas_item.h:84`、`:370`、`canvas_item.cpp:1801`、`:1806`", "CanvasItem 持有 canvas item RID，构造时创建，析构时释放，真实渲染对象在 RenderingServer 后端。"]
        ]
      },
      {
        type: "table",
        title: "案例二：追主场景加载",
        headers: ["步骤", "源码锚点", "读到什么"],
        rows: [
          ["Main 选择主场景", "`main/main.cpp:4737`", "普通游戏路径会用 `ResourceLoader::load(local_game_path)` 加载主场景资源。"],
          ["统一资源入口", "`core/io/resource_loader.cpp:513`", "`ResourceLoader::load()` 处理路径、类型 hint、缓存模式和具体格式 loader。"],
          ["PackedScene 是 Resource", "`scene/resources/packed_scene.h:246`", "场景文件被加载成 `PackedScene` 资源，不是直接加载成节点树。"],
          ["实例化节点树", "`scene/resources/packed_scene.cpp:2507`、`SceneState::instantiate()` 在 `:155`", "`PackedScene::instantiate()` 委托 SceneState 创建 Node 树。"],
          ["类实例创建", "`scene/resources/packed_scene.cpp:318`", "实例化节点时会通过 `ClassDB::instantiate()` 创建记录的类型。"]
        ]
      },
      {
        type: "code",
        language: "text",
        title: "功能追踪笔记模板",
        code: [
          "问题：这个功能从哪里进来，在哪里真正生效？",
          "",
          "1. 用户入口：类名 / 方法名 / 属性名 / 编辑器按钮",
          "2. 绑定入口：_bind_methods、ADD_PROPERTY、信号、文档类",
          "3. 内部状态：头文件字段、setter/getter、缓存、dirty flag",
          "4. 资源边界：ResourceLoader、ResourceSaver、ResourceCache、PackedScene",
          "5. Server 边界：RID、RenderingServer、PhysicsServer、AudioServer、TextServer",
          "6. 主循环阶段：process、physics_process、MessageQueue、sync、draw、flush",
          "7. 影响面：脚本 API、Inspector、序列化、导入导出、线程、测试"
        ].join("\n")
      },
      {
        type: "heading",
        title: "深入层：修改前写影响面"
      },
      {
        type: "paragraph",
        text: "深入层的目标是安全改引擎。Godot 的改动经常跨越脚本 API、编辑器 UI、资源保存、导入缓存、导出模板、多线程和 GDExtension ABI。尤其是 `core`、`servers`、`platform`、`main`、`scene` 这些层，局部函数看起来很小，实际可能影响脚本绑定、ClassDB 元数据、保存文件格式和扩展二进制接口。"
      },
      {
        type: "table",
        title: "影响面检查表",
        headers: ["问题", "为什么重要", "常见源码入口"],
        rows: [
          ["是否改了公开 API？", "脚本、文档、Inspector、C# 和 GDExtension 可能都受影响。", "`_bind_methods()`、`ClassDB::bind_method()`、`ADD_PROPERTY`。"],
          ["是否改了保存格式？", "已有项目文件、场景、资源能否继续加载取决于兼容逻辑。", "`ResourceFormatLoader`、`ResourceFormatSaver`、`PackedScene`、`SceneState`。"],
          ["是否改了生命周期？", "ObjectDB、RefCounted、ResourceCache、RID、queue_free 都可能出现悬空引用或泄漏。", "`memnew`、`memdelete`、`queue_free`、`RID_Owner::free()`。"],
          ["是否改了线程时序？", "主线程、WorkerThreadPool、Server wrap_mt 和 process thread group 的边界可能被破坏。", "`WorkerThreadPool`、`MessageQueue`、`wrap_mt`、`SceneTree::process`。"],
          ["是否只在编辑器存在？", "导出模板通常没有 editor-only 类型。", "`TOOLS_ENABLED`、`EditorPlugin`、`register_editor_types`。"],
          ["是否影响构建开关？", "模块可能被禁用，不同平台启用的后端也不同。", "`SConstruct:485`、`modules/SCsub`、`modules/*/config.py`、`modules_enabled.gen.h`。"]
        ]
      },
      {
        type: "flow",
        title: "从“能读”到“能改”的闭环",
        steps: [
          { title: "画路径", text: "先画功能调用链，明确用户入口、绑定、状态、资源/Server、主循环。" },
          { title: "列边界", text: "标出所有权、线程、注册、缓存、生成代码和平台差异。" },
          { title: "定改点", text: "选择最靠近真实原因且影响面最小的位置，不把表现层 bug 硬改到底层。" },
          { title: "补验证", text: "按影响面跑单元测试、编辑器路径、导入导出、脚本 API 或平台构建检查。" },
          { title: "回写笔记", text: "把最终路径和坑点补回文档，下一次读同类功能就能复用。"}
        ]
      },
      {
        type: "heading",
        title: "常见误区"
      },
      {
        type: "list",
        items: [
          "误区一：从最底层开始读最专业。实际更容易迷路；先从用户可见功能向下追，路径更完整。",
          "误区二：搜索函数名就等于理解功能。Godot 的公开 API、Inspector、文档和脚本绑定通常要看 `_bind_methods()` 和属性注册。",
          "误区三：看到 ResourceLoader 就只看文件格式。还要看 ResourceCache、路径 remap、UID、导入资源和实例共享。",
          "误区四：看到 RID 就以为是普通 ID。RID 背后有 Server owner、validator、显式释放和后端线程边界。",
          "误区五：编辑器里可用就代表游戏运行时可用。`TOOLS_ENABLED` 和 editor 初始化层经常只存在于工具构建。",
          "误区六：改完一个类能编译就结束。还要考虑脚本 API、保存格式、导入缓存、导出模板、多线程和测试。"
        ]
      },
      {
        type: "code",
        language: "text",
        title: "改动前影响面模板",
        code: [
          "改动目标：",
          "真正原因：",
          "",
          "公开 API：是否改 _bind_methods / ClassDB / 属性 / 信号？",
          "保存格式：是否影响 .tscn / .tres / imported 资源？",
          "生命周期：谁拥有对象？什么时候释放？是否涉及 queue_free / Ref / RID？",
          "线程边界：是否跨 WorkerThreadPool / MessageQueue / wrap_mt / process thread group？",
          "编辑器/导出：是否被 TOOLS_ENABLED 包住？导出模板是否需要这个能力？",
          "构建配置：模块开关、平台后端、thirdparty 依赖是否变化？",
          "验证：跑哪些测试、打开哪个页面、加载哪个示例场景？"
        ].join("\n")
      },
      {
        type: "heading",
        title: "建议读法"
      },
      {
        type: "list",
        ordered: true,
        items: [
          "先读 `README.md`、`version.py` 和顶层目录，确认源码树、版本和主要分区。",
          "沿平台入口到 `Main::setup()`、`Main::start()`、`SceneTree`，画出启动和主循环骨架。",
          "选一个具体功能，不要泛泛读目录；例如 Sprite2D.texture、PackedScene.instantiate、Button.pressed。",
          "搜索 `_bind_methods()`，确认脚本 API、Inspector 和文档如何看到这个功能。",
          "读头文件字段和 setter/getter，区分真实状态、缓存、dirty flag、Resource 和 RID。",
          "看到 ResourceLoader/PackedScene/RID/Server 就继续追边界，确认真实数据在哪里。",
          "回到主循环阶段，确认状态什么时候刷新、什么时候删除、什么时候提交给 Server。",
          "如果要改代码，先写影响面清单，再决定测试和回归路径。"
        ]
      },
      {
        type: "callout",
        title: "一句话总结",
        text: "读 Godot 源码的稳定方法是：先建地图，再追一条真实功能链，最后按影响面检查能不能安全修改；不要用随机搜索替代路径，不要用能编译替代验证。"
      }
    ]
  }
];

const startup = [
  {
    title: "1. 平台入口",
    source: "platform/windows/godot_windows.cpp:68",
    body: "平台入口负责把操作系统世界翻译成 Godot 能理解的跨平台入口。Windows 版本会创建 OS_Windows、把命令行转成 UTF-8、调用 Main::setup，然后在 Main::start 成功后进入 OS_Windows::run。",
    points: ["入口在 platform/*，不是 main/main.cpp。", "平台层处理原生命令行、进程退出码、窗口事件循环和 OS 子类。", "读其他平台时也按“创建 OS 子类 -> Main::setup -> Main::start -> OS::run”的形状找。"]
  },
  {
    title: "2. Main::setup",
    source: "main/main.cpp:1027",
    body: "第一阶段初始化 core 世界：让 OS 做基础初始化，创建 Engine，注册 core 类型，建立 ProjectSettings、InputMap、TranslationServer、Performance 等核心对象，并解析大量命令行和项目设置。",
    points: ["register_core_types 在 main/main.cpp:1059 附近发生。", "Object、Variant、ClassDB、ResourceLoader 这类基础设施必须先可用。", "渲染驱动、窗口参数、调试参数和项目路径很多都在这里写入全局状态。"]
  },
  {
    title: "3. Main::setup2",
    source: "main/main.cpp:3008",
    body: "第二阶段初始化依赖 core 的系统：注册 servers 类型，初始化 SERVERS level 模块和扩展，创建 Input、AccessibilityServer、DisplayServer，并在后段注册 scene 与 editor 类型。",
    points: ["register_server_types 在 main/main.cpp:3197 附近。", "DisplayServer::create 在 main/main.cpp:3343 附近，负责窗口和显示服务。", "register_scene_types 和 register_editor_types 更晚发生，SceneTree 还不是在这里创建。"]
  },
  {
    title: "4. 类型与模块层级",
    source: "main/main.cpp:3759",
    body: "Godot 按 CORE、SERVERS、SCENE、EDITOR 分层注册类型、模块和 GDExtension。模块调度代码由 modules/SCsub 与 modules/modules_builders.py 生成，决定每个模块在什么阶段能触碰哪些系统。",
    points: ["scene 类型在 server 层完成后注册。", "editor 类型只在 TOOLS_ENABLED 构建中注册。", "模块初始化 level 选错，常见后果是访问了还没建立的系统或污染了导出模板。"]
  },
  {
    title: "5. Main::start",
    source: "main/main.cpp:3988",
    body: "start 不是继续搭底层环境，而是决定这次进程到底要跑什么：编辑器、项目管理器、脚本、自定义 MainLoop、导出流程、文档工具，或者普通游戏主场景。",
    points: ["默认 MainLoop 是 SceneTree，也可以用 --main-loop 或脚本替换。", "main/main.cpp:4411 把创建好的 MainLoop 交给 OS。", "工具流程可能在 start 内提前返回，不进入普通帧循环。"]
  },
  {
    title: "6. EditorNode 或主场景",
    source: "main/main.cpp:4585",
    body: "选好 MainLoop 后，start 才把真正要运行的内容塞进 SceneTree。编辑器路径创建 EditorNode，项目管理器路径创建 ProjectManager，游戏路径加载 autoload 和主场景 PackedScene。",
    points: ["EditorNode 是场景树里的一个大型工具节点。", "普通游戏会用 ResourceLoader 加载主场景，再实例化成 Node 树。", "autoload 在主场景前加载，因此脚本里能把它当全局单例使用。"]
  },
  {
    title: "7. OS::run + Main::iteration",
    source: "platform/windows/os_windows.cpp:2343 / main/main.cpp:4896",
    body: "OS::run 是平台事件循环；Main::iteration 是跨平台一帧。Windows 每轮先处理 DisplayServer 事件，再调用 Main::iteration，由它推进计时、物理固定步、process、消息队列、导航、渲染、音频和调试器。",
    points: ["固定物理步可能一帧执行多次。", "SceneTree 的 process/physics_process 是 MainLoop 回调，不是平台代码直接调 Node。", "RenderingServer::sync/draw 在 process 后统一提交。"]
  },
  {
    title: "8. Main::cleanup",
    source: "main/main.cpp:5191",
    body: "cleanup 按反向依赖释放系统：先删除 MainLoop 和高层对象，再注销 editor、scene、servers、core 类型，并反向 uninitialize 模块和 GDExtension。",
    points: ["先高层，后底层。", "模块 uninitialize 按 EDITOR -> SCENE -> SERVERS -> CORE 反向执行。", "关闭崩溃通常要从残留对象、单例释放顺序和扩展卸载顺序定位。"]
  }
];

const paths = [
  {
    key: "2D 渲染路径",
    source: "scene/main/canvas_item.* -> servers/rendering",
    stack: ["Sprite2D/Control 改变可见状态或绘制命令", "CanvasItem 保存 canvas item RID", "Viewport/World2D 组织 canvas", "RenderingServer 接收 canvas_item_* 调用", "RendererCanvas/RendererRD 生成绘制命令", "RenderingDevice/DisplayServer 提交到窗口"],
    explain: "2D 的核心是 CanvasItem。Node2D、Control、Sprite2D 都在 CanvasItem 体系下，节点不直接画像素，而是把绘制状态和命令写入 RenderingServer。这样 GUI、2D 游戏对象和编辑器控件能共享一套渲染基础。"
  },
  {
    key: "3D Mesh 路径",
    source: "scene/3d -> RenderingServer -> renderer_rd",
    stack: ["MeshInstance3D 持有 Mesh Resource", "VisualInstance3D 在世界 scenario 中注册实例", "Material/Geometry/Transform 变更转成 RID 更新", "RenderingServer 维护 instance、mesh、material、light", "RendererSceneRD 做可见性、光照、阴影、后处理", "RenderingDevice 录制 GPU 命令"],
    explain: "3D 读源码要把资源和实例分开：Mesh/Material 是资源，MeshInstance3D 是场景实例，RenderingServer 内部还有真实渲染对象。一个 Mesh 可以被多个实例复用，每个实例有自己的变换、可见层和场景信息。"
  },
  {
    key: "资源加载路径",
    source: "core/io/resource_loader.h:103",
    stack: ["ResourceLoader::load(path)", "路径 remap 和 UID 解析", "选择 ResourceFormatLoader", "解析文本/二进制/导入资源", "ResourceCache 复用或替换", "返回 Ref<Resource> 给场景或编辑器"],
    explain: "资源系统的重点是统一入口和缓存策略。图片、场景、脚本、材质、字体、音频都是 Resource，但具体格式解析器不同。ResourceLoader 负责调度，ResourceFormatLoader 负责实际解析。"
  },
  {
    key: "脚本调用 C++",
    source: "core/object/class_db.h + method_bind.h",
    stack: ["GDScript 发起方法调用", "对象类型和方法名进入 ClassDB", "找到 MethodBind", "Variant 参数检查和转换", "调用 C++ 成员函数", "Variant 返回给脚本"],
    explain: "这个路径解释了为什么 Godot 的 C++ 类需要 _bind_methods。没有绑定，脚本、Inspector、序列化和文档都不知道这个 C++ 方法或属性存在。"
  },
  {
    key: "物理固定步",
    source: "main/main.cpp:4896 + scene/main/scene_tree.cpp:639",
    stack: ["MainTimerSync 计算需要几个 physics_steps", "SceneTree::iteration_prepare", "PhysicsServer sync / flush_queries", "SceneTree::physics_process", "NavigationServer physics_process", "PhysicsServer step", "MessageQueue flush 和 delete queue"],
    explain: "物理不是每个显示帧只跑一次，而是按固定 tick 追赶。Godot 在物理步中小心安排 sync、query、process、step 和 flush，避免用户脚本、物理后端和场景树状态互相踩踏。"
  },
  {
    key: "编辑器 Inspector",
    source: "editor/inspector + ClassDB PropertyInfo",
    stack: ["选中一个 Object/Node/Resource", "EditorInspector 查询属性列表", "ClassDB/对象自定义属性返回 PropertyInfo", "EditorProperty 控件按类型生成", "用户修改值", "Object::set / UndoRedo / 序列化更新"],
    explain: "Inspector 是理解编辑器复用运行时反射系统的最佳例子。C++ 属性通过 ClassDB 暴露后，脚本能访问，文档能生成，Inspector 也能自动创建编辑控件。"
  }
];

const beginnerGuides = {
  "这个模型怎么用": [
    "这一小节先教你把 Godot 想成几条同时运转的线，而不是一堆文件夹。最简单的入口是：用户在编辑器里看到 Node、Resource、Script；运行时用 SceneTree 安排这些对象什么时候进入树、什么时候 ready、什么时候每帧执行；更底层的 RenderingServer、PhysicsServer、AudioServer、DisplayServer 等负责和显卡、物理后端、音频设备、窗口系统打交道。",
    "读源码时不要急着问某个函数具体怎么写，先问它站在哪条线上：它是不是在创建进程和主循环，还是在注册 C++ 类型，还是在保存场景树状态，还是在把工作交给 Server 后端。能这样分类以后，你看到 Object、ClassDB、Variant、PackedScene、SceneTree、RID 这些词，就不会觉得它们是散的名词，而会知道它们分别解决可见对象、类型暴露、值传递、场景模板、帧调度和底层句柄的问题。"
  ],
  "把抽象落到一次真实动作：点击运行主场景": [
    "这一节用点击运行主场景把抽象概念串成一条真实路径。小白最容易误会的是：按下运行按钮以后，并不是编辑器里面直接把当前场景变成游戏，而是编辑器先做保存、检查、拼命令行参数，然后启动一个新的 Godot 游戏进程。这个新进程再像普通程序一样从平台入口启动，重新初始化 Main、Server、SceneTree，最后加载主场景。",
    "所以这条链路要分成两段看：编辑器段回答按钮在哪里、如何保存、如何创建游戏进程；游戏段回答新进程如何进入 Main::setup、Main::start，如何用 ResourceLoader 加载 PackedScene，如何实例化 Node 树并交给 SceneTree。你把这两段分清以后，就能解释为什么运行按钮在 editor 目录，主循环在 main 和 scene，资源加载在 core/io，渲染物理又会落到 servers。"
  ],
  "四个“不要混淆”": [
    "这一小节是在帮你建立防混淆清单。Godot 里面很多词看起来都像“对象”或“资源”，但源码职责完全不同。Object 是反射、信号、脚本挂载的基础；Node 是能放进 SceneTree 的 Object；Resource 是可保存、可缓存、可引用计数的数据；RID 是 Server 内部对象的编号句柄。它们都能在一次功能里出现，但不能互相替代。",
    "初学者读源码卡住，很多时候不是细节看不懂，而是把层级混了：以为 MeshInstance3D 就是 GPU mesh，以为 ResourceLoader 返回的 PackedScene 已经是场景树，以为 remove_child 会释放 Node，以为 RID 像 Ref 一样会自动管理生命周期。这个表的作用就是让你每次看到一个名词，都先问它到底保存用户语义、磁盘数据、树结构，还是底层执行状态。"
  ],
  "五个定位问题": [
    "这一小节给你一个固定提问模板。面对任何源码点，先别在函数里迷路，而是依次问：它怎么被创建，它的数据放在哪里，它什么时候执行，它怎么暴露给脚本和 Inspector，它会不会因为构建选项或模块开关而不存在。很多问题靠这五问就能定位到正确目录，而不是在全仓库乱搜。",
    "例如某个按钮属性不显示，先问它有没有通过 _bind_methods 和 ADD_PROPERTY 暴露；某个节点每帧不动，问它是否进入 SceneTree 的 process 列表；某个类源码存在但运行时找不到，问它是否被模块禁用或初始化 level 没跑到。五问本质上是在帮你从“现象”跳到“创建、状态、时机、公开 API、构建边界”这五个源码坐标。"
  ],
  "心智模型对应的源码锚点": [
    "这一小节把前面的心智模型落到具体文件。小白读大项目最怕地图只有概念没有坐标，所以这里列出的锚点可以当成第一批书签：平台入口看 platform，启动和每帧看 main/main.cpp，类型反射看 Object/ClassDB/MethodBind，值传递看 Variant，资源加载看 ResourceLoader/PackedScene，场景调度看 Node/SceneTree。",
    "读这些锚点时不要试图一次读完所有实现。更好的方式是先打开头文件看这个类保存什么字段、提供什么公开方法，再回到 cpp 看关键阶段。比如 SceneTree 先看它继承 MainLoop、保存 root/current_scene/delete queue，再看每帧 process 和 physics 的实现。锚点不是终点，而是你进入正确源码区域的入口。"
  ],
  "从一个功能反查源码的固定步骤": [
    "这一节教你用用户看得见的功能倒推源码。不要从底层猜起，而是从节点名、资源名、脚本 API、编辑器按钮、菜单项、文件格式开始。先找到公开名字，再找 _bind_methods 或注册代码，接着看这个功能的数据字段放在哪个类里，最后追它什么时候进入 Resource、SceneTree、Server 或平台层。",
    "这种读法特别适合 Godot，因为很多功能跨层很长：一个 Sprite2D 的 texture 看似只是属性，实际上会经过绑定、资源引用、CanvasItem、RenderingServer、渲染后端和主循环刷新。固定步骤能让你每次都产出一条链：用户入口 -> 绑定/注册 -> 对象状态 -> 底层执行 -> 刷新时机。"
  ],
  "顶层目录的职责边界": [
    "这一小节是在告诉你每个大目录大概负责什么。core 是所有系统都要用的地基，scene 是用户最常接触的节点和资源层，servers 是底层服务接口，modules 是可开关功能接入点，platform 是操作系统差异，editor 是编辑器工具程序，thirdparty 是外部库源码。",
    "小白不要把目录理解成普通分层书架，因为 modules 会插到不同初始化阶段，editor 又大量复用 scene/gui 和 core。更实用的判断是：这个文件定义的是基础类型、用户对象、底层服务、可选功能、平台适配，还是工具界面。只要能判断职责边界，你就能知道改某个功能应该先看哪个目录，哪些目录最好不要随便互相依赖。"
  ],
  "交互目录地图": [
    "这一小节的交互地图相当于一张可点击的源码城市地图。左侧选择目录，右侧会显示职责、边界、源码锚点、典型追踪路径、常见问题和误区。对初学者来说，它不是让你背每个目录，而是帮你在遇到问题时快速判断该从哪里进入。",
    "使用时可以带着一个具体问题点目录。比如你想知道 .tscn 怎么加载，就先看 core 的 ResourceLoader，再看 scene 的 PackedScene；想知道窗口和输入，就看 servers 的 DisplayServer 和 platform 的实现；想知道某个格式支持，就看 modules 或 drivers。地图的价值是让你少走弯路，先从 Godot 封装层看，再决定是否进入第三方库。"
  ],
  "目录之间的调用方向": [
    "这一节讲的是依赖方向。Godot 的大体原则是越底层越通用，越上层越接近用户：core 不应该知道 scene 的具体节点；scene 可以调用 servers；editor 可以复用 core、scene、servers；platform 把 OS 和 DisplayServer 的抽象落地；modules 按初始化级别插入不同层。",
    "小白可以把它想成盖楼：core 是地基，servers 是水电管线接口，scene 是房间和家具，editor 是用这些房间搭出来的装修工具，platform 是把房子接到不同城市的真实水电网络。地基不能依赖楼上的家具，否则一改上层就会震到底层；这就是为什么读源码时必须尊重调用方向。"
  ],
  "三条典型追踪路线": [
    "这一小节给你几个可复用的练习题：属性如何出现在 Inspector，.tscn 如何变成场景树，Mesh 如何显示到窗口，物理移动如何得到碰撞结果。它们覆盖了 Godot 最常见的跨层路径：绑定和编辑器、资源加载和实例化、场景对象到渲染 Server、场景物理到物理后端。",
    "小白读的时候不要只看表里的第一列，要特别关注结束标准。能说出函数名还不够，你要能解释数据从哪里来、状态存在谁身上、什么时候刷新、最后谁执行。比如 Mesh 路线的结束不是找到 MeshInstance3D，而是能区分 Mesh Resource、场景实例 RID、RendererRD 后端对象和 DisplayServer 窗口输出。"
  ],
  "构建系统的可用心智模型": [
    "这一节把 SCons 构建系统解释成装配线。SConstruct 不是普通配置文件，而是会被 Python 执行的总调度脚本；platform/*/detect.py 判断当前平台能不能构建；各目录 SCsub 把自己的 cpp、生成文件、第三方源码和宏开关加入构建环境；最后平台层把对象文件链接成 editor 或 export template。",
    "小白最需要记住的是：构建系统会决定哪些源码真的存在于最终二进制。你在仓库里看到一个类，不代表运行时一定能创建它，因为模块可能被关闭，disabled_classes 可能裁剪类，TOOLS_ENABLED 可能让编辑器代码只存在于 editor 目标。读构建系统，就是读“哪些代码会被编译、哪些宏会生效、哪些注册函数会进入运行时”。"
  ],
  "一次 scons platform=windows target=editor 发生什么": [
    "这一节把一次具体构建拆开。运行 `scons platform=windows target=editor` 时，SCons 先执行 SConstruct，解析 platform、target、arch、模块开关和裁剪选项；再检查 Windows 平台能否构建；然后按固定顺序进入 core、servers、scene、editor、drivers、platform、modules、tests、main 等 SCsub 收集源码。",
    "对小白来说，关键不是记住所有行号，而是分清构建的阶段：选项解析、平台检测、模块发现、源码收集、生成代码、编译、链接。构建失败时也按这个顺序排查。比如类找不到可能是模块没启用，链接失败可能是新 cpp 没加入 SCsub，template 构建失败可能是运行时代码无条件引用了 editor-only 类型。"
  ],
  "构建参数如何改变源码世界": [
    "这一节解释为什么同一份源码在不同构建参数下会变成不同世界。platform 会决定 OS、DisplayServer、系统库和平台 SCsub；target 会决定是编辑器还是导出模板；TOOLS_ENABLED 会让编辑器、导入器和调试工具参与编译；module_* 和 disable_* 会决定某些类、格式、后端是否存在。",
    "小白读代码时如果看到 #ifdef、MODULE_xxx_ENABLED、TOOLS_ENABLED、disabled class，不要把它当成边角细节。它们可能直接解释为什么你本地能运行、模板不能运行，为什么源码里有类但脚本创建失败，为什么某个平台有某个后端而另一个平台没有。构建参数就是源码世界的开关面板。"
  ],
  "SCsub 的固定读法": [
    "这一节教你怎么读每个目录里的 SCsub。第一眼先找它把文件加入哪个 source 列表，例如 core_sources、scene_sources、servers_sources、editor_sources 或 modules_sources；第二眼看它递归进入哪些子目录；第三眼看有没有 CommandNoCache 或 builder 生成 .gen.cpp/.gen.h。",
    "SCsub 不是运行时逻辑，而是编译前的装配说明。读它可以回答很多现实问题：为什么新文件没被编译，为什么某个第三方库被加进来了，为什么生成文件内容不是手写的，为什么 editor 目标有这个代码而 template 没有。把 SCsub 读顺以后，你就能从构建错误反推到具体目录的装配规则。"
  ],
  "生成文件不要手改，要改生成源头": [
    "这一节强调一个很重要的工程习惯：看到 .gen.h 或 .gen.cpp，不要把它当成普通源码直接改。生成文件通常由 SCons builder 根据模块列表、文档、类裁剪、shader、图标或接口描述自动生成。你手改生成物，下次构建就会被覆盖，还会让真正的源头没有变化。",
    "小白可以把生成文件想成打印出来的报表，真正的数据在表格源、脚本或配置里。比如 modules/register_module_types.gen.cpp 来自 modules_builders.py 和模块列表；disabled_classes.gen.h 来自 build profile；编辑器文档头文件来自 doc XML。要改行为，就回到 builder、SCsub、config.py 或输入数据。"
  ],
  "模块注册：构建结果怎样进入运行时": [
    "这一节解释模块不是编译进去就自动生效。Godot 先在构建阶段决定启用哪些 modules，再生成统一注册函数；运行时 Main 会在 CORE、SERVERS、SCENE、EDITOR 不同初始化级别调用 initialize_modules。每个模块根据当前 level 注册基础类型、Server 后端、Node/Resource、编辑器插件等内容。",
    "小白最容易忽略初始化级别。一个模块如果在 CORE 阶段注册，就不能依赖还没建立的 SceneTree 或 Server；如果在 EDITOR 阶段注册，就通常不能进入导出模板。读模块时先看 register_types.cpp 和 config.py，再问它在哪个 level 做了什么，这样才能解释功能为什么在某些构建里存在、在另一些构建里不存在。"
  ],
  "读构建错误的定位顺序": [
    "这一节给你排查构建错误的顺序。不要看到报错就只盯最后一行，而是先判断失败发生在哪个阶段：平台识别失败、模块没启用、生成文件没生成、编译找不到头文件、链接找不到符号、模板目标引用了 editor-only 类型。不同阶段要看的文件完全不同。",
    "比如平台不在列表里看 platform/*/detect.py；模块没有编译看 config.py、SCsub、register_types.h 和 module_* 开关；类源码存在但运行时找不到看 ClassDB 注册和 disabled_classes；链接失败看新 cpp 是否加入正确列表。按阶段定位，能把构建问题从一团日志缩小到几类可验证原因。"
  ],
  "启动链路总图": [
    "这一节用图把 Godot 启动拆成几个大步骤：平台入口创建 OS 子类，Main::setup 建立 core 世界，Main::setup2 建立 Server、Display 和 Scene 类型，Main::start 决定运行编辑器、游戏、项目管理器或工具流程，OS::run 进入平台事件循环，Main::iteration 推进每一帧，最后 cleanup 反向清理。",
    "小白不要把启动理解成一个函数从头跑到尾。Godot 的启动像搭舞台：先把跨平台地基搭好，再把底层服务和场景类型注册好，然后才决定今天演的是编辑器、游戏还是导出工具。每一帧也不是 OS 直接调脚本，而是 OS 的事件循环定期把控制权交给 Main::iteration，由它统一安排物理、process、消息队列、渲染和清理。"
  ],
  "从 Windows exe 到第一帧的源码追踪": [
    "这一节选择 Windows 作为最直观样本，带你从可执行文件入口跟到第一帧。Windows 入口先处理宽字符命令行、创建 OS_Windows，再调用 Main::setup；setup 建立 core 单例和项目设置；setup2 注册 Server、DisplayServer、Scene 和 Editor 类型；start 创建 MainLoop，并按运行模式加入 EditorNode 或主场景。",
    "初学者看这条链时要关注“谁负责跨平台，谁负责平台差异”。platform/windows 只处理 Windows 的入口、窗口事件循环、系统 API 差异；真正的跨平台初始化顺序在 main/main.cpp。这样你读 Linux、macOS、Android、Web 时也能迁移同一套骨架：平台 glue 不同，但最终都会回到 Main 和 MainLoop。"
  ],
  "点击查看每个阶段": [
    "这一小节把启动阶段做成可点击列表，是为了让小白一次只看一个阶段。启动源码很长，如果从 main.cpp 顶部一路读到底，很容易忘记自己在哪个阶段。点开阶段时只问三个问题：这一阶段的输入是什么，创建了哪些全局对象或类型，下一阶段依赖它提供什么。",
    "比如 Main::setup 的重点是 core、命令行、ProjectSettings、InputMap；setup2 的重点是 Server、DisplayServer、Scene 类型；start 的重点是选择运行模式和 MainLoop；OS::run 的重点是平台事件循环如何反复调用 Main::iteration。分阶段读，能把几千行启动代码拆成可消化的小块。"
  ],
  "不同运行模式的分叉点": [
    "这一节解释同一个 Godot 可执行文件为什么能当编辑器、项目管理器、游戏、脚本执行器、导出工具或文档工具。分叉点主要在 Main::start，它会根据命令行参数、项目设置和构建目标决定创建 EditorNode、ProjectManager、SceneTree 主场景，还是执行其他工具流程。",
    "小白要记住：setup/setup2 更多是在搭环境，start 才是在决定跑什么。你调试“为什么没有进入游戏主场景”时，不应该只看 ResourceLoader，也要看命令行和 Main::start 的运行模式判断。编辑器路径和游戏路径在同一套运行时地基上分叉，这也是为什么编辑器能复用场景树、GUI、资源系统和 ClassDB。"
  ],
  "ProjectSettings：项目配置如何影响启动": [
    "这一节补的是启动链路里的项目配置层。Godot 不是硬编码主场景、autoload 和输入映射，而是先把 project.godot、命令行、资源包和 feature override 变成 ProjectSettings，再由 Main::start 按这些设置决定这次游戏进程要加载什么。",
    "小白可以把 ProjectSettings 想成项目的总配置表。主场景路径、autoload 单例、InputMap、窗口和语言设置都可能从这里进入运行时。排查启动后没有主场景、autoload 不存在、输入 action 不生效、资源包路径不对时，要把 ProjectSettings 放在 Main 和 ResourceLoader 之间一起看。"
  ],
  "主循环一帧里发生什么": [
    "这一节讲的是 Godot 真正“动起来”的一帧。Main::iteration 会计算时间，决定固定物理步要跑几次，调用 SceneTree 的 physics_process 和 process，刷新 MessageQueue，推进导航，最后让 RenderingServer sync/draw。用户脚本看到的 _physics_process 和 _process，其实都被放在这条时间轴里。",
    "初学者最容易混淆显示帧和物理帧。显示帧跟屏幕刷新和渲染节奏有关，物理帧通常按固定 tick 运行，一帧显示里可能跑零次、一次或多次物理步。理解这一点以后，你就能解释为什么物理相关逻辑要放在 _physics_process，为什么延迟调用和 queue_free 要等安全点执行，为什么渲染提交通常在 process 后统一发生。"
  ],
  "读启动源码时最容易踩的坑": [
    "这一节列的是启动源码常见误区。不要以为 main/main.cpp 是平台入口，各平台真正入口在 platform；不要以为 Main::start 是全部初始化，很多 core 和 Server 类型在 setup/setup2 已经注册；不要以为编辑器和游戏是完全不同程序，编辑器也是 SceneTree 上的大型工具场景。",
    "另一个常见坑是忽略清理顺序。Godot 关闭时要反向注销 Editor、Scene、Servers、Core，模块也按反向 level 清理。如果某个高层对象在底层单例释放后还访问它，就可能出现关闭崩溃。读启动源码要把初始化和 cleanup 放在一起看，因为生命周期的前半段和后半段是互相对应的。"
  ],
  "运行时核心部件": [
    "这一节把 Object、ClassDB、Variant、StringName、MethodBind、Signal、Callable、MessageQueue、ObjectDB 等核心概念放在一起。它们不是独立知识点，而是共同支撑“C++ 对象能被脚本、编辑器、序列化、信号系统统一使用”的运行时地基。",
    "小白可以这样理解：Object 给对象一张身份证和基础能力；ClassDB 像登记处，记录类、方法、属性、信号；Variant 是统一的参数盒子；StringName 是高效的名字标签；MethodBind 是脚本调 C++ 的桥；Signal 和 Callable 负责事件和可调用目标；MessageQueue 负责延迟到安全点执行；ObjectDB 负责用 ObjectID 找回对象但不拥有对象。"
  ],
  "一个 C++ 类怎样变成 Godot API": [
    "这一节解释 C++ 类不是写出来就会自动出现在脚本和 Inspector 里。它通常要继承 Object 体系，使用 GDCLASS 建立类信息，在 register_types 或对应注册函数里进入 ClassDB，再在 _bind_methods 里绑定方法、属性、信号和常量。这样脚本、编辑器、文档、序列化才知道它的公开 API。",
    "小白可以把这个过程想成把内部零件上架到公共商店。C++ 函数本来只是编译器知道的成员函数；bind_method 以后，Godot 的运行时表里才有名字、参数、返回值和调用方式。Inspector 看到属性，不是编辑器猜出来的，而是 ClassDB 和对象属性列表提供的。"
  ],
  "MethodBind：脚本调用 C++ 的桥": [
    "这一节专门讲 MethodBind。脚本调用 node.rotate(1.0) 时，脚本并不知道 C++ 成员函数指针在哪里，它只知道对象和方法名。ClassDB 根据对象类型和方法名找到 MethodBind，MethodBind 再负责检查参数数量、默认值、类型转换、返回值，并最终调用真实 C++ 函数。",
    "对小白来说，MethodBind 的意义是把动态世界和静态 C++ 世界接起来。脚本里的值通常以 Variant 形式传入，通用但有包装成本；高性能路径可以用 ptrcall 直接传原始指针。读脚本调用问题时，如果方法存在但调用失败，就要看绑定签名、默认参数、Variant 转换和 MethodBind 的错误返回。"
  ],
  "Object 生命周期：普通对象、Node、Resource 不一样": [
    "这一节强调 Object 体系里的对象释放规则并不相同。普通 Object 通常由创建者手动释放或由特定 API 管理；Node 可以进入 SceneTree，运行中常用 queue_free 延迟删除；Resource 继承 RefCounted，常用 Ref 引用计数管理；Callable 通常保存对象 ID 和方法名，调用前要检查目标是否还存在。",
    "小白最容易把 Node 当智能指针对象，或者把 Resource 当场景节点。记住：Node 的父子关系管理树结构和析构清理，但不是 Ref<Node>；Resource 可以被 ResourceLoader 缓存和多个使用者共享，但不能挂到场景树；ObjectDB 只登记对象 ID，不负责延长生命。生命周期不分清，读源码时很容易误判泄漏、重复释放或悬空引用。"
  ],
  "信号、Callable、MessageQueue 如何协作": [
    "这一节解释 Godot 的事件和延迟调用机制。Signal 不是全局广播站，而是某个 Object 保存的一组连接；Callable 统一表示一个可以被调用的目标，可能是对象方法、脚本函数或自定义 callable；MessageQueue 则把 call_deferred、set_deferred、notification 等操作暂存起来，等安全点再执行。",
    "小白可以把它想成：Signal 负责“有人按了门铃”，Callable 记录“应该通知谁、怎么通知”，MessageQueue 负责“现在不方便马上进屋，先排队，等当前遍历或物理同步结束再处理”。这样做是为了避免在发信号、遍历场景树、同步物理时立刻改结构导致重入、迭代器失效或对象已经释放还被访问。"
  ],
  "读运行时问题的定位问题": [
    "这一节把运行时常见现象转换成定位问题。脚本找不到方法，就查类是否注册、_bind_methods 是否绑定、方法名是否一致；Inspector 不显示属性，就查 ADD_PROPERTY、PropertyInfo usage 和 getter/setter；信号不触发，就查信号注册、连接、Callable 有效性和 block signals。",
    "小白调试时不要只看报错点，要顺着运行时核心的责任链查。对象访问异常可能是保存了裸指针而不是 ObjectID 或 Ref；延迟调用时机不对可能是 MessageQueue flush 阶段和你想的不一样；属性变化没生效可能是绑定到了 setter 但内部状态或通知没更新。把现象翻译成这张表里的问题，定位会快很多。"
  ],
  "从 ResourceLoader 到 PackedScene": [
    "这一节讲 .tscn 进入运行时的第一半路径。ResourceLoader 是统一调度器，它根据路径、类型、缓存模式和已注册的 ResourceFormatLoader 选择具体解析器；文本场景、二进制资源、图片、脚本、导入产物都可以通过这套入口加载。加载 .tscn 的结果通常是 PackedScene，而不是直接得到一棵已经运行的 Node 树。",
    "小白要把文件、资源模板、节点实例分开：.tscn 是磁盘上的描述文件，PackedScene 是内存里的 Resource 模板，SceneState 是 PackedScene 内部保存节点表、属性表、连接表的数据，instantiate 之后才创建真实 Node 对象。ResourceLoader 也可能命中缓存，所以“加载到资源”不等于“创建了新的场景实例”。"
  ],
  "Node 与 SceneTree 的职责边界": [
    "这一节把 Node 和 SceneTree 分清。Node 保存自己的父子关系、owner、组、路径、process 开关、生命周期通知和脚本入口；SceneTree 是 MainLoop 的实现，保存根窗口、当前场景、组调用、Timer/Tween、删除队列和每帧调度。Node 是树上的单位，SceneTree 是调度整棵树的系统。",
    "小白可以用班级比喻：Node 是每个学生，保存自己的名字、所在小组和状态；SceneTree 是班主任和课程表，决定什么时候点名、上课、分组活动、放学清理。学生自己不会决定全班每一帧怎么运行，SceneTree 也不会保存每个学生的业务属性。把职责边界分清，读生命周期和场景切换就不容易乱。"
  ],
  "Node 生命周期逐步看": [
    "这一节把 Node 从创建到进入树、ready、process、退出、释放的过程拆成步骤。一个 Node 被 new 或 PackedScene 实例化以后，还不一定在 SceneTree 里；add_child 以后才会获得 tree 指针并触发 enter_tree；子树准备好后触发 ready；开启 process 或 physics_process 后才会被每帧调度。",
    "小白特别要理解 ready 和 process 的区别。ready 是节点第一次进入树并且子节点也准备好后的初始化点，适合拿子节点引用；process 是每帧持续运行；physics_process 是固定物理步运行。queue_free 也不是立刻 delete，而是进入删除队列，等安全点释放，这样可以避免遍历节点树时把当前对象删掉。"
  ],
  "owner：为什么节点在树里却没有保存进 .tscn": [
    "这一节讲 owner，这是编辑器保存场景时非常容易让小白困惑的概念。parent 决定节点在运行时树上的父子结构，owner 决定这个节点属于哪个可保存的场景资源。一个节点可以挂在树里，但如果 owner 不属于当前保存的场景，保存 .tscn 时它可能不会被写进去。",
    "你可以把 parent 想成“住在哪个房间”，owner 想成“归哪份房产证登记”。编辑器里动态创建子节点后，如果只 add_child 而没有设置 owner，它在树里能看到、能运行，但保存场景时不会作为当前场景的一部分持久化。读 PackedScene.pack、编辑器保存、子场景实例和 editable children 时，owner 是必须看的字段。"
  ],
  "SceneTree 每帧的内部节奏": [
    "这一节把 SceneTree 放回每帧时间轴。Main::iteration 会把控制权交给 MainLoop，SceneTree 再处理 physics_process、process、group call、Timer、Tween、场景切换、删除队列和消息队列。不同阶段之间会有 flush 和安全点，保证延迟调用、删除、物理同步不会互相踩踏。",
    "小白不要把每帧理解成简单地遍历所有 Node 调 _process。实际 Godot 要考虑固定物理步、暂停模式、process 优先级、线程组、消息队列、导航、渲染同步和删除队列。读某个节点为什么这一帧没被调用时，要看它是否在树里、process 是否启用、是否暂停、是否处在正确线程组，以及当前帧到底跑的是物理阶段还是普通 process 阶段。"
  ],
  "Viewport、Window 和 World：场景连接输入与渲染的枢纽": [
    "这一节补的是 SceneTree 和 RenderingServer 之间的中间层。Viewport 不是普通节点容器，它负责输入分发、GUI 根、2D canvas、3D world、渲染目标和调试绘制；Window 是能接到 DisplayServer 的 Viewport；World2D/World3D 则保存对应空间的物理、导航和渲染归属。",
    "小白排查“节点在树里但不显示”“鼠标事件没到 GUI”“3D 对象没有进入世界”“窗口 resize 后画面不对”时，不要直接从 Node 跳到 GPU。先问这个节点属于哪个 Viewport，Viewport 绑定哪个 Window 或 render target，2D/3D 对象进入了哪个 World。"
  ],
  "场景与资源常见定位问题": [
    "这一节把场景和资源问题整理成排查表。加载不到资源先看路径 remap、UID、导入产物和缓存模式；场景实例化缺节点先看 PackedScene 的 SceneState、owner 和子场景实例；节点回调没触发先看是否进入 SceneTree、process 是否开启、暂停模式是否阻止。",
    "小白调试时要避免把所有问题都归到 ResourceLoader。ResourceLoader 只负责把路径变成 Resource；instantiate、enter_tree、ready、process、owner、queue_free 都是后面的系统。把问题分到“文件加载、资源模板、节点实例、树上生命周期、每帧调度、保存持久化”这几个阶段，基本就能知道下一步该看 core/io、scene/resources 还是 scene/main。"
  ],
  "RID：Server 世界里的轻量句柄": [
    "这一节讲 RID。小白可以先把 RID 理解成 Server 内部对象的号码牌。场景层的 MeshInstance3D、RigidBody3D、AudioStreamPlayer 等对象不会直接拿着 GPU 对象、物理 body 或音频后端对象的 C++ 指针，而是拿一个 RID，通过 RenderingServer、PhysicsServer、AudioServer 这类接口去操作后端状态。",
    "RID 的好处是隔离和可替换：场景层只知道有一个句柄，不需要知道后端是 Vulkan、Jolt、Godot Physics 还是别的实现。代价是生命周期必须按对应 Server 的规则释放，不能把 RID 当 Resource 或 Ref。读 Server 问题时，先问这个 RID 是哪个 Server 创建的，谁持有，什么时候 free，是否跨线程延迟执行。"
  ],
  "从场景节点到 Server 后端": [
    "这一节说明高级节点如何把用户语义交给底层系统。场景节点保存的是游戏开发者能理解的状态，比如位置、材质、碰撞形状、音量、文本内容；真正执行时，它会通过 Server API 创建或更新 RID，让后端保存更适合计算和提交的内部对象。",
    "小白可以用餐厅点单理解：Node 像服务员手上的菜单和桌号，Server 像后厨系统，RID 是订单号。服务员不直接做菜，也不直接碰厨房设备；它把点单和修改通过订单号交给后厨。这样 scene 代码保持用户友好，后端可以换实现、做线程隔离、批量提交、延迟刷新。"
  ],
  "主要 Server 的职责边界": [
    "这一节帮你区分各个 Server 管什么。RenderingServer 管可见对象、材质、纹理、canvas、viewport 和绘制提交；PhysicsServer 管 body、area、shape、space、joint 和物理步；AudioServer 管总线、混音和播放；DisplayServer 管窗口、屏幕、输入法、剪贴板；TextServer 管字体 shaping 和 glyph。",
    "小白不要把 Server 看成一个万能黑盒。每个 Server 都是一类底层能力的统一入口，它们和 scene 节点之间通过 RID、回调或资源数据连接。你遇到“节点设置了属性但底层没变化”时，要先判断它应该落到哪个 Server，再看场景节点有没有把状态同步给对应 Server。"
  ],
  "读 Server 源码的三个固定问题": [
    "这一节给 Server 源码的固定读法：第一，API 契约在哪里定义，通常先看 servers/*_server.h；第二，真实后端在哪里，可能在 renderer_rd、godot_physics、jolt、text_server_adv、platform DisplayServer；第三，调用是立即执行还是被 wrap_mt、command queue、sync/draw/step 延迟。",
    "小白读 Server 时不要一头扎进后端算法。先看抽象接口能做什么，再找谁实现这个接口，最后看每帧在哪个阶段刷新。这样你会知道 RenderingServer::draw、PhysicsServer::step、DisplayServer::process_events 不是普通工具函数，而是把前面积累的状态真正推进到底层系统的时间点。"
  ],
  "渲染链路图": [
    "这一节用图展示渲染从节点属性到 GPU 命令的大方向。场景节点如 Sprite2D、Control、MeshInstance3D 先保存用户可理解的属性；CanvasItem 或 VisualInstance3D 把它们转换成 RenderingServer 能理解的 RID 和状态；Renderer 后端再做可见性、排序、材质、光照、pass 组织，最后通过 RenderingDevice 和平台窗口输出。",
    "小白要记住，节点不是直接向屏幕画东西。节点更像是在描述“我要显示什么、在哪里、用什么材质”；RenderingServer 和 Renderer 才负责把这些描述变成渲染命令。这样做能让 2D、3D、GUI、编辑器视图共享一套底层渲染管线，也能让后端实现和场景层解耦。"
  ],
  "2D 路径：CanvasItem 把绘制命令交给 Server": [
    "这一节讲 2D 和 GUI 常走的 CanvasItem 路径。Node2D、Control、Sprite2D 等都站在 CanvasItem 体系上，CanvasItem 持有 canvas item RID，负责把 draw 命令、纹理、材质、可见性、变换、z index 等状态交给 RenderingServer。",
    "小白可以把 CanvasItem 想成一张绘图订单。Sprite2D 不会自己把像素刷到屏幕上，它会告诉 CanvasItem 和 RenderingServer：我要在这个位置画这张纹理，使用这个材质和排序。真正的绘制会在渲染阶段统一处理，所以修改 2D 属性时常能看到 queue_redraw、notification、RID 更新和 RenderingServer 调用。"
  ],
  "3D 路径：VisualInstance3D 只持有实例语义": [
    "这一节讲 3D 节点到渲染后端的分工。Mesh、Material、Texture 是资源；MeshInstance3D 是场景里的实例；VisualInstance3D 负责让这个实例进入 3D 世界和 scenario；RenderingServer 维护 instance RID，后端再处理可见性、光照、阴影和 draw pass。",
    "小白要特别分清“资源”和“实例”。同一个 Mesh Resource 可以被多个 MeshInstance3D 使用，每个实例有自己的 Transform、可见层、材质覆盖和场景信息。读 3D 渲染源码时，如果只看 Mesh 数据，你还没看到它什么时候进入世界；如果只看 Node3D 变换，你也还没看到它如何同步到 RenderingServer。"
  ],
  "渲染链路和源码入口": [
    "这一节把渲染路径落到源码入口。2D 先看 CanvasItem 和 RenderingServer canvas API，3D 先看 VisualInstance3D、GeometryInstance3D、RenderingServer instance API，再进入 renderer_rd、RendererSceneRD、RendererCanvas、RenderingDevice。窗口输出相关问题再看 DisplayServer 和 platform。",
    "小白读渲染时最容易直接跳到 Vulkan 或 RD 后端，然后被大量 GPU 细节淹没。更稳的路线是先从用户节点找到它提交给 RenderingServer 的调用，确认状态如何变成 RID，再进入 renderer 后端看这些 RID 如何被组织成可见列表、材质和命令。先看“状态怎么提交”，再看“GPU 怎么执行”。"
  ],
  "几个阅读抓手": [
    "这一节列的是读渲染源码的抓手。看到 queue_redraw、notification、RID、material、scenario、viewport、RenderingServer::sync/draw，要把它们放到一条时间线上。属性改变通常只是标记或更新状态，真正绘制往往等到渲染阶段统一发生。",
    "小白可以用三个问题读渲染：这个对象有没有 RID；它的可见性、变换、材质什么时候同步给 Server；最后在哪个 draw/sync 阶段进入后端。只要这三点清楚，复杂的 2D/3D/GPU 后端就不会完全失去方向。"
  ],
  "资源、Shader 和 RD 的边界": [
    "这一节讲渲染资源、Shader 和 RenderingDevice 的边界。Texture、Mesh、Material、Shader 是用户可见或资源系统管理的数据；RenderingServer 把它们转成后端能使用的 RID；RenderingDevice 更接近底层 GPU 抽象，负责 buffer、texture、pipeline、command list 等低层操作。",
    "小白不要把 Shader 代码、Material 参数、GPU pipeline 混成一件事。Shader Resource 描述用户写的着色逻辑，Material 保存参数和状态，Renderer 把它们编译或组合成后端需要的形式，RD 再负责和图形 API 交互。调试渲染问题时，先判断问题在资源数据、Server 状态、Renderer 组织，还是 RD/GPU 提交。"
  ],
  "物理源码的分界": [
    "这一节帮你分清物理的几个层次。scene 里的 RigidBody、CharacterBody、CollisionShape 是用户节点；PhysicsServer2D/3D 是抽象 API；Godot Physics、Jolt 或 GDExtension 后端是真正求解碰撞和约束的实现；Main/SceneTree 的固定物理步决定什么时候同步和回调。",
    "小白不要以为 RigidBody3D 就是物理引擎本体。它更像一个场景层代理，保存用户属性并把 body、shape、space 等交给 PhysicsServer。真正的广义碰撞、窄相、积分、约束求解在后端。读物理 bug 时先区分节点状态、RID 状态、后端求解状态和回调时机。"
  ],
  "从 RigidBody3D + CollisionShape3D 到一次物理求解": [
    "这一节用 RigidBody3D 和 CollisionShape3D 串起物理路径。节点创建后会为 body 和 shape 创建 RID，设置质量、模式、层、mask、transform、shape 数据等；进入 SceneTree 和物理世界后，这些 RID 会被放入 space；固定物理步时 PhysicsServer 调后端 step，后端计算碰撞和运动结果，再把状态回传给场景层。",
    "小白要看到两次同步：场景层把用户设置同步到物理后端，物理后端在 step 后把结果同步回场景层。中间还有 query、flush、监视区域、信号、直接状态访问等安全边界。这样你才能解释为什么有些物理属性要在 physics_process 里改，为什么查询要注意当前物理步，为什么直接改 transform 可能和物理后端状态打架。"
  ],
  "物理概念到源码对象的对应关系": [
    "这一节把物理术语翻译成源码对象。Body 是会参与运动和碰撞的物体，Area 是检测区域和环境影响，Shape 是几何形状，Space 是同一物理世界，Joint 是约束，Query 是查询接口，RID 是这些后端对象的句柄。",
    "小白读物理源码时，先把游戏里看到的节点名翻译成这些后端概念。CollisionShape3D 本身不是碰撞体，它提供 Shape 给 Body 或 Area；World3D 里有 space；PhysicsDirectSpaceState 做查询；PhysicsServer 管 RID。翻译准确以后，读 scene/3d 和 servers/physics_3d 的关系就会清楚很多。"
  ],
  "固定物理步的真实顺序": [
    "这一节强调物理按固定 tick 推进，不是跟显示帧完全同步。Main::iteration 会根据时间累计决定跑几次物理步；每个物理步中要先同步必要状态，执行 SceneTree 的 physics_process，再让 PhysicsServer step，期间还要处理查询、消息队列、导航和删除队列的安全时机。",
    "小白可以把物理步想成固定节拍的钟。显示帧可能快慢不一，但物理希望按稳定间隔计算，才能让碰撞和运动稳定。理解顺序以后，你会知道输入、角色移动、物理查询、信号、queue_free 放在哪个阶段更合理，也能解释为什么在 process 里做物理改动有时会出现时序问题。"
  ],
  "查询、CharacterBody 和 RigidBody 的边界": [
    "这一节区分几种常见物理用法。查询是问当前物理世界某条射线、形状或点会碰到什么；CharacterBody 偏向脚本控制的角色移动，通过 move_and_slide 等 API 与世界交互；RigidBody 偏向由物理后端根据力、碰撞和约束自动求解。",
    "小白不要把 CharacterBody 当成普通 RigidBody 的简化版。CharacterBody 的核心是用户代码主动给速度和移动请求，物理系统帮助它滑动、碰撞、贴地；RigidBody 则更强调物理模拟本身。读问题时先问：这个对象是主动控制，还是交给物理世界模拟，还是只是做查询拿结果。"
  ],
  "后端选择：Godot Physics、Jolt、Extension、WrapMT": [
    "这一节讲物理后端为什么可以替换。PhysicsServer 定义统一接口，Godot Physics、Jolt 或扩展后端可以实现这些接口；WrapMT 又可能把调用包装到多线程边界中。scene 层不应该知道具体后端内部的数据结构，只通过 PhysicsServer 和 RID 操作。",
    "小白可以把它想成同一套方向盘可以接不同发动机。方向盘的接口不能乱变，否则所有车都要改；发动机可以换，但必须遵守接口。读后端问题时先看当前项目选择了哪个 physics server，再看调用是否被多线程包装延迟，最后进入具体后端实现。"
  ],
  "导航不是物理的一部分": [
    "这一节提醒你不要把导航和物理混为一谈。物理负责碰撞、运动、力、约束和空间查询；导航负责导航网格、路径搜索、避障、agent、region 和 map。它们都和空间有关，也都可能在固定步附近更新，但源码系统和数据结构不同。",
    "小白可以这样区分：物理回答“我撞到了什么、能不能移动过去”，导航回答“从 A 到 B 应该走哪条路”。导航可能参考场景几何或物理障碍，但它不是 PhysicsServer 的子功能。读寻路、导航网格烘焙、agent 避障时，应进入 NavigationServer 和 scene/navigation 相关路径。"
  ],
  "读物理/导航源码的常见误区": [
    "这一节列出物理和导航最常见的错误理解：把节点当后端本体，把 CollisionShape 当 body，把导航当物理，把显示帧当物理步，把 RID 当自动释放资源，把后端选择忽略掉。每一个误区都会让你看错源码位置。",
    "小白排查时先分层：scene 节点负责用户 API 和状态，Server 负责抽象接口，后端负责算法，Main/SceneTree 负责时机。物理和导航问题看起来都发生在游戏世界里，但一个是碰撞求解，一个是路径规划，读源码时必须走不同入口。"
  ],
  "六个系统的职责边界": [
    "这一节把音频、输入、GUI、动画、Tween、调试性能放在一起，是因为它们都直接影响用户体验，但内部路径不同。音频走 AudioServer 和播放资源；输入从平台事件进入 Input/Viewport/Control/脚本；GUI 由 Control、Theme、TextServer、RenderingServer 协作；动画和 Tween 都改属性但驱动方式不同；性能采样由运行时收集，编辑器展示。",
    "小白不要把它们都理解成 scene 的普通节点逻辑。它们有的接平台，有的接 Server，有的接 Object 属性系统，有的接编辑器工具。读这一组系统时先问：事件从哪里来，状态存在哪里，谁每帧推进，最后是否交给 Server 或平台。"
  ],
  "音频：从 play() 到扬声器": [
    "这一节讲音频播放路径。AudioStreamPlayer 调 play 以后，场景节点保存播放状态和资源引用，创建 playback，声音数据最终进入 AudioServer 的混音系统，再由音频 driver 或平台后端输出到设备。节点本身不是扬声器，也不是混音器。",
    "小白可以把 AudioStream 看成音频素材，AudioStreamPlayer 看成场景里的播放器，AudioServer 看成混音台，平台音频后端看成真正接到扬声器的线。调试没声音时，要分别检查资源是否加载、player 是否在树里和播放、bus 和音量是否正确、AudioServer 是否混音、平台设备是否输出。"
  ],
  "输入：平台事件怎样走到 GUI 和脚本": [
    "这一节讲输入从操作系统进入 Godot 的路径。平台窗口收到键盘、鼠标、触摸、手柄、输入法等事件后，DisplayServer 或平台层把它转成 Godot 的 InputEvent；Input 记录全局状态，Viewport 分发事件，Control 可以处理 GUI 输入，脚本也能通过 _input、_unhandled_input 等回调接收。",
    "小白要把“按键状态”和“事件分发”分开。Input.is_action_pressed 查的是当前状态和 InputMap 映射；_input 收到的是事件流；GUI 控件可能先消费事件，导致后续 unhandled 输入不再触发。读输入 bug 时，要看平台事件、InputMap、Viewport 分发、Control 处理和脚本回调的顺序。"
  ],
  "GUI：Control 布局、Theme 和 TextServer": [
    "这一节讲 GUI 控件并不只是画矩形。Control 负责锚点、大小、鼠标过滤、焦点、布局通知；Container 负责对子控件排版；Theme 提供颜色、字体、样式盒和图标；TextServer 负责文字 shaping、换行、字形；RenderingServer 最终绘制。",
    "小白可以把 GUI 看成多人合作：Control 确定控件在界面中的位置和交互，Container 负责摆放，Theme 负责长相，TextServer 负责文字能正确显示，RenderingServer 负责画出来。排查按钮位置、字体、主题、点击区域、文字不显示时，要判断是哪一层的问题。"
  ],
  "动画与 Tween：两套“时间驱动属性变化”机制": [
    "这一节区分 AnimationPlayer/AnimationTree 和 Tween。它们都能让属性随时间变化，但设计目标不同：AnimationPlayer 播放资源化的轨道和关键帧，适合可编辑、可保存、可复用的动画；Tween 更像运行时临时创建的过渡任务，适合代码里做 UI 动效、移动、淡入淡出。",
    "小白不要因为它们都改属性就混用理解。Animation 关注轨道、关键帧、插值、调用方法、音频轨、混合和状态机；Tween 关注从当前值到目标值的短期插值和回调。读源码时先问这个时间变化是资源驱动，还是运行时代码临时驱动。"
  ],
  "AnimationPlayer / AnimationTree": [
    "这一小节讲资源化动画。AnimationPlayer 持有 Animation 资源并按轨道更新目标对象属性、方法调用、音频等；AnimationTree 在此基础上做混合、状态机、BlendTree，让多个动画按权重和状态组合起来。它们适合角色动作、复杂 UI 状态或编辑器里可视化编辑的动画。",
    "小白可以把 AnimationPlayer 想成按时间表执行的播放器，AnimationTree 像把多个播放器接到混音台。调试动画时，要看目标 NodePath 是否正确、轨道类型是否匹配、属性是否可写、播放状态是否正确、混合树是否把结果权重传到最终输出。"
  ],
  "Tween / SceneTree": [
    "这一小节讲 Tween。Tween 通常由 SceneTree 或节点创建，描述某个属性、方法或回调在一段时间内从一个值过渡到另一个值。它更轻量，更适合运行时临时效果，比如按钮 hover、窗口淡入、相机移动、小物体漂浮。",
    "小白要注意 Tween 的生命周期和宿主关系。Tween 不像 Animation 资源那样主要靠编辑器保存，它常常是代码创建、运行中推进、结束后清理。读 Tween 问题时，看它绑定到哪个节点或 SceneTree，是否暂停，是否被 kill，目标对象是否还存在，以及每帧由哪个阶段推进。"
  ],
  "调试和性能：运行时采样，编辑器呈现": [
    "这一节说明调试和性能面板不是凭空得出数据。运行时系统会采样帧时间、对象数量、内存、渲染、物理、脚本等指标，Performance、debugger、profiler 等把数据收集起来，编辑器只是把这些运行时数据通过 UI 面板展示出来。",
    "小白可以把它分成两端：游戏进程或运行时负责产生数据，编辑器负责请求、接收、整理和展示。调试性能问题时，不要只看编辑器面板代码，也要回到 Performance 指标、Server 统计、调试协议和采样点，确认数据在哪里产生、什么时候更新、是否只在 debug/editor 构建里可用。"
  ],
  "读这一组系统的固定问题": [
    "这一节给音频、输入、GUI、动画、调试性能的共同读法。先问事件或时间从哪里来：平台输入、AudioServer 混音 tick、SceneTree 帧推进、AnimationPlayer 播放时间、调试采样时机。再问状态存在哪里：节点、Resource、Server、Theme、InputMap、Performance。",
    "小白只要坚持这两个问题，就不会把用户体验层看成一团 UI 代码。比如输入问题看平台事件和 Viewport 分发；GUI 问题看 Control 布局和 Theme；动画问题看属性轨道和目标对象；音频问题看资源播放和混音；性能问题看采样源和编辑器展示。"
  ],
  "核心抽象：ScriptServer、Script、ScriptInstance": [
    "这一节讲脚本系统的三层抽象。ScriptServer 管理已注册的脚本语言；Script 是一个脚本资源，例如 .gd 或 C# 脚本文件；ScriptInstance 是脚本挂到某个 Object 上以后生成的运行实例，负责保存脚本变量、方法和与宿主对象的连接。",
    "小白可以把 Script 当剧本，ScriptInstance 当演员正在某个对象身上演这份剧本，ScriptServer 当剧院管理处，知道有哪些语言和剧本系统可用。这样你就能解释为什么同一个 GDScript 资源可以挂到多个节点，每个节点有自己的脚本实例和变量状态。"
  ],
  "GDScript：从 .gd 到字节码函数": [
    "这一节讲 GDScript 从文件到可执行代码的路径。.gd 文件先作为 Resource 被加载，经过 tokenizer/parser 建 AST，分析类型和作用域，再编译成函数、常量、字节码或内部表示，运行时由 GDScript VM 或相关执行路径调用。",
    "小白不要把 .gd 文件理解成每次运行都逐行解释文本。Godot 会把脚本文本变成更适合运行的结构，错误检查、类型推断、补全、调试信息也在这个过程中建立。读 GDScript 问题时，要区分解析错误、编译错误、运行时调用错误和与 Object/ClassDB 交互的问题。"
  ],
  "Object 怎样把调用转给脚本": [
    "这一节解释 Object 和脚本实例如何协作。一个 Object 可以挂 Script，脚本实例会接管或补充方法、属性、通知等行为。当外部调用方法或发通知时，Object 先查 C++ 绑定和脚本实例，再把能由脚本处理的部分转给 ScriptInstance。",
    "小白可以把 C++ Object 想成基础机体，ScriptInstance 像装在它身上的控制程序。机体提供统一身份、信号、属性、生命周期；脚本提供用户写的行为。读脚本回调不触发时，要看脚本是否挂上、实例是否创建、方法名是否正确、Object 生命周期是否已经结束。"
  ],
  "C# / Mono 模块：同一个 ScriptLanguage，另一套运行时": [
    "这一节讲 C# 不是绕过 Godot 对象系统，而是通过 Mono 模块实现另一种 ScriptLanguage。它有自己的编译、程序集加载、托管对象生命周期和 C# 绑定层，但仍要和 Object、ClassDB、Variant、Script、ScriptInstance 这些 Godot 核心抽象对接。",
    "小白可以把 GDScript 和 C# 看成两种语言接到同一个舞台。语言内部运行时不同，但它们都要能挂到 Object 上、调用 Godot API、响应信号、暴露属性、参与编辑器和调试。读 C# 问题时要多看一层托管/原生桥，不要只按 GDScript 的路径找。"
  ],
  "GDExtension：外部动态库怎样加入对象系统": [
    "这一节讲 GDExtension。它允许外部动态库在不重新编译引擎的情况下注册类、方法、属性、资源和回调，接入 Godot 的 Object/ClassDB/Variant 世界。扩展通过一套 C ABI 和初始化函数告诉引擎自己提供哪些能力。",
    "小白可以把 GDExtension 想成外部插件带着身份证明进场。它不是随便把 C++ 指针塞进 Godot，而是要按 Godot 规定的接口注册类型和方法，使用 Variant、StringName、ObjectID、Ref、RID 等共同语言。读扩展问题时，要看初始化级别、类注册、方法绑定、生命周期和 ABI 是否匹配。"
  ],
  "ClassDB、Variant、MethodBind 是脚本和扩展的共同语言": [
    "这一节强调脚本语言和扩展虽然实现不同，但它们都要经过同一套运行时桥梁。ClassDB 告诉外部有哪些类、方法和属性；MethodBind 负责把名字调用变成 C++ 函数调用；Variant 负责在不同语言和系统之间传递值。",
    "小白理解这点后，就不会把 GDScript、C#、GDExtension 当成完全独立世界。它们的语法、编译器、运行时可以不同，但要调用 Godot API，就必须说 Object/ClassDB/Variant 这套共同语言。很多跨语言 bug，本质上是绑定签名、Variant 转换、对象生命周期或线程边界的问题。"
  ],
  "读脚本和扩展源码的常见误区": [
    "这一节列出脚本和扩展的常见误区：以为脚本直接调用任意 C++ 函数，忽略 _bind_methods；以为 .gd 文本运行时逐行解释，忽略编译和缓存；以为 C# 路径和 GDScript 完全一样，忽略托管运行时；以为 GDExtension 只要能加载动态库就安全，忽略 ABI 和初始化级别。",
    "小白读这类源码要先找共同桥梁，再进入语言特有实现。共同桥梁是 ScriptLanguage、Script、ScriptInstance、ClassDB、Variant、MethodBind；语言特有部分才是 parser、compiler、VM、Mono、extension ABI。顺序反了，就会在某个语言内部细节里迷路，看不到它怎样接回 Godot 对象系统。"
  ],
  "先建立可用心智模型": [
    "这一节讲编辑器的总模型：Godot 编辑器不是外部程序套壳，而是一个用 Godot 自己的场景树、GUI、资源系统、反射系统搭出来的大型工具应用。EditorNode 是主控节点，Dock、Inspector、FileSystem、SceneTree 面板、导入导出、插件系统都挂在这套工具场景里。",
    "小白理解编辑器时要把它分成两层：它一方面是普通 Godot 程序，复用 Control、SceneTree、ResourceLoader、ClassDB；另一方面又是工具层，负责编辑项目、保存场景、导入资源、启动游戏进程。读 editor 目录时，不要忘记它大量调用运行时 API，而不是重新实现一套引擎。"
  ],
  "编辑器启动：从命令行到可见工具界面": [
    "这一节讲编辑器如何启动。前面的 platform、Main::setup、setup2 和普通运行时一样，差异主要在 Main::start 选择编辑器模式后，会创建 EditorNode、加载编辑器设置、项目数据、主题、插件、Dock 和主界面。也就是说，编辑器界面出现之前，core、servers、scene、editor 类型已经按顺序注册。",
    "小白不要把编辑器启动看成打开一个 UI 窗口这么简单。它要先有 Object/ClassDB、ResourceLoader、DisplayServer、SceneTree、Control、Theme、EditorPlugin 等基础，才能显示工具界面。编辑器启动问题常常不是单个按钮问题，而是运行模式、TOOLS_ENABLED、项目路径、资源加载或插件初始化顺序的问题。"
  ],
  "打开、切换、保存场景：编辑器怎样操作 PackedScene": [
    "这一节讲编辑器对场景的操作。打开场景时，编辑器通过 ResourceLoader 得到 PackedScene，再实例化成可编辑的 Node 树；切换场景时，编辑器维护当前编辑上下文；保存时，PackedScene.pack 根据 owner、节点属性、资源引用、信号连接、子场景实例等生成 SceneState 并写回 .tscn 或 .scn。",
    "小白要把编辑器里看到的场景树和磁盘上的 .tscn 分开。你在编辑器里改的是运行时 Node 对象和资源状态，保存才会把可持久化部分打包回 PackedScene。节点在树里但 owner 不对，可能不会保存；资源引用、信号、脚本、子场景实例也都有各自的序列化规则。"
  ],
  "Inspector：为什么 C++、脚本和扩展属性都能被编辑": [
    "这一节讲 Inspector 的核心不是为每个类手写 UI，而是查询 Object 的属性列表和 ClassDB/脚本/扩展提供的 PropertyInfo，然后按类型生成 EditorProperty 控件。C++ 属性、脚本导出变量、GDExtension 属性只要正确暴露，就能走同一套编辑流程。",
    "小白可以把 Inspector 想成万能表单生成器。它先问对象“你有哪些可编辑属性”，再根据类型生成输入框、勾选框、资源选择器、颜色选择器等控件。用户改值时，还会涉及 Object::set、UndoRedo、通知、场景 dirty 标记和保存。属性不显示时，应该先查暴露信息，而不是先改 Inspector UI。"
  ],
  "EditorPlugin：内置工具和用户插件共享同一套扩展口": [
    "这一节讲 EditorPlugin。Godot 的很多编辑器工具和用户插件都通过相似的扩展口把菜单、Dock、Inspector 插件、Gizmo、导入器、编辑工具挂进 EditorNode。它是编辑器扩展能力的统一入口。",
    "小白可以把 EditorPlugin 理解成编辑器的插槽系统。插件不是随便改主界面，而是向 EditorNode 注册自己要加的菜单、面板、处理逻辑和生命周期回调。读内置工具时也可以按插件思路看：它什么时候注册，挂到哪里，响应什么对象，退出时如何清理。"
  ],
  "文件系统、导入、导出：资源管线在编辑器里如何闭环": [
    "这一节讲编辑器资源管线。FileSystemDock 和 EditorFileSystem 扫描项目文件；导入系统把外部素材转换成 Godot 更适合运行的资源和 .import 产物；ResourceLoader 在运行时加载这些资源；导出系统再把项目资源、平台模板和配置打包成最终应用。",
    "小白要看到这是闭环：文件出现 -> 编辑器识别 -> 导入器转换 -> 资源被场景引用 -> 运行时加载 -> 导出时打包。资源显示不对、导入缓存旧、导出缺文件时，不要只看 ResourceLoader，也要看 editor/import、.godot/imported、EditorFileSystem、ExportPlatform 和平台打包规则。"
  ],
  "资源导入管线：从源文件到运行时资源": [
    "这一节把导入管线拆成源文件、.import 元数据、内部资源、UID 和导出包几层。编辑器扫描项目时决定是否需要重新导入；ResourceImporter 生成运行时更适合加载的资源；ResourceFormatImporter 让 ResourceLoader 能从原始路径转到导入产物。",
    "小白调试资源问题时要问清楚：当前看到的是原始素材，还是导入后的内部资源；路径是普通 res://，还是 uid://；问题发生在编辑器扫描、重导入、运行时加载，还是导出打包。这样才能避免只改原文件却忘了 .import 或导出 preset。"
  ],
  "运行和调试：编辑器并不“在自己里面运行游戏”": [
    "这一节纠正一个常见误解。点击运行时，编辑器通常会启动一个新的 Godot 游戏进程，并把项目路径、场景、调试参数等传过去。编辑器负责保存、构建、启动、连接调试器和展示输出；游戏逻辑在另一个进程里按普通 Main/SceneTree 路径运行。",
    "小白理解这一点后，就能解释为什么编辑器崩溃和游戏崩溃可能是两个进程，为什么调试器需要通信协议，为什么运行按钮源码在 editor/run，而主场景加载仍在 main 和 ResourceLoader。调试运行问题时，要分清编辑器侧准备过程和游戏进程启动过程。"
  ],
  "读编辑器源码的常见误区": [
    "这一节列出读 editor 目录的坑：以为编辑器完全独立于运行时，忽略它复用 scene/gui 和 ClassDB；以为 Inspector 手写每个属性，忽略反射；以为运行按钮直接在编辑器里跑游戏，忽略新进程；以为 editor 代码能进导出模板，忽略 TOOLS_ENABLED。",
    "小白读编辑器源码要一直问三个问题：这段代码是在操作运行时对象，还是编辑器 UI，还是资源/项目文件；它是否只在 TOOLS_ENABLED 存在；它最终调用的是编辑器自己的逻辑，还是回到 core/scene/servers 的运行时 API。这样才能避免把工具层和运行时层混在一起。"
  ],
  "读平台层的正确入口：先找抽象，再找实现": [
    "这一节讲平台代码的读法。不要一上来就读 Windows API、Android Java 或 Web JS glue，而是先找 Godot 抽象：OS、DisplayServer、FileAccess、Input、AudioDriver、Thread、Semaphore 等。抽象告诉你 Godot 需要什么能力，平台实现告诉你这个能力如何落到具体系统。",
    "小白可以把平台层想成翻译官。Godot 内部说的是统一接口，不同操作系统说不同语言，platform 代码负责翻译。先看抽象能避免被某个平台的细节带偏，也能判断某个行为是 Godot 通用语义，还是 Windows、Web、Android 特有实现。"
  ],
  "桌面平台启动链：Windows 是最直观样本": [
    "这一节用 Windows 展示桌面平台启动链。Windows 入口处理宽字符命令行、创建 OS_Windows、调用 Main::setup 和 Main::start，然后 OS_Windows::run 进入消息循环，处理窗口事件并反复调用 Main::iteration。Linux 和 macOS 形状类似，但系统 API 和事件循环细节不同。",
    "小白读 Windows 样本的价值在于它接近传统桌面程序模型：进程入口、命令行、窗口消息、主循环都比较直观。读懂它以后，再看 macOS 的应用生命周期、Linux 的窗口后端、Web 的浏览器回调、Android 的 Activity/JNI，就能区分共同骨架和平台特殊 glue。"
  ],
  "移动端和 Web：主循环由宿主平台驱动": [
    "这一节讲移动端和 Web 与桌面不同。桌面程序通常自己控制 while 循环；移动端和浏览器环境往往由宿主平台控制生命周期和帧回调。Godot 需要把自己的 MainLoop 接到 Activity、UIKit、浏览器 requestAnimationFrame 或平台回调里。",
    "小白可以理解为：桌面像你自己开车，移动和 Web 像坐在平台安排的轨道车上，你只能在平台给你的时间点推进一帧。读这些平台时，要特别关注暂停/恢复、窗口重建、触摸输入、音频焦点、Web canvas、JS bridge 和资源加载限制。"
  ],
  "DisplayServer：窗口和输入的集中边界": [
    "这一节讲 DisplayServer。它统一处理窗口、屏幕、鼠标、键盘、输入法、剪贴板、显示模式、窗口事件等能力。scene 和 GUI 不应该直接调用某个平台的窗口 API，而是通过 DisplayServer 和 InputEvent 这套抽象接收和处理。",
    "小白可以把 DisplayServer 想成 Godot 和桌面/移动/浏览器窗口系统之间的总前台。窗口大小变化、鼠标移动、键盘输入、IME 输入、屏幕列表、DPI 等都从这里进入统一世界。GUI 和脚本看到的是 Godot 的事件和窗口对象，而不是每个平台原生事件。"
  ],
  "驱动、第三方、模块：三者不要混在一起读": [
    "这一节帮你区分 drivers、thirdparty、modules。thirdparty 是外部库源码；drivers 是 Godot 对某些底层能力或格式的封装和 glue；modules 是可选功能接入点，可能封装 thirdparty，也可能注册脚本语言、物理后端、导入器或编辑器工具。",
    "小白读外部库相关功能时，不要直接从 thirdparty 开始。先找 Godot 的封装层，看看它如何把外部库能力接进 ImageLoader、AudioStream、ResourceImporter、Server 或编辑器。只有定位到外部库 bug 或升级依赖时，才深入 thirdparty 细节。"
  ],
  "平台代码最容易踩的判断点": [
    "这一节列平台代码常见坑：把某个平台行为当成所有平台通用；在 core/scene 里直接引用平台实现；忽略 TOOLS_ENABLED 和导出模板差异；忘记移动端和 Web 生命周期由宿主控制；忽略动态库、路径、权限、输入法、DPI、线程 API 差异。",
    "小白改平台代码前要先回答：这个行为应该是 Godot 抽象的一部分，还是某个平台特殊适配；是否需要在 OS 或 DisplayServer 接口加能力；其他平台能否合理实现；导出模板和编辑器是否都需要。平台层改动很容易牵动多平台，所以边界判断比单个平台能跑更重要。"
  ],
  "读一个模块时先看这四个文件": [
    "这一节给模块阅读入口。先看 config.py 了解模块是否默认启用、依赖什么；看 SCsub 了解编译哪些源码和 thirdparty；看 register_types.h/cpp 了解在哪些初始化 level 注册什么；再看模块的资源、Server、编辑器或导入器实现。这样读模块不会迷路。",
    "小白不要把 modules 目录当成一堆同类功能。GDScript、Jolt、glTF、WebSocket、TextServer、OpenXR 的职责完全不同，但它们都要通过构建和注册接入引擎。四个文件能先回答“是否编译、何时初始化、注册什么、依赖什么”，再决定进入具体实现。"
  ],
  "模块不是固定目录层，而是按初始化级别接入": [
    "这一节再次强调 modules 不是 core/scene/servers/editor 之外的第五层。一个模块可以在 CORE 注册基础资源或脚本语言，在 SERVERS 注册后端，在 SCENE 注册节点和资源，在 EDITOR 注册插件、导入器或面板。关键要看 initialize_<module>_module(p_level) 对不同 level 做了什么。",
    "小白可以把模块想成插卡式扩展，插到哪一层取决于它需要接入什么能力。比如脚本语言可能很早就要接入 core，对应后端可能在 servers，用户可见节点在 scene，工具 UI 在 editor。读模块时按 level 切分，比按文件夹名猜层级可靠得多。"
  ],
  "所有权矩阵：先分清谁真的拥有谁": [
    "这一节讲所有权。Godot 里 Object、Node、Resource、RefCounted、RID、Callable、ObjectID 的生命周期规则不同。ObjectDB 只登记不拥有；Node 的树关系管理结构和析构清理，但不是 Ref；Resource 通常由 Ref 引用计数管理；RID 由对应 Server 释放；Callable 目标需要调用时重新验证。",
    "小白读崩溃、泄漏、重复释放、对象失效问题时，第一步就是画所有权矩阵。谁创建对象，谁保存引用，谁负责释放，释放后是否还有 ObjectID、Callable、RID 或裸指针留着。很多底层 bug 不是算法错，而是对象还没创建、已经释放、或释放顺序和依赖顺序不一致。"
  ],
  "RID 的设计：轻量句柄和 Server 内部对象分离": [
    "这一节从设计角度解释 RID。RID 把场景层和 Server 内部对象分开，让场景层不需要知道渲染、物理、音频等后端对象的真实类型，也避免直接持有后端指针。Server 可以用 RID_Owner 管理内部对象、做验证、延迟释放和线程隔离。",
    "小白要明白 RID 简单但不自动。它像订单号，不像智能指针。订单号能让你找后厨状态，但不会自动让后厨把菜退掉；必须调用对应 Server 的 free 或专用释放接口。把 RID 混用到错误 Server，或者忘记释放，都会造成难查的问题。"
  ],
  "注册顺序的意义：类型必须在被使用前出现": [
    "这一节讲注册顺序为什么重要。ClassDB、Server 单例、ResourceFormatLoader、ScriptLanguage、EditorPlugin、模块后端都必须在被使用前注册。Main::setup、setup2、start 和 cleanup 的顺序，本质上是在保证依赖先建立，使用者后出现，关闭时反过来释放。",
    "小白可以把注册看成开店前挂招牌和录入系统。没有注册，脚本查不到类，ResourceLoader 找不到格式，Inspector 看不到属性，Server 后端无法创建。模块 level 选错，可能在需要 Node 时 scene 还没注册，或者在导出模板里引用了 editor 类型。"
  ],
  "ObjectDB：对象索引，不是所有权系统": [
    "这一节讲 ObjectDB。它给 Object 分配和登记 ObjectID，让系统可以通过 ID 查回对象，常用于 Callable、信号、延迟调用、调试器等场景。但 ObjectDB 不拥有对象，不会因为你保存了 ObjectID 就阻止对象被释放。",
    "小白可以把 ObjectDB 想成通讯录，不是保险箱。通讯录里有号码，说明曾经或当前能查到这个对象；对象释放后，号码就不能当活对象使用。调用前要重新查并验证，不能把 ObjectID 当强引用，更不能保存裸指针后假设 ObjectDB 会保护它。"
  ],
  "memnew / memdelete：Godot 自己的分配钩子": [
    "这一节讲 Godot 为什么不用普通 new/delete 写到底。memnew/memdelete 接入了 Godot 自己的内存统计、调试检查、Object post-initialize、pre-delete 等机制。对 Object 来说，创建和释放不仅是分配内存，还要进入/离开 ObjectDB、触发生命周期钩子。",
    "小白可以把 memnew/memdelete 理解成带登记流程的创建和销毁。普通 C++ new 只是拿内存和构造对象；Godot 的宏还要让引擎知道这个对象存在、方便调试和统计。读对象生命周期时，看到 memnew、memdelete、queue_free、RefCounted、RID free，要马上意识到它们代表不同释放通道。"
  ],
  "延迟调用和删除队列：为什么不能随手立即改树": [
    "这一节讲为什么 Godot 经常延迟执行。场景树、信号连接、物理同步、GUI 事件分发都可能正在遍历数据结构，如果在遍历中立刻删除节点、改父子关系或调用复杂逻辑，就可能破坏当前迭代状态。call_deferred、set_deferred、queue_free、MessageQueue 和 SceneTree 删除队列就是为了解决这个问题。",
    "小白可以把它想成会议中不能随便把正在发言的人从名单里删掉，而是先记到会后处理清单。延迟不代表不执行，而是等到安全点统一执行。读时序问题时，要看这个操作是立即生效，还是进了 MessageQueue/删除队列，以及它在哪个帧阶段 flush。"
  ],
  "线程边界：为什么有 WorkerThreadPool、process thread group 和 wrap_mt": [
    "这一节讲多线程边界。Godot 有主线程、工作线程池、Server 后端线程、process thread group 和各种 wrap_mt 包装。它们的目标是让耗时任务并行，同时避免任意线程随便改 Object、SceneTree 或后端状态造成竞态。",
    "小白要先记住：不是所有 API 都能在任意线程调用。WorkerThreadPool 适合并行任务，process thread group 控制节点处理分组，wrap_mt 可能把 Server 调用排队到安全线程执行。读线程问题时，先问当前代码在哪个线程，能否访问这个对象，调用是立即执行还是被排队。"
  ],
  "ResourceCache：为什么同一路径加载出来可能是同一个对象": [
    "这一节讲 ResourceCache。Godot 会按路径和缓存模式复用已加载资源，所以多次加载同一路径可能拿到同一个 Resource 实例。这样能节省内存和保持资源共享，但也意味着你修改资源状态时，其他引用者可能看到同一份变化。",
    "小白可以把 ResourceCache 想成图书馆的同一本书被多人借阅的登记系统。你以为自己重新拿了一本，其实可能拿到同一个对象引用。调试资源状态串改、重新导入不生效、加载旧内容时，要检查缓存模式、路径 remap、UID、take_over_path 和资源是否被多个地方共享。"
  ],
  "错误处理风格": [
    "这一节讲 Godot 的错误处理习惯。源码里常见 ERR_FAIL_COND、ERR_FAIL_NULL、ERR_FAIL_INDEX、ERR_FAIL_V、WARN_PRINT、CRASH_COND 等宏。它们既表达防御性检查，也统一错误日志、返回值、调试行为和 release/debug 差异。",
    "小白读这些宏时，不要把它们当普通 if。ERR_FAIL_* 往往意味着函数在非法输入下提前返回，返回值可能是空对象、false、错误码或默认值；CRASH_* 更偏向发现绝不应该发生的内部错误。调试时要看宏后面的函数是否继续执行，以及调用者有没有处理失败返回。"
  ],
  "StringName、NodePath、错误宏和内存宏：读 C++ 基础设施的四个抓手": [
    "这一节补的是读 Godot C++ 时每天都会遇到的小基础设施。StringName 负责高频名字查找，NodePath 负责节点和属性路径，错误宏负责统一早返回和日志，memnew/memdelete 负责接入 Godot 的内存和对象生命周期钩子。",
    "小白不要把它们当语法噪音。看到 SNAME 要想到名字池和快速比较；看到 NodePath 要想到路径解析上下文；看到 ERR_FAIL_* 要先看返回值和调用者；看到 memnew/memdelete 要想到 Object 的 postinitialize、predelete 和 ObjectDB。"
  ],
  "为什么生成代码这么多": [
    "这一节解释 Godot 为什么有很多生成代码。大型引擎需要把模块列表、注册函数、文档、类裁剪、shader、图标、接口表、绑定信息等从结构化输入转成 C++ 或头文件。生成代码能减少手写重复，也能根据构建选项生成不同结果。",
    "小白不要看到 .gen.cpp 就害怕，也不要手改它。把它当成构建系统打印出来的中间产物，真正要读的是生成器脚本和输入数据。生成代码多，说明 Godot 把大量重复注册和资源打包自动化了；理解生成源头，比盯生成物更有用。"
  ],
  "每条路线的结束标准": [
    "这一节告诉你什么叫读到位。入门路线结束时，你应该能画出目录依赖、启动流程和主循环阶段；机制路线结束时，你应该能追一条功能从用户入口到绑定、对象状态、Resource/Server 和刷新时机；深入路线结束时，你应该能列出生命周期、线程、注册、保存格式、导出和测试影响面。",
    "小白不要把读完一页或看过几个文件当成结束标准。真正的结束标准是你能用自己的话回答问题，并能定位下次遇到同类问题该从哪里查。读源码的目的不是收藏文件名，而是建立可复用的路径和判断力。"
  ],
  "建议的“功能追踪法”": [
    "这一节给最终读法：选一个具体功能，从用户能看到的入口开始，沿着绑定、状态、资源、Server、主循环一路追到底，再回头记录影响面。不要泛泛地读整个目录，也不要从最底层算法开始，因为这样很容易看见很多细节却不知道它们服务哪个功能。",
    "小白可以每次只追一个问题，例如 Button.pressed 怎么发信号，Sprite2D.texture 怎么显示，PackedScene.instantiate 怎么创建节点，RigidBody3D 怎么参与物理步。每条追踪都写下入口、关键类、状态字段、刷新时机、常见坑。几条功能链积累起来，就会自然形成完整的 Godot 源码地图。"
  ]
};

const sourceMapGraph = {
  viewBox: { width: 3280, height: 1120 },
  relationTypes: {
    initializes: { label: "初始化", color: "#2a5a9f" },
    registers: { label: "注册/暴露", color: "#1a7f64" },
    "object-model": { label: "对象模型", color: "#a45b11" },
    "resource-flow": { label: "资源流", color: "#7c5cc4" },
    "scene-lifecycle": { label: "场景生命周期", color: "#27835f" },
    "server-delegate": { label: "委托给 Server", color: "#365c8d" },
    backend: { label: "后端实现", color: "#6552a8" },
    "editor-tooling": { label: "编辑器工具链", color: "#8a5a1f" },
    design: { label: "底层设计", color: "#6f6f45" },
    "runtime-loop": { label: "每帧调度", color: "#b05252" }
  },
  groups: [
    { id: "startup", title: "启动 / 平台入口", x: 40, y: 40, width: 700, height: 330, color: "#e9f3ff", description: "从平台可执行文件进入 Main，再创建 MainLoop 并驱动每帧。" },
    { id: "core", title: "core：对象和统一值", x: 790, y: 40, width: 760, height: 410, color: "#fff6e7", description: "让 C++ 类型能被脚本、编辑器、资源和扩展统一识别。" },
    { id: "scene", title: "scene：资源和运行中的树", x: 1600, y: 40, width: 780, height: 560, color: "#eef8ef", description: "把资源实例化成 Node 树，并把用户语义转成 Server 状态。" },
    { id: "servers", title: "servers：底层执行边界", x: 2430, y: 40, width: 800, height: 610, color: "#f2edfb", description: "渲染、物理、音频、显示、导航和输入的抽象 API 与后端入口。" },
    { id: "modules", title: "modules：可选能力接入", x: 790, y: 480, width: 760, height: 360, color: "#f6f0e6", description: "按初始化级别把脚本语言、导入器、网络、文本、物理后端等插入引擎。" },
    { id: "editor", title: "editor：工具运行时", x: 1600, y: 660, width: 780, height: 330, color: "#eaf2f6", description: "编辑器本体也是引擎里的场景树，复用 Object、Resource、SceneTree 和 Server。" },
    { id: "deep", title: "底层设计抓手", x: 40, y: 440, width: 700, height: 550, color: "#f8faf7", description: "读生命周期、线程、错误处理和源码路线时反复使用的判断框架。" }
  ],
  nodes: [
    {
      id: "platform-entry",
      title: "平台入口",
      group: "startup",
      summary: "各平台的 godot_*.cpp 把命令行、编码、OS 对象和平台事件循环准备好，再把控制权交给 Main。",
      beginner: "把它看成 Godot 可执行文件的门口。Windows、Linux、Web 的门口不一样，但进门后都会走向 Main。",
      conceptId: "platformdrivers",
      articleHref: "index.html#startup",
      sourceAnchors: ["platform/windows/godot_windows.cpp:68", "platform/linuxbsd/godot_linuxbsd.cpp:111"],
      tags: ["platform", "entry", "OS"],
      x: 150,
      y: 150,
      importance: 3
    },
    {
      id: "main-setup",
      title: "Main::setup",
      group: "startup",
      summary: "建立 core 层单例、注册基础类型、解析命令行和项目设置，让后续系统有统一对象世界。",
      beginner: "这一步像开机自检：先把对象系统、项目配置和最基础的服务台搭起来。",
      articleHref: "index.html#startup",
      sourceAnchors: ["main/main.cpp:1027", "main/main.cpp:1059", "main/main.cpp:2255"],
      tags: ["Main", "core", "setup"],
      x: 390,
      y: 120,
      importance: 4
    },
    {
      id: "main-setup2",
      title: "Main::setup2",
      group: "startup",
      summary: "注册 Server、Scene、模块和编辑器类型，创建渲染、物理、音频、文本等高层运行时基础。",
      beginner: "如果 setup 是打地基，setup2 就是在地基上安装渲染、物理、场景和编辑器这些大系统。",
      articleHref: "index.html#startup",
      sourceAnchors: ["main/main.cpp:3008", "main/main.cpp:3197", "main/main.cpp:3759"],
      tags: ["Main", "servers", "scene"],
      x: 630,
      y: 150,
      importance: 4
    },
    {
      id: "main-start",
      title: "Main::start",
      group: "startup",
      summary: "决定运行编辑器、项目管理器、命令行工具、自定义 MainLoop，还是默认 SceneTree 和主场景。",
      beginner: "这一步决定本次进程到底要做什么：开编辑器、跑游戏、打开项目管理器，还是执行工具命令。",
      articleHref: "index.html#startup",
      sourceAnchors: ["main/main.cpp:3988", "main/main.cpp:4335", "main/main.cpp:4737"],
      tags: ["Main", "SceneTree", "main scene"],
      x: 390,
      y: 280,
      importance: 4
    },
    {
      id: "main-iteration",
      title: "Main::iteration",
      group: "startup",
      summary: "每帧调度中心：物理步、process、消息队列、导航、渲染 sync/draw、调试和性能统计。",
      beginner: "游戏看起来在连续运行，本质上是这里一帧一帧推进各个系统。",
      articleHref: "index.html#startup",
      sourceAnchors: ["main/main.cpp:4896", "main/main.cpp:5037", "main/main.cpp:5052"],
      tags: ["frame", "process", "draw"],
      x: 630,
      y: 285,
      importance: 5
    },
    {
      id: "project-settings",
      title: "ProjectSettings",
      group: "startup",
      summary: "读取 project.godot、autoload、feature override、主场景路径和运行参数，给 Main、ResourceLoader、InputMap 使用。",
      beginner: "它像项目的配置总账。主场景在哪里、自动加载哪些脚本、输入和资源路径怎么解释，都要查它。",
      conceptId: "projectsettings",
      articleHref: "concepts.html#concept-projectsettings",
      sourceAnchors: ["core/config/project_settings.h:57", "main/main.cpp:4737"],
      tags: ["project.godot", "autoload", "main_scene"],
      x: 150,
      y: 285,
      importance: 3
    },
    {
      id: "object",
      title: "Object",
      group: "core",
      summary: "Godot 大多数运行时对象的共同根，提供类型名、属性、方法、信号、脚本实例、元数据和 ObjectID。",
      beginner: "Object 不是场景物体，而是 Godot 认识一个 C++ 对象的基础身份规则。",
      conceptId: "object",
      articleHref: "concepts.html#concept-object",
      sourceAnchors: ["core/object/object.h:349", "core/object/object.cpp:2294"],
      tags: ["Object", "base class"],
      x: 930,
      y: 280,
      importance: 5
    },
    {
      id: "objectdb",
      title: "ObjectDB",
      group: "core",
      summary: "全局弱索引，把 Object 登记成 ObjectID，让 Callable、消息队列和删除队列能重新查找活对象。",
      beginner: "它像通讯录，不是保险箱。能查到对象，但不会替你拥有对象。",
      conceptId: "objectdb",
      articleHref: "concepts.html#concept-objectdb",
      sourceAnchors: ["core/object/object.h:908", "core/object/object.cpp:2450"],
      tags: ["ObjectID", "weak lookup"],
      x: 1170,
      y: 280,
      importance: 3
    },
    {
      id: "classdb",
      title: "ClassDB",
      group: "core",
      summary: "运行时类型登记表，保存类名、继承、创建函数、方法、属性、信号和常量。",
      beginner: "脚本和 Inspector 能认识 C++ 类，靠的就是这张类型登记表。",
      conceptId: "classdb",
      articleHref: "concepts.html#concept-classdb",
      sourceAnchors: ["core/object/class_db.h:97", "core/object/class_db.cpp"],
      tags: ["reflection", "GDCLASS"],
      x: 1170,
      y: 155,
      importance: 5
    },
    {
      id: "methodbind",
      title: "MethodBind",
      group: "core",
      summary: "把 C++ 成员函数包装成运行时可调用对象，供 Object::callp、脚本和扩展统一调用。",
      beginner: "它像 C++ 方法和脚本调用之间的转接头。",
      conceptId: "methodbind",
      articleHref: "concepts.html#concept-methodbind",
      sourceAnchors: ["core/object/method_bind.h:38", "core/object/object.cpp:768"],
      tags: ["bind_method", "ptrcall"],
      x: 1410,
      y: 155,
      importance: 4
    },
    {
      id: "variant",
      title: "Variant",
      group: "core",
      summary: "统一值容器，让脚本参数、属性、信号、序列化和扩展边界能传递同一套值。",
      beginner: "Variant 像一个通用盒子，能装数字、字符串、对象、数组、字典等很多 Godot 值。",
      conceptId: "variant",
      articleHref: "concepts.html#concept-variant",
      sourceAnchors: ["core/variant/variant.h:93", "core/variant/variant.cpp"],
      tags: ["value", "script", "serialization"],
      x: 1410,
      y: 280,
      importance: 5
    },
    {
      id: "callable-signal",
      title: "Callable / Signal",
      group: "core",
      summary: "统一表示之后可以调用的目标和对象发出的事件，串起信号连接、延迟调用和脚本函数。",
      beginner: "Signal 是“发生了什么”，Callable 是“之后该叫谁来处理”。",
      conceptId: "callable-signal",
      articleHref: "concepts.html#concept-callable-signal",
      sourceAnchors: ["core/variant/callable.h:42", "core/object/object.cpp:1541"],
      tags: ["signal", "callable"],
      x: 1305,
      y: 390,
      importance: 3
    },
    {
      id: "messagequeue",
      title: "MessageQueue",
      group: "core",
      summary: "把 call_deferred、set_deferred、通知等操作放到安全点统一 flush，避免遍历中改树或重入。",
      beginner: "它像会后处理清单：先记下来，等安全时统一执行。",
      conceptId: "messagequeue",
      articleHref: "concepts.html#concept-messagequeue",
      sourceAnchors: ["core/object/message_queue.h:42", "core/object/message_queue.cpp:68"],
      tags: ["deferred", "queue_free"],
      x: 1035,
      y: 390,
      importance: 4
    },
    {
      id: "stringnamenodepath",
      title: "StringName / NodePath",
      group: "core",
      summary: "高频名字键和节点/属性路径基础设施，支撑方法名、信号名、组名、动画路径和场景引用。",
      beginner: "StringName 让名字查找更快，NodePath 让节点和属性路径能被保存和解析。",
      conceptId: "stringnamenodepath",
      articleHref: "concepts.html#concept-stringnamenodepath",
      sourceAnchors: ["core/string/string_name.h:38", "core/string/node_path.h:34"],
      tags: ["SNAME", "path"],
      x: 930,
      y: 155,
      importance: 3
    },
    {
      id: "resource",
      title: "Resource",
      group: "scene",
      summary: "可加载、可保存、可共享的数据对象基类；用 RefCounted 管生命周期，用 resource_path 接入 ResourceCache。",
      beginner: "它像项目里的资料卡：贴图、材质、脚本、场景蓝图都可以作为资源被多个地方共用。",
      conceptId: "resource",
      articleHref: "concepts.html#concept-resource",
      sourceAnchors: ["core/io/resource.h:52", "core/io/resource.cpp:733", "core/io/resource.cpp:829"],
      tags: ["Resource", "resource_path", "ResourceCache"],
      x: 1720,
      y: 290,
      importance: 5
    },
    {
      id: "resourceloader",
      title: "ResourceLoader",
      group: "scene",
      summary: "资源加载调度器，标准化路径、处理 remap/UID/cache，选择 ResourceFormatLoader。",
      beginner: "它负责把 .tscn、.tres、.gd、图片等文件变成 Godot 能用的 Resource。",
      conceptId: "resourceloader",
      articleHref: "concepts.html#concept-resourceloader",
      sourceAnchors: ["core/io/resource_loader.h:103", "core/io/resource_loader.cpp:513"],
      tags: ["Resource", "loader", "cache"],
      x: 1720,
      y: 155,
      importance: 5
    },
    {
      id: "packedscene",
      title: "PackedScene",
      group: "scene",
      summary: "场景模板资源；加载后还不是运行节点，instantiate() 才创建真正的 Node 树。",
      beginner: "它像场景蓝图，只有实例化以后才变成运行中的节点树。",
      conceptId: "packedscene",
      articleHref: "concepts.html#concept-packedscene",
      sourceAnchors: ["scene/resources/packed_scene.h:246", "scene/resources/packed_scene.cpp"],
      tags: [".tscn", "scene resource"],
      x: 1950,
      y: 155,
      importance: 5
    },
    {
      id: "scenestate",
      title: "SceneState",
      group: "scene",
      summary: "PackedScene 内部的数据表，保存节点、属性、连接、路径和资源引用，供实例化恢复树结构。",
      beginner: "它是 PackedScene 里面那张真正记录“有哪些节点、属性和连接”的表。",
      conceptId: "scenestate",
      articleHref: "concepts.html#concept-scenestate",
      sourceAnchors: ["scene/resources/packed_scene.h:102", "scene/resources/packed_scene.cpp"],
      tags: ["SceneState", "_bundled"],
      x: 2180,
      y: 155,
      importance: 3
    },
    {
      id: "node",
      title: "Node",
      group: "scene",
      summary: "场景树基本单位，负责父子关系、owner、组、路径、生命周期通知和 process 开关。",
      beginner: "Node 是游戏对象出现在场景树里的基本单位。",
      conceptId: "node",
      articleHref: "concepts.html#concept-node",
      sourceAnchors: ["scene/main/node.h:54", "scene/main/node.cpp"],
      tags: ["Node", "owner", "lifecycle"],
      x: 1950,
      y: 290,
      importance: 5
    },
    {
      id: "scenetree",
      title: "SceneTree",
      group: "scene",
      summary: "默认 MainLoop，负责 root/current_scene、process/physics、组调用、场景切换、timer/tween、删除队列。",
      beginner: "它是节点世界的调度员：谁进树、谁 ready、谁每帧执行，都由它安排。",
      conceptId: "scenetree",
      articleHref: "concepts.html#concept-scenetree",
      sourceAnchors: ["scene/main/scene_tree.h:89", "scene/main/scene_tree.cpp:639"],
      tags: ["MainLoop", "process", "queue_delete"],
      x: 2180,
      y: 290,
      importance: 5
    },
    {
      id: "viewportwindowworld",
      title: "Viewport / Window / World",
      group: "scene",
      summary: "连接场景树、渲染目标、输入、GUI、World2D/World3D 和平台窗口的关键枢纽。",
      beginner: "它决定输入打到哪里、画面渲到哪里、2D/3D 世界上下文是谁。",
      conceptId: "viewportwindowworld",
      articleHref: "concepts.html#concept-viewportwindowworld",
      sourceAnchors: ["scene/main/viewport.h:51", "scene/main/window.h:42"],
      tags: ["Viewport", "Window", "World3D"],
      x: 2180,
      y: 430,
      importance: 4
    },
    {
      id: "canvasitem",
      title: "CanvasItem",
      group: "scene",
      summary: "2D 和 GUI 的共同渲染基类，持有 canvas item RID 并向 RenderingServer 提交 draw 状态。",
      beginner: "2D 节点和 GUI 控件能画出来，通常都要经过 CanvasItem。",
      conceptId: "canvasitem",
      articleHref: "concepts.html#concept-canvasitem",
      sourceAnchors: ["scene/main/canvas_item.h:47", "scene/main/canvas_item.cpp"],
      tags: ["2D", "GUI", "_draw"],
      x: 1720,
      y: 430,
      importance: 4
    },
    {
      id: "visualinstance3d",
      title: "VisualInstance3D",
      group: "scene",
      summary: "3D 可见对象基础节点，把 Mesh/Light/Decal 等 base、scenario 和 transform 提交给 RenderingServer。",
      beginner: "3D 物体能进入渲染世界，通常靠它把场景节点翻译成渲染实例。",
      conceptId: "visualinstance3d",
      articleHref: "concepts.html#concept-visualinstance3d",
      sourceAnchors: ["scene/3d/visual_instance_3d.h:51", "scene/3d/visual_instance_3d.cpp"],
      tags: ["3D", "MeshInstance3D"],
      x: 1950,
      y: 430,
      importance: 4
    },
    {
      id: "controlgui",
      title: "Control / GUI",
      group: "scene",
      summary: "GUI 节点层，在 CanvasItem 之上增加布局、焦点、鼠标过滤、主题查找和 TextServer 测量。",
      beginner: "按钮、面板、输入框这些 UI 控件，核心入口是 Control。",
      conceptId: "controlgui",
      articleHref: "concepts.html#concept-controlgui",
      sourceAnchors: ["scene/gui/control.h:58", "scene/theme/theme_db.h:44"],
      tags: ["Control", "Theme", "TextServer"],
      x: 1720,
      y: 545,
      importance: 3
    },
    {
      id: "animationtween",
      title: "Animation / Tween",
      group: "scene",
      summary: "Animation 保存可序列化时间轴，Tween 是 SceneTree 管理的运行时插值对象。",
      beginner: "Animation 适合保存成资源，Tween 更像脚本临时创建的一段变化。",
      conceptId: "animationtween",
      articleHref: "concepts.html#concept-animationtween",
      sourceAnchors: ["scene/animation/animation_player.h:43", "scene/animation/tween.h:52"],
      tags: ["AnimationPlayer", "Tween"],
      x: 2280,
      y: 545,
      importance: 3
    },
    {
      id: "server",
      title: "Server 架构",
      group: "servers",
      summary: "场景节点表达用户语义，Server 保存真实执行状态，后端把状态变成渲染、物理、音频等操作。",
      beginner: "节点不直接干重活，而是把工作交给更底层的 Server。",
      conceptId: "server",
      articleHref: "concepts.html#concept-server",
      sourceAnchors: ["servers/register_server_types.cpp:146", "servers/rendering/rendering_server.h:64"],
      tags: ["Server", "RID"],
      x: 2560,
      y: 170,
      importance: 5
    },
    {
      id: "rid",
      title: "RID",
      group: "servers",
      summary: "Server 世界里的轻量句柄；真实 texture/body/mesh 等对象由对应 Server 的 RID_Owner 管理。",
      beginner: "RID 像订单号，不是智能指针。它能让 Server 找到内部对象，但不能自动释放。",
      conceptId: "rid",
      articleHref: "concepts.html#concept-rid",
      sourceAnchors: ["core/templates/rid.h:38", "core/templates/rid_owner.h:517"],
      tags: ["RID_Owner", "handle"],
      x: 2785,
      y: 150,
      importance: 5
    },
    {
      id: "renderingserver",
      title: "RenderingServer",
      group: "servers",
      summary: "统一渲染 Server API，接收 texture、mesh、canvas item、3D instance、viewport 等 RID 状态。",
      beginner: "场景层想显示东西，最后通常要把状态交给 RenderingServer。",
      conceptId: "renderingserver",
      articleHref: "concepts.html#concept-renderingserver",
      sourceAnchors: ["servers/rendering/rendering_server.h:64", "servers/rendering/rendering_server_default.cpp"],
      tags: ["render", "canvas", "mesh"],
      x: 2785,
      y: 285,
      importance: 5
    },
    {
      id: "rendererrd",
      title: "RendererRD",
      group: "servers",
      summary: "RD 渲染后端组织层，按 viewport 组织 2D/3D 渲染，再通过 RenderingDevice 提交命令。",
      beginner: "RenderingServer 接到任务后，RendererRD 负责把它组织成具体渲染流程。",
      conceptId: "rendererrd",
      articleHref: "concepts.html#concept-rendererrd",
      sourceAnchors: ["servers/rendering/renderer_rd/renderer_compositor_rd.h:47", "servers/rendering/renderer_viewport.cpp"],
      tags: ["RendererRD", "viewport"],
      x: 3015,
      y: 285,
      importance: 4
    },
    {
      id: "renderingdevice",
      title: "RenderingDevice",
      group: "servers",
      summary: "底层图形 API 抽象，管理 texture、buffer、shader、pipeline、uniform set 和 draw/compute list。",
      beginner: "它是 Godot 和 Vulkan 等底层图形能力之间更贴近 GPU 的接口。",
      conceptId: "renderingdevice",
      articleHref: "concepts.html#concept-renderingdevice",
      sourceAnchors: ["servers/rendering/rendering_device.h:67", "servers/rendering/rendering_device_driver.h:90"],
      tags: ["RD", "GPU", "Vulkan"],
      x: 3115,
      y: 420,
      importance: 4
    },
    {
      id: "displayserver",
      title: "DisplayServer",
      group: "servers",
      summary: "平台显示与窗口边界，负责窗口、surface、输入事件、屏幕、鼠标、键盘、剪贴板和 IME。",
      beginner: "它把 Godot 的窗口和输入抽象落到真实操作系统。",
      conceptId: "displayserver",
      articleHref: "concepts.html#concept-displayserver",
      sourceAnchors: ["servers/display/display_server.h:62", "servers/display/display_server.cpp"],
      tags: ["window", "input", "platform"],
      x: 3015,
      y: 150,
      importance: 4
    },
    {
      id: "physicsserver",
      title: "PhysicsServer",
      group: "servers",
      summary: "物理节点和真实物理后端之间的 RID 边界，管理 Space、Body、Area、Shape、Joint。",
      beginner: "RigidBody、Area、Shape 等场景对象真正求解碰撞时要进入 PhysicsServer。",
      conceptId: "physicsserver",
      articleHref: "concepts.html#concept-physicsserver",
      sourceAnchors: ["servers/physics_3d/physics_server_3d.h:236", "servers/physics_2d/physics_server_2d.h:225"],
      tags: ["physics", "body", "shape"],
      x: 2785,
      y: 425,
      importance: 4
    },
    {
      id: "audioserver",
      title: "AudioServer",
      group: "servers",
      summary: "音频混音中心，维护 playback、bus、effect 和混音缓冲，再交给 AudioDriver 输出。",
      beginner: "播放节点发起播放，真正混音和送到设备的是 AudioServer 和驱动。",
      conceptId: "audioserver",
      articleHref: "concepts.html#concept-audioserver",
      sourceAnchors: ["servers/audio/audio_server.h:45", "servers/audio/audio_driver_dummy.h:40"],
      tags: ["audio", "bus", "driver"],
      x: 3015,
      y: 545,
      importance: 3
    },
    {
      id: "navigationserver",
      title: "NavigationServer",
      group: "servers",
      summary: "管理导航 map、region、link、agent、路径查询和避障，和物理相邻但不是物理子功能。",
      beginner: "物理回答撞没撞，导航回答怎么走。",
      conceptId: "navigationserver",
      articleHref: "concepts.html#concept-navigationserver",
      sourceAnchors: ["servers/navigation_3d/navigation_server_3d.h:46", "servers/navigation_2d/navigation_server_2d.h:50"],
      tags: ["navigation", "agent", "map"],
      x: 2560,
      y: 545,
      importance: 3
    },
    {
      id: "inputsystem",
      title: "Input / InputMap",
      group: "servers",
      summary: "平台/DisplayServer 产生 InputEvent，Input 更新状态，InputMap 判断 action，Viewport/Control 分发事件。",
      beginner: "按键从系统进来后，要经过输入状态、动作映射，再交给 GUI 或脚本处理。",
      conceptId: "inputsystem",
      articleHref: "concepts.html#concept-inputsystem",
      sourceAnchors: ["core/input/input.h:61", "core/input/input_map.h:44"],
      tags: ["InputEvent", "action"],
      x: 2560,
      y: 315,
      importance: 4
    },
    {
      id: "moduleatlas",
      title: "Modules / register_types",
      group: "modules",
      summary: "模块通过 config.py、SCsub 和 register_types 按初始化级别把能力接入 core、servers、scene、editor。",
      beginner: "modules 不是固定层，而是一张张插卡，按需要插到不同初始化阶段。",
      conceptId: "moduleatlas",
      articleHref: "concepts.html#concept-moduleatlas",
      sourceAnchors: ["modules/SCsub:20", "modules/modules_builders.py:43", "main/main.cpp:2255"],
      tags: ["modules", "register_types"],
      x: 935,
      y: 600,
      importance: 5
    },
    {
      id: "scriptextension",
      title: "Script / GDExtension",
      group: "modules",
      summary: "脚本语言通过 ScriptLanguage 注册，脚本文件是 Script 资源，Object 持有 ScriptInstance；GDExtension 通过 C ABI 接入 ClassDB。",
      beginner: "脚本不是 Node 的特例，而是 Object 上挂的运行时实例和语言插件体系。",
      conceptId: "scriptextension",
      articleHref: "concepts.html#concept-scriptextension",
      sourceAnchors: ["core/object/script_language.h:46", "modules/gdscript/register_types.cpp:137", "core/extension/gdextension.cpp"],
      tags: ["GDScript", "GDExtension"],
      x: 1190,
      y: 600,
      importance: 5
    },
    {
      id: "resourceimportpipeline",
      title: "Resource Import Pipeline",
      group: "modules",
      summary: "编辑器扫描源文件和 .import，生成内部资源，用 UID 维持引用，再供 ResourceLoader 和导出流程消费。",
      beginner: "导入管线把原始素材变成 Godot 运行时更适合读取的资源。",
      conceptId: "resourceimportpipeline",
      articleHref: "concepts.html#concept-resourceimportpipeline",
      sourceAnchors: ["core/io/resource_importer.h:43", "editor/import/3d/resource_importer_scene.h:155"],
      tags: [".import", "ResourceUID"],
      x: 1415,
      y: 690,
      importance: 4
    },
    {
      id: "multiplayerapi",
      title: "MultiplayerAPI / RPC",
      group: "modules",
      summary: "Node/SceneTree 提供入口，MultiplayerAPI 管 RPC、peer 和场景复制，传输层由模块或平台 peer 实现。",
      beginner: "多人同步的用户入口在场景层，网络传输和 RPC 调度在更专门的系统里。",
      conceptId: "multiplayerapi",
      articleHref: "concepts.html#concept-multiplayerapi",
      sourceAnchors: ["modules/multiplayer/scene_multiplayer.h:48", "scene/main/multiplayer_api.h:39"],
      tags: ["RPC", "network"],
      x: 1190,
      y: 760,
      importance: 3
    },
    {
      id: "translationserver",
      title: "TranslationServer",
      group: "modules",
      summary: "本地化中心，管理语言环境、翻译资源和文本查找，GUI、脚本和导入器都围绕它协作。",
      beginner: "它负责把同一个文本键映射成当前语言下该显示的文字。",
      conceptId: "translationserver",
      articleHref: "concepts.html#concept-translationserver",
      sourceAnchors: ["core/string/translation.h:47", "core/string/translation.cpp"],
      tags: ["locale", "translation"],
      x: 960,
      y: 760,
      importance: 3
    },
    {
      id: "editorarchitecture",
      title: "EditorNode / Editor",
      group: "editor",
      summary: "编辑器本体运行在引擎里的 SceneTree 上，EditorNode 挂接主界面、状态、dock、菜单和工具系统。",
      beginner: "Godot 编辑器不是引擎外部程序，而是引擎运行时里的一棵复杂工具场景。",
      conceptId: "editorarchitecture",
      articleHref: "concepts.html#concept-editorarchitecture",
      sourceAnchors: ["editor/editor_node.h:120", "editor/editor_node.cpp"],
      tags: ["EditorNode", "tools"],
      x: 1745,
      y: 775,
      importance: 5
    },
    {
      id: "editor-inspector",
      title: "Inspector",
      group: "editor",
      summary: "Inspector 通过 Object/ClassDB/ScriptInstance 的属性列表生成编辑 UI，并配合 UndoRedo 修改对象。",
      beginner: "属性面板不是手写每个类的 UI，而是读对象暴露出来的属性表。",
      conceptId: "editorarchitecture",
      articleHref: "index.html#editor",
      sourceAnchors: ["editor/inspector/editor_inspector.h:731", "editor/inspector/editor_inspector.cpp"],
      tags: ["Inspector", "PropertyInfo"],
      x: 1980,
      y: 760,
      importance: 4
    },
    {
      id: "editor-importer",
      title: "Editor Import / Export",
      group: "editor",
      summary: "导入导出工具把编辑器资源、平台模板和模块能力组织成可运行项目或导入产物。",
      beginner: "编辑器负责把素材准备好，也负责把项目打包到不同平台。",
      conceptId: "resourceimportpipeline",
      articleHref: "index.html#editor",
      sourceAnchors: ["editor/import/editor_import_plugin.h:53", "editor/export/editor_export_platform.h:87"],
      tags: ["import", "export"],
      x: 2230,
      y: 835,
      importance: 4
    },
    {
      id: "debugperformance",
      title: "Debugger / Performance",
      group: "editor",
      summary: "运行时 Performance/EngineDebugger 采样并发消息，编辑器调试面板接收、解析和展示。",
      beginner: "调试数据一半在游戏进程采样，一半在编辑器 UI 里展示。",
      conceptId: "debugperformance",
      articleHref: "concepts.html#concept-debugperformance",
      sourceAnchors: ["main/performance.cpp:147", "editor/debugger/editor_debugger_node.h:47"],
      tags: ["debugger", "profiler"],
      x: 1985,
      y: 920,
      importance: 3
    },
    {
      id: "deepdesignpatterns",
      title: "Ownership / Lifecycle",
      group: "deep",
      summary: "区分 Object、Node、Resource、RefCounted、RID、Callable、ObjectID 的拥有关系和释放通道。",
      beginner: "读崩溃和泄漏时，先问谁创建、谁保存引用、谁负责释放。",
      conceptId: "deepdesignpatterns",
      articleHref: "concepts.html#concept-deepdesignpatterns",
      sourceAnchors: ["core/object/object.h:349", "scene/main/node.h:54", "core/templates/rid.h:38"],
      tags: ["ownership", "lifecycle"],
      x: 175,
      y: 575,
      importance: 5
    },
    {
      id: "errorandmemorymacros",
      title: "Error Macros / Memory",
      group: "deep",
      summary: "ERR_FAIL_* 统一早返回、日志和失败语义；memnew/memdelete 接入内存统计和 Object 生命周期钩子。",
      beginner: "这些宏不是噪音，它们告诉你非法输入、释放路径和调试行为怎么处理。",
      conceptId: "errorandmemorymacros",
      articleHref: "concepts.html#concept-errorandmemorymacros",
      sourceAnchors: ["core/error/error_macros.h:134", "core/os/memory.h:150"],
      tags: ["ERR_FAIL", "memnew"],
      x: 525,
      y: 575,
      importance: 4
    },
    {
      id: "renderthread",
      title: "Render Thread / wrap_mt",
      group: "deep",
      summary: "主线程提交 RenderingServer API，单线程模式直接执行，Separate 模式通过命令队列给渲染线程执行。",
      beginner: "很多 Server 调用看起来立即发生，实际可能先排队，等 sync/draw 或专用线程处理。",
      conceptId: "renderthread",
      articleHref: "concepts.html#concept-renderthread",
      sourceAnchors: ["servers/server_wrap_mt_common.h", "main/main.cpp:5052"],
      tags: ["thread", "sync", "draw"],
      x: 525,
      y: 760,
      importance: 4
    },
    {
      id: "sourcereadingroadmap",
      title: "功能追踪法",
      group: "deep",
      summary: "从用户 API 开始，找绑定、状态、Resource/Server 边界、主循环刷新点，再回头记录影响面。",
      beginner: "不要漫游目录。每次拿一个具体功能，从用户入口一路追到底。",
      conceptId: "sourcereadingroadmap",
      articleHref: "concepts.html#concept-sourcereadingroadmap",
      sourceAnchors: ["doc/study/godot-source-guide/index.html:2536", "doc/study/godot_engine_source_reading_guide.md:1"],
      tags: ["reading route", "trace"],
      x: 175,
      y: 850,
      importance: 4
    }
  ],
  edges: [
    { from: "platform-entry", to: "main-setup", type: "initializes", label: "进入底层 setup", explanation: "平台入口完成 OS 和参数准备后调用 Main::setup。", weight: 3 },
    { from: "main-setup", to: "object", type: "initializes", label: "建立对象世界", explanation: "register_core_types 在 setup 阶段让 Object、Variant、Resource 等基础类型可用。", weight: 5 },
    { from: "main-setup", to: "resource", type: "initializes", label: "注册 Resource 基类", explanation: "register_core_types 在 core 阶段注册 Resource、基础二进制加载器和资源缓存入口。", weight: 4 },
    { from: "main-setup", to: "project-settings", type: "initializes", label: "读取项目配置", explanation: "Main::setup 解析项目设置、命令行和基础资源路径。", weight: 4 },
    { from: "main-setup", to: "moduleatlas", type: "initializes", label: "CORE 模块", explanation: "模块先在 CORE 级别接入资源格式和基础能力。", weight: 3 },
    { from: "main-setup", to: "main-setup2", type: "initializes", label: "进入高层运行时", explanation: "setup 完成后，setup2 注册 Server、Scene 和高层模块。", weight: 4 },
    { from: "main-setup2", to: "server", type: "initializes", label: "注册 Server 类型", explanation: "register_server_types 创建底层服务 API 和单例边界。", weight: 5 },
    { from: "main-setup2", to: "node", type: "initializes", label: "注册 Scene 类型", explanation: "register_scene_types 让 Node、SceneTree、Control 等用户对象可实例化。", weight: 5 },
    { from: "main-setup2", to: "scriptextension", type: "initializes", label: "脚本语言接入", explanation: "脚本语言模块在对应初始化级别注册 ScriptLanguage 和资源 loader。", weight: 4 },
    { from: "main-setup2", to: "main-start", type: "initializes", label: "准备运行模式", explanation: "setup2 完成后，Main::start 决定运行编辑器、项目或工具。", weight: 4 },
    { from: "main-start", to: "scenetree", type: "scene-lifecycle", label: "创建默认 MainLoop", explanation: "没有自定义 MainLoop 时，Main::start 创建 SceneTree。", weight: 5 },
    { from: "main-start", to: "resourceloader", type: "resource-flow", label: "加载主场景", explanation: "游戏启动路径把 main_scene 或 --scene 交给 ResourceLoader。", weight: 5 },
    { from: "main-start", to: "editorarchitecture", type: "editor-tooling", label: "编辑器模式", explanation: "编辑器运行时在 SceneTree 里挂接 EditorNode。", weight: 4 },
    { from: "main-iteration", to: "scenetree", type: "runtime-loop", label: "process / physics", explanation: "Main::iteration 调用 SceneTree 的物理和空闲处理。", weight: 5 },
    { from: "main-iteration", to: "messagequeue", type: "runtime-loop", label: "安全点 flush", explanation: "每帧安全点刷新延迟调用和删除队列。", weight: 4 },
    { from: "main-iteration", to: "renderingserver", type: "runtime-loop", label: "sync / draw", explanation: "渲染命令在 Main::iteration 的 sync/draw 阶段被推进。", weight: 5 },
    { from: "main-iteration", to: "physicsserver", type: "runtime-loop", label: "固定物理步", explanation: "物理 Server 在固定步同步、查询、求解。", weight: 4 },
    { from: "object", to: "objectdb", type: "object-model", label: "登记 ObjectID", explanation: "Object 构造和析构时进入/离开 ObjectDB。", weight: 3 },
    { from: "object", to: "classdb", type: "object-model", label: "类型元数据", explanation: "Object 子类通过 ClassDB 暴露类名、继承、方法、属性和信号。", weight: 5 },
    { from: "classdb", to: "methodbind", type: "registers", label: "保存绑定方法", explanation: "_bind_methods 把 C++ 方法包装成 MethodBind 并登记到 ClassDB。", weight: 5 },
    { from: "methodbind", to: "variant", type: "object-model", label: "统一参数", explanation: "MethodBind 通过 Variant call 或 ptrcall 接收脚本和扩展传入的参数。", weight: 4 },
    { from: "object", to: "callable-signal", type: "object-model", label: "信号和调用目标", explanation: "Signal 和 Callable 都围绕 ObjectID、方法名和 Variant 参数工作。", weight: 3, defaultVisible: false },
    { from: "callable-signal", to: "messagequeue", type: "object-model", label: "延迟调用", explanation: "call_deferred 等操作通过 MessageQueue 延后执行。", weight: 3 },
    { from: "stringnamenodepath", to: "classdb", type: "object-model", label: "名字键", explanation: "方法名、信号名、属性名大量依赖 StringName。", weight: 2, defaultVisible: false },
    { from: "stringnamenodepath", to: "node", type: "scene-lifecycle", label: "节点路径", explanation: "NodePath 保存场景节点和属性路径引用。", weight: 2, defaultVisible: false },
    { from: "project-settings", to: "resourceloader", type: "resource-flow", label: "路径和 remap", explanation: "项目设置提供资源路径、feature override 和主场景位置。", weight: 3 },
    { from: "object", to: "resource", type: "object-model", label: "Object + RefCounted 基础", explanation: "Resource 继承 RefCounted，仍然拥有 Object 的属性、方法、信号和 ObjectID。", weight: 4 },
    { from: "resourceloader", to: "resource", type: "resource-flow", label: "返回资源对象", explanation: "ResourceLoader 通过格式加载器把路径解析成 Ref<Resource>，再按路径进入 ResourceCache。", weight: 5 },
    { from: "resource", to: "packedscene", type: "object-model", label: "场景也是资源", explanation: "PackedScene 继承 Resource，加载 .tscn 后先得到可实例化的资源蓝图。", weight: 5 },
    { from: "resourceloader", to: "packedscene", type: "resource-flow", label: "加载场景资源", explanation: ".tscn/.scn 通过 ResourceLoader 变成 PackedScene。", weight: 5 },
    { from: "packedscene", to: "scenestate", type: "resource-flow", label: "内部场景表", explanation: "PackedScene 用 SceneState 保存节点、属性和连接数据。", weight: 4 },
    { from: "scenestate", to: "node", type: "scene-lifecycle", label: "实例化 Node", explanation: "instantiate 按 SceneState 创建节点、恢复属性和连接。", weight: 5 },
    { from: "node", to: "scenetree", type: "scene-lifecycle", label: "进入树和回调", explanation: "SceneTree 管理 enter_tree、ready、process、退出和删除。", weight: 5 },
    { from: "scenetree", to: "messagequeue", type: "scene-lifecycle", label: "删除和延迟队列", explanation: "queue_free 和 queue_delete 在 SceneTree 安全点处理。", weight: 4 },
    { from: "scenetree", to: "viewportwindowworld", type: "scene-lifecycle", label: "root Window / Viewport", explanation: "SceneTree 持有 root Window，Viewport 连接输入、GUI、世界和渲染目标。", weight: 4 },
    { from: "node", to: "canvasitem", type: "server-delegate", label: "2D/GGUI 可见节点", explanation: "CanvasItem 是 Node 的可绘制 2D/GUI 分支。", weight: 3 },
    { from: "node", to: "visualinstance3d", type: "server-delegate", label: "3D 可见节点", explanation: "VisualInstance3D 是 Node 的 3D 可见对象分支。", weight: 3 },
    { from: "canvasitem", to: "renderingserver", type: "server-delegate", label: "canvas item RID", explanation: "CanvasItem 把绘制命令和状态提交给 RenderingServer。", weight: 5 },
    { from: "visualinstance3d", to: "renderingserver", type: "server-delegate", label: "instance RID", explanation: "3D 可见节点把 base、transform 和 scenario 提交给 RenderingServer。", weight: 5 },
    { from: "viewportwindowworld", to: "renderingserver", type: "server-delegate", label: "viewport / render target", explanation: "Viewport 决定渲染目标和世界上下文，最终由 RenderingServer 绘制。", weight: 4 },
    { from: "viewportwindowworld", to: "displayserver", type: "server-delegate", label: "窗口和 surface", explanation: "Window/Viewport 把显示和输入边界接到 DisplayServer。", weight: 4 },
    { from: "controlgui", to: "canvasitem", type: "scene-lifecycle", label: "GUI 继承绘制层", explanation: "Control 建立在 CanvasItem 之上，增加布局、焦点和主题。", weight: 3 },
    { from: "controlgui", to: "inputsystem", type: "server-delegate", label: "GUI 输入分发", explanation: "InputEvent 进入 Viewport 后可能先被 Control 处理。", weight: 3 },
    { from: "animationtween", to: "scenetree", type: "runtime-loop", label: "按帧推进", explanation: "Tween 和 AnimationPlayer 随 SceneTree 的处理阶段推进。", weight: 3, defaultVisible: false },
    { from: "server", to: "rid", type: "design", label: "句柄边界", explanation: "Server 用 RID 隐藏真实后端对象。", weight: 5 },
    { from: "resource", to: "rid", type: "server-delegate", label: "可暴露底层 RID", explanation: "Texture、Mesh 等 Resource 可通过 get_rid() 返回对应 Server 的底层句柄。", weight: 3, defaultVisible: false },
    { from: "renderingserver", to: "rid", type: "server-delegate", label: "渲染资源句柄", explanation: "texture、mesh、instance、canvas item 都以 RID 暴露给场景层。", weight: 4 },
    { from: "renderingserver", to: "rendererrd", type: "backend", label: "默认 RD 后端", explanation: "RenderingServerDefault 将状态分发给 RendererRD 体系。", weight: 5 },
    { from: "rendererrd", to: "renderingdevice", type: "backend", label: "提交 GPU 命令", explanation: "RendererRD 通过 RenderingDevice 创建资源并提交 draw/compute。", weight: 5 },
    { from: "renderingserver", to: "renderthread", type: "design", label: "线程隔离", explanation: "RenderingServer 调用可能通过命令队列进入渲染线程。", weight: 4 },
    { from: "displayserver", to: "platform-entry", type: "backend", label: "平台实现", explanation: "DisplayServer 的具体窗口/输入实现落在 platform/*。", weight: 3 },
    { from: "inputsystem", to: "displayserver", type: "backend", label: "系统事件来源", explanation: "平台显示层抽取系统输入事件，再交给 Input/Viewport。", weight: 3 },
    { from: "physicsserver", to: "rid", type: "server-delegate", label: "body / shape RID", explanation: "物理对象通过 RID 进入对应后端空间。", weight: 4 },
    { from: "audioserver", to: "server", type: "server-delegate", label: "音频服务", explanation: "AudioServer 是 Server 架构的一部分，但输出最终由 AudioDriver 完成。", weight: 2, defaultVisible: false },
    { from: "navigationserver", to: "scenetree", type: "runtime-loop", label: "导航处理阶段", explanation: "导航更新在主循环的特定阶段推进。", weight: 3, defaultVisible: false },
    { from: "moduleatlas", to: "scriptextension", type: "registers", label: "脚本语言模块", explanation: "gdscript/mono 等模块注册 ScriptLanguage、Script 资源和编辑器工具。", weight: 5 },
    { from: "scriptextension", to: "object", type: "object-model", label: "ScriptInstance 挂到 Object", explanation: "Object 持有脚本实例，set/get/call 会先尝试脚本侧。", weight: 5 },
    { from: "scriptextension", to: "classdb", type: "registers", label: "扩展类注册", explanation: "GDExtension 通过 ClassDB 和 MethodBind 接入原生类型系统。", weight: 4 },
    { from: "moduleatlas", to: "physicsserver", type: "backend", label: "物理后端模块", explanation: "Godot Physics、Jolt 等模块注册 PhysicsServer 后端。", weight: 4 },
    { from: "moduleatlas", to: "resourceimportpipeline", type: "editor-tooling", label: "格式和导入器", explanation: "glTF、图片、音频等模块和编辑器导入器共同支持资源格式。", weight: 4 },
    { from: "resourceimportpipeline", to: "resourceloader", type: "resource-flow", label: "导入产物被加载", explanation: "编辑器生成的导入资源最终仍由 ResourceLoader 消费。", weight: 4 },
    { from: "moduleatlas", to: "multiplayerapi", type: "registers", label: "网络能力", explanation: "高层多人 API 与传输实现按模块和 core/io 边界接入。", weight: 2, defaultVisible: false },
    { from: "translationserver", to: "controlgui", type: "server-delegate", label: "本地化文本", explanation: "GUI 和脚本显示文本时会查询 TranslationServer 和 TextServer。", weight: 2, defaultVisible: false },
    { from: "editorarchitecture", to: "scenetree", type: "editor-tooling", label: "编辑器也是场景树", explanation: "EditorNode 挂在 SceneTree 里运行。", weight: 5 },
    { from: "editor-inspector", to: "classdb", type: "editor-tooling", label: "读取属性元数据", explanation: "Inspector 根据 ClassDB、Object 和 ScriptInstance 的属性列表生成 UI。", weight: 5 },
    { from: "editor-inspector", to: "object", type: "editor-tooling", label: "编辑 Object 属性", explanation: "Inspector 最终对选中 Object 做 get/set 和 UndoRedo 修改。", weight: 4 },
    { from: "editor-importer", to: "resourceimportpipeline", type: "editor-tooling", label: "导入导出流程", explanation: "编辑器导入导出工具围绕 ResourceImporter、ResourceUID 和平台导出能力组织。", weight: 4 },
    { from: "debugperformance", to: "main-iteration", type: "runtime-loop", label: "采样每帧数据", explanation: "性能统计和调试器在主循环中采样并发送消息。", weight: 3, defaultVisible: false },
    { from: "deepdesignpatterns", to: "object", type: "design", label: "对象生命周期", explanation: "Object、ObjectDB、Callable 和消息队列都要按所有权规则阅读。", weight: 4, defaultVisible: false },
    { from: "deepdesignpatterns", to: "node", type: "design", label: "树和删除", explanation: "Node 的 parent/owner/queue_free 与 Object 生命周期不同。", weight: 4, defaultVisible: false },
    { from: "deepdesignpatterns", to: "resource", type: "design", label: "资源共享和隔离", explanation: "Resource 的 Ref、ResourceCache、duplicate 和 local-to-scene 决定数据对象如何共享或复制。", weight: 4, defaultVisible: false },
    { from: "deepdesignpatterns", to: "rid", type: "design", label: "Server 内部所有权", explanation: "RID 句柄不拥有真实后端对象，释放必须回到对应 Server。", weight: 4, defaultVisible: false },
    { from: "errorandmemorymacros", to: "object", type: "design", label: "创建和释放钩子", explanation: "memnew/memdelete 影响 Object postinitialize、predelete 和调试统计。", weight: 3, defaultVisible: false },
    { from: "sourcereadingroadmap", to: "classdb", type: "design", label: "先找绑定", explanation: "功能追踪通常从用户 API 进入 _bind_methods 和 ClassDB。", weight: 3, defaultVisible: false },
    { from: "sourcereadingroadmap", to: "server", type: "design", label: "再找 Server 边界", explanation: "看到 RID/Server/ResourceLoader 时沿边界继续追。", weight: 3, defaultVisible: false }
  ]
};

// Concept entries use the minimal shape below.
// Required: id, title, article. Optional: aliases, summary.
// article can be a plain string or an array of blocks:
// { type: "heading" | "subheading" | "lead" | "paragraph" | "list" | "code" | "table" | "flow" | "callout", ... }
// {
//   id: "object",
//   title: "Object",
//   aliases: ["Object", "对象系统"],
//   article: `
//     在这里直接写完整解释即可。可以从直觉、源码入口、实现细节、常见误区一路自然写下去。
//
//     段落之间空一行；右侧概念面板会按普通文章渲染。
//   `
// }

const modulesWithoutRegister = new Set(["freetype", "msdfgen"]);
