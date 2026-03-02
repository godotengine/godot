# Godot Engine Architecture

## Directory Structure

```
godot-engine/
  core/          Core engine: Object system, Variant, math, IO, templates, strings
  scene/         Scene tree: Node types (2D, 3D, GUI), animation, resources
  servers/       Backend singletons: rendering, physics, audio, display, navigation
  modules/       57 optional modules: gdscript, mono, physics, formats, networking
  editor/        Editor UI: plugins, inspectors, debugger, import/export
  drivers/       Hardware abstraction: Vulkan, GLES3, Metal, D3D12, audio drivers
  platform/      Platform-specific: Windows, Linux, macOS, Android, iOS, Web
  main/          Engine entry point and main loop
  tests/         Unit tests (doctest framework)
  thirdparty/    External dependencies (vendored)
  doc/           Class reference XML files
  misc/          Utility scripts (formatting, validation, CI)
```

## Object System Hierarchy

```
Object                          <- Base of everything. Manual lifetime.
  +-- RefCounted                <- Automatic reference counting via Ref<T>
  |     +-- Resource            <- Loadable/saveable assets (Mesh, Texture, Material...)
  |     +-- Script              <- GDScript, C# scripts
  |     +-- ResourceFormatLoader
  |     +-- EditorInspectorPlugin
  |
  +-- Node                      <- Scene tree participant. Parent owns children.
  |     +-- Node2D              <- 2D transform
  |     +-- Node3D              <- 3D transform
  |     |     +-- VisualInstance3D -> MeshInstance3D, Light3D, Camera3D...
  |     |     +-- CollisionObject3D -> bodies, areas
  |     +-- Control             <- GUI base
  |     |     +-- Button, Label, Container, LineEdit...
  |     +-- Viewport
  |     +-- Window
  |
  +-- MainLoop                  <- Drives the engine loop
        +-- SceneTree           <- The active scene manager
```

## Scene <-> Server Architecture

The engine uses a **scene/server split**. Scene-side nodes hold user-facing state. Server singletons hold the actual rendering/physics/audio data. They communicate via **RID** (Resource ID) handles.

```
 Scene Side (main thread)              Server Side (can be threaded)
 ========================              ============================
 MeshInstance3D                        RenderingServer (singleton)
   mesh: Ref<Mesh>          ------>      instance RID
   set_mesh(mesh)                        instance_set_base(rid, mesh_rid)
   _notification(ENTER_TREE)             instance_set_scenario(rid, world_rid)
   _notification(TRANSFORM)              instance_set_transform(rid, xform)
   ~MeshInstance3D()                     free_rid(rid)
```

### Communication Pattern

1. **Construction**: Scene node creates a server-side resource in constructor
   ```cpp
   VisualInstance3D::VisualInstance3D() {
       instance = RS::get_singleton()->instance_create();
       RS::get_singleton()->instance_attach_object_instance_id(instance, get_instance_id());
   }
   ```

2. **Enter World**: Node registers with server scenario
   ```cpp
   case NOTIFICATION_ENTER_WORLD:
       RS::get_singleton()->instance_set_scenario(instance, get_world_3d()->get_scenario());
   ```

3. **Updates**: Changes forwarded via RID
   ```cpp
   void MeshInstance3D::set_mesh(const Ref<Mesh> &p_mesh) {
       mesh = p_mesh;
       set_base(mesh.is_valid() ? mesh->get_rid() : RID());
   }
   ```

4. **Destruction**: Free server-side resource
   ```cpp
   VisualInstance3D::~VisualInstance3D() {
       RS::get_singleton()->free_rid(instance);
   }
   ```

### Server Singletons

| Server | Shorthand | Purpose |
|--------|-----------|---------|
| `RenderingServer` | `RS` | All rendering (meshes, lights, viewports, shaders) |
| `PhysicsServer3D` | | 3D physics simulation |
| `PhysicsServer2D` | | 2D physics simulation |
| `AudioServer` | | Audio mixing and playback |
| `DisplayServer` | `DS` | Window management, input |
| `NavigationServer3D` | | 3D pathfinding |
| `NavigationServer2D` | | 2D pathfinding |
| `TextServer` | `TS` | Text layout and rendering |
| `XRServer` | | VR/AR tracking |

All follow the same pattern:
```cpp
class RenderingServer : public Object {
    GDCLASS(RenderingServer, Object);
    static RenderingServer *singleton;
public:
    static RenderingServer *get_singleton();
    // Pure virtual interface methods that take RIDs
    virtual void instance_set_transform(RID p_instance, const Transform3D &p_xform) = 0;
};
```

## RID System

RID is a 64-bit opaque handle: `[validator (32-bit) | index (32-bit)]`.

```cpp
class RID {
    uint64_t _id = 0;
public:
    bool is_valid() const { return _id != 0; }
    bool is_null() const { return _id == 0; }
};
```

- Created by `RID_Alloc<T>` with chunked allocation and validator to prevent use-after-free
- Thread-safe allocation available via template parameter
- Zero-copy passing between scene and server (just a uint64)
- `RID()` (null) is used to clear/unset server resources

## Node Lifecycle

```
Construction                    memnew(MyNode)
     |
     v
NOTIFICATION_PARENTED (18)      When added as child (before tree entry)
     |
     v
NOTIFICATION_ENTER_TREE (10)    Entering the SceneTree
     |                          -> Register with servers, resolve paths
     v
NOTIFICATION_POST_ENTER_TREE (27)  After enter_tree completes
     |
     v
NOTIFICATION_READY (13)         All children ready, one-time setup
     |                          -> Called in reverse order (children first)
     v
  [Main Loop]
     |
     +-> NOTIFICATION_PHYSICS_PROCESS (16)    Fixed timestep (60 Hz default)
     +-> NOTIFICATION_PROCESS (17)             Variable timestep (every frame)
     +-> NOTIFICATION_INTERNAL_PHYSICS_PROCESS (26)  Engine-internal physics
     +-> NOTIFICATION_INTERNAL_PROCESS (25)          Engine-internal process
     |
     v
NOTIFICATION_EXIT_TREE (11)     Leaving the SceneTree
     |                          -> Unregister from servers
     v
NOTIFICATION_UNPARENTED (19)    Removed from parent
     |
     v
Destruction                     memdelete / queue_free
```

### _notification() Pattern
```cpp
void MyNode::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            // Setup: connect signals, register with servers
        } break;
        case NOTIFICATION_READY: {
            // One-time init after all children are ready
        } break;
        case NOTIFICATION_PROCESS: {
            GDVIRTUAL_CALL(_process, get_process_delta_time());
        } break;
        case NOTIFICATION_EXIT_TREE: {
            // Cleanup: disconnect, unregister
        } break;
    }
}
```

### Propagation Order
- `ENTER_TREE`: parent first, then children (top-down)
- `READY`: children first, then parent (bottom-up)
- `EXIT_TREE`: children first, then parent (bottom-up)

## Scene Tree Process Pipeline

### Every Physics Tick (fixed, 60 Hz default)
```
SceneTree::physics_process(delta)
  1. flush_transform_notifications()    -> sync transforms to servers
  2. MainLoop::physics_process()
  3. emit_signal("physics_frame")
  4. _process_picking()                 -> input raycasts
  5. _process(true)                     -> NOTIFICATION_PHYSICS_PROCESS to all nodes
  6. process_timers()
  7. process_tweens()
  8. flush_transform_notifications()
  9. _flush_delete_queue()              -> free queue_free'd nodes
```

### Every Frame (variable rate)
```
SceneTree::process(delta)
  1. Fixed timestep interpolation (FTI) update
  2. MainLoop::process()
  3. multiplayer->poll()
  4. emit_signal("process_frame")
  5. _process(false)                    -> NOTIFICATION_PROCESS to all nodes
  6. process_timers()
  7. process_tweens()
  8. flush_transform_notifications()
  9. _flush_delete_queue()
```

### Thread Groups
Nodes can be assigned to process thread groups:
- `PROCESS_THREAD_GROUP_INHERIT` (default) — inherit from parent
- `PROCESS_THREAD_GROUP_MAIN_THREAD` — always main thread
- `PROCESS_THREAD_GROUP_SUB_THREAD` — processed on WorkerThreadPool

Groups with sub-thread processing run in parallel via `WorkerThreadPool::add_template_group_task()`.

## Resource System

```
Resource : RefCounted
  - Automatic lifetime via Ref<T>
  - Unique path: "res://path/to/resource.tres"
  - Persistent UID via ResourceUID (int64)
  - Cached by ResourceLoader

Loading:
  Ref<Mesh> mesh = ResourceLoader::load("res://mesh.tres");

Saving:
  ResourceSaver::save(mesh, "res://mesh.tres");

Custom Loaders:
  class MyLoader : public ResourceFormatLoader {
      GDCLASS(MyLoader, ResourceFormatLoader);
      GDVIRTUAL4RC_REQUIRED(Variant, _load, String, String, bool, int)
      // ... implement virtual methods
  };
```

### ResourceFormatLoader Pipeline
1. `get_recognized_extensions()` — which file extensions this loader handles
2. `recognize_path()` / `handles_type()` — can this loader handle this path/type?
3. `load()` — actually load the resource, with caching support
4. Cache modes: `IGNORE`, `REUSE`, `REPLACE`, `IGNORE_DEEP`, `REPLACE_DEEP`

## Module System

### Required Files
```
modules/my_module/
  config.py              # Build configuration
  register_types.h       # Registration declarations
  register_types.cpp     # Registration implementation
  SCsub                  # Build file
```

### Initialization Levels (order matters)
```
MODULE_INITIALIZATION_LEVEL_CORE      # Core types, no scene/server deps
MODULE_INITIALIZATION_LEVEL_SERVERS   # Server-level types
MODULE_INITIALIZATION_LEVEL_SCENE     # Scene-level types (most classes here)
MODULE_INITIALIZATION_LEVEL_EDITOR    # Editor plugins (ifdef TOOLS_ENABLED)
```

### Registration Pattern
```cpp
void initialize_my_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        GDREGISTER_CLASS(MyClass);
        GDREGISTER_ABSTRACT_CLASS(MyBaseClass);
    }
#ifdef TOOLS_ENABLED
    if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
        EditorPlugins::add_by_type<MyEditorPlugin>();
    }
#endif
}
```

## Code Generation

Python scripts generate `.gen.h`/`.gen.cpp` files during build:

| Generator | Output | Purpose |
|-----------|--------|---------|
| `core/object/make_virtuals.py` | `gdvirtual.gen.h` (515KB) | GDVIRTUAL macros |
| `core/core_builders.py` | `version_generated.gen.h`, `disabled_classes.gen.h`, etc. | Version info, config |
| `editor/editor_builders.py` | `register_exporters.gen.cpp`, doc paths | Editor integration |
| `gles3_builders.py` | `*.glsl.gen.h` | GLSL shaders baked into C++ |
| `glsl_builders.py` | `*.glsl.gen.h` | RD shader compilation |
| Build system | `register_module_types.gen.cpp` | Master module registration |
| Build system | `modules_enabled.gen.h` | Module enable flags |

**Never edit `.gen.*` files.** They are overwritten on every build.
