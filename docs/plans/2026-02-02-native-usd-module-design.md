# Native USD Module for Godot Engine - Design Document

**Date**: 2026-02-02
**Branch**: `feature/usd-support`
**Status**: Design Phase

---

## 1. Executive Summary

Add native Universal Scene Description (USD) import support to Godot Engine as a built-in module (`modules/usd/`), following the same architectural patterns as the existing glTF and FBX modules. This addresses the #1 community-requested 3D format (60+ upvotes on godot-proposals#7744) and enables production pipeline integration with Blender, Houdini, Maya, and other DCC tools.

**Scope**: Phase 1 focuses on **import-only** support for `.usd`, `.usda`, `.usdc`, and `.usdz` files, converting USD scenes into Godot's native scene tree. Export and live-stage features are deferred to future phases.

---

## 2. Architecture Overview

### 2.1 Module Structure

```
modules/usd/
├── config.py                    # Module build configuration
├── register_types.cpp/h         # Module registration (SCENE + EDITOR levels)
├── SCsub                        # SCons build script
├── usd_document.cpp/h           # Core USD document handler (main parsing logic)
├── usd_state.cpp/h              # Import state machine (stores parsed data)
├── usd_defines.h                # Type definitions, index types, constants
├── structures/                  # Data model classes
│   ├── usd_node.cpp/h           # Hierarchical node representation
│   ├── usd_mesh.cpp/h           # Mesh data wrapper
│   ├── usd_material.cpp/h       # Material data wrapper
│   ├── usd_skeleton.cpp/h       # Skeleton/skin data wrapper
│   ├── usd_animation.cpp/h      # Animation data wrapper
│   ├── usd_light.cpp/h          # Light data wrapper
│   ├── usd_camera.cpp/h         # Camera data wrapper
│   └── SCsub
├── editor/
│   ├── editor_scene_importer_usd.cpp/h  # EditorSceneFormatImporter implementation
│   └── SCsub
├── thirdparty/                  # tinyusdz library (header-only/minimal dependency)
│   └── README.md                # Build instructions for OpenUSD alternative
└── doc_classes/                 # XML documentation for all exported classes
    ├── USDDocument.xml
    └── USDState.xml
```

### 2.2 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **USD Library** | tinyusdz (primary), OpenUSD (optional) | tinyusdz is dependency-free C++14, fits Godot's thirdparty model. OpenUSD requires TBB and is gigabytes. Calinou (Godot maintainer) specifically suggested tinyusdz. |
| **Integration Pattern** | Follow FBX module pattern | FBX wraps external C library + extends glTF document architecture. Proven, maintainable pattern. |
| **Import Strategy** | Convert to Godot scene tree at import time | Matches Godot convention (like glTF/FBX). Simpler than live-stage approach. |
| **Data Flow** | USD → USDState → ImporterMesh/Node3D scene tree | Same pipeline as glTF: parse → intermediate state → generate scene. |
| **Coordinate System** | Z-up → Y-up conversion at import | USD uses Z-up by default; Godot uses Y-up. Apply rotation during import. |
| **Material Mapping** | UsdPreviewSurface → StandardMaterial3D | Direct PBR parameter mapping (diffuseColor→albedo, metallic→metallic, etc.) |

### 2.3 Data Flow

```
.usd/.usda/.usdc/.usdz file
        │
        ▼
┌─────────────────────┐
│  EditorSceneFormat   │  Entry point: recognized extensions, import flags
│  ImporterUSD         │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│    USDDocument       │  Core parser: reads USD via tinyusdz
│    ::append_from_    │  Populates USDState with parsed data
│    file()            │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│    USDState          │  Intermediate representation:
│    (meshes, nodes,   │  - Vector<USDNode> nodes
│     materials,       │  - Vector<USDMesh> meshes
│     skeletons,       │  - Vector<Material> materials
│     animations,      │  - Vector<USDSkeleton> skeletons
│     cameras, lights) │  - Vector<USDAnimation> animations
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│    USDDocument       │  Scene generation:
│    ::generate_       │  USDNode → Node3D / ImporterMeshInstance3D
│    scene()           │  USDMesh → ImporterMesh → ArrayMesh
│                      │  USDMaterial → StandardMaterial3D
│                      │  USDSkeleton → Skeleton3D + Skin
│                      │  USDAnimation → Animation + AnimationPlayer
│                      │  USDLight → Light3D (OmniLight3D/SpotLight3D/DirectionalLight3D)
│                      │  USDCamera → Camera3D
└─────────┬───────────┘
          ▼
     Godot Scene Tree (Node3D hierarchy)
```

---

## 3. Component Design

### 3.1 USDState (Import State Machine)

```cpp
class USDState : public Resource {
    GDCLASS(USDState, Resource);

    // File metadata
    String base_path;
    String filename;

    // Scene structure
    Vector<Ref<USDNode>> nodes;
    Vector<int> root_nodes;
    String scene_name;

    // Resources parsed from USD
    Vector<Ref<USDMesh>> meshes;
    Vector<Ref<Material>> materials;
    Vector<Ref<Texture2D>> images;
    Vector<Ref<USDSkeleton>> skeletons;
    Vector<Ref<USDAnimation>> animations;
    Vector<Ref<USDCamera>> cameras;
    Vector<Ref<USDLight>> lights;

    // Node-to-Godot mapping (populated during generation)
    HashMap<int, Node *> scene_nodes;
    HashMap<int, ImporterMeshInstance3D *> scene_mesh_instances;

    // Import options
    double bake_fps = 30.0;
    bool create_animations = true;
    bool force_generate_tangents = false;
};
```

### 3.2 USDDocument (Core Parser)

```cpp
class USDDocument : public Resource {
    GDCLASS(USDDocument, Resource);

public:
    // Import pipeline
    Error append_from_file(const String &p_path, Ref<USDState> p_state,
                          uint32_t p_flags, const String &p_base_path = "");
    Node *generate_scene(Ref<USDState> p_state, float p_bake_fps = 30.0,
                        bool p_trimming = false, bool p_remove_immutable_tracks = true);

private:
    // Parsing stages (populate USDState)
    Error _parse_scene(Ref<USDState> p_state, const String &p_path);
    Error _parse_nodes(Ref<USDState> p_state);       // UsdGeomXform hierarchy
    Error _parse_meshes(Ref<USDState> p_state);       // UsdGeomMesh
    Error _parse_materials(Ref<USDState> p_state);     // UsdShadeMaterial + UsdPreviewSurface
    Error _parse_skeletons(Ref<USDState> p_state);     // UsdSkelSkeleton
    Error _parse_animations(Ref<USDState> p_state);    // UsdSkelAnimation + time-sampled xforms
    Error _parse_cameras(Ref<USDState> p_state);       // UsdGeomCamera
    Error _parse_lights(Ref<USDState> p_state);        // UsdLux*

    // Scene generation (USDState → Godot nodes)
    Node3D *_generate_node(Ref<USDState> p_state, int p_node_idx, Node *p_parent);
    ImporterMeshInstance3D *_generate_mesh_instance(Ref<USDState> p_state, int p_mesh_idx);
    Skeleton3D *_generate_skeleton(Ref<USDState> p_state, int p_skel_idx);
    Camera3D *_generate_camera(Ref<USDState> p_state, int p_camera_idx);
    Light3D *_generate_light(Ref<USDState> p_state, int p_light_idx);
    void _generate_animations(Ref<USDState> p_state, Node *p_root);

    // Material conversion
    Ref<StandardMaterial3D> _convert_usd_preview_surface(Ref<USDState> p_state, int p_mat_idx);
    Ref<Texture2D> _load_texture(Ref<USDState> p_state, const String &p_asset_path);

    // Coordinate system conversion
    static Transform3D _convert_usd_transform(const Transform3D &p_usd_transform);
    static Vector3 _convert_usd_position(const Vector3 &p_pos);
};
```

### 3.3 Data Structures

```cpp
// modules/usd/structures/usd_node.h
class USDNode : public Resource {
    String name;
    int parent = -1;
    Vector<int> children;
    Transform3D xform;        // Local transform (already Y-up converted)
    int mesh = -1;            // Index into USDState::meshes (-1 = no mesh)
    int camera = -1;
    int light = -1;
    int skeleton = -1;
    int skin = -1;
    bool visible = true;
    Dictionary additional_data;  // For extensions
};

// modules/usd/structures/usd_mesh.h
class USDMesh : public Resource {
    struct Surface {
        Array arrays;                           // Godot vertex arrays
        Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
        int material = -1;                      // Index into USDState::materials
        String name;
        Vector<Array> blend_shape_arrays;
    };
    Vector<Surface> surfaces;
    Vector<String> blend_shapes;
    // Instance support: if this mesh is used by multiple nodes, they share the same USDMesh
};

// modules/usd/structures/usd_material.h
class USDMaterial : public Resource {
    String name;
    // UsdPreviewSurface parameters
    Color diffuse_color = Color(0.18, 0.18, 0.18);
    float metallic = 0.0;
    float roughness = 0.5;
    Color emissive_color = Color(0, 0, 0);
    float opacity = 1.0;
    float ior = 1.5;
    float clearcoat = 0.0;
    float clearcoat_roughness = 0.01;
    // Texture paths (resolved relative to USD file)
    String diffuse_texture;
    String metallic_texture;
    String roughness_texture;
    String normal_texture;
    String emissive_texture;
    String occlusion_texture;
    String opacity_texture;
};

// modules/usd/structures/usd_skeleton.h
class USDSkeleton : public Resource {
    Vector<String> joint_paths;     // Joint topology (ordered)
    Vector<int> joint_parents;      // Parent indices
    Vector<Transform3D> bind_transforms;  // World-space bind pose
    Vector<Transform3D> rest_transforms;  // Joint-local rest pose
};

// modules/usd/structures/usd_animation.h
class USDAnimation : public Resource {
    String name;
    struct Track {
        String target_path;  // Node path in scene
        enum Type { POSITION, ROTATION, SCALE, BLEND_SHAPE, JOINT_TRANSFORM };
        Type type;
        Vector<double> times;
        Vector<Variant> values;  // Vector3 for pos/scale, Quaternion for rot, float for blend
    };
    Vector<Track> tracks;
    double start_time = 0.0;
    double end_time = 0.0;
    double time_codes_per_second = 24.0;
};

// modules/usd/structures/usd_light.h
class USDLight : public Resource {
    enum LightType { DISTANT, SPHERE, DISK, RECT, CYLINDER, DOME };
    LightType type = SPHERE;
    Color color = Color(1, 1, 1);
    float intensity = 1.0;
    float exposure = 0.0;
    float radius = 0.5;          // For sphere/disk/cylinder
    float width = 1.0;           // For rect
    float height = 1.0;          // For rect
    float cone_angle = 90.0;     // For shaping (spot-light behavior)
    float cone_softness = 0.0;
    bool cast_shadows = true;
    String dome_texture;          // For dome lights (IBL)
};

// modules/usd/structures/usd_camera.h
class USDCamera : public Resource {
    enum ProjectionType { PERSPECTIVE, ORTHOGRAPHIC };
    ProjectionType projection = PERSPECTIVE;
    float focal_length = 50.0;   // mm
    float horizontal_aperture = 20.955;  // mm (default from USD)
    float vertical_aperture = 15.2908;
    float near_clip = 1.0;
    float far_clip = 1000000.0;
};
```

### 3.4 EditorSceneFormatImporterUSD

```cpp
class EditorSceneFormatImporterUSD : public EditorSceneFormatImporter {
    GDCLASS(EditorSceneFormatImporterUSD, EditorSceneFormatImporter);

public:
    void get_extensions(List<String> *r_extensions) const override;
    // Returns: "usd", "usda", "usdc", "usdz"

    Node *import_scene(const String &p_path, uint32_t p_flags,
                      const HashMap<StringName, Variant> &p_options,
                      List<String> *r_missing_deps,
                      Error *r_err = nullptr) override;

    void get_import_options(const String &p_path,
                           List<ResourceImporter::ImportOption> *r_options) override;
    // Options: bake_fps, generate_tangents, import_materials, import_animations,
    //          import_lights, import_cameras, coordinate_system (Z-up/Y-up auto)
};
```

---

## 4. USD → Godot Mapping

### 4.1 Node Types

| USD Type | Godot Type |
|----------|-----------|
| UsdGeomXform | Node3D |
| UsdGeomMesh | ImporterMeshInstance3D → MeshInstance3D |
| UsdGeomCamera | Camera3D |
| UsdLuxDistantLight | DirectionalLight3D |
| UsdLuxSphereLight (treatAsPoint) | OmniLight3D |
| UsdLuxSphereLight (shaping cone) | SpotLight3D |
| UsdLuxRectLight / UsdLuxDiskLight | OmniLight3D (approximation) |
| UsdLuxDomeLight | WorldEnvironment + Sky |
| UsdSkelRoot + UsdSkelSkeleton | Skeleton3D |
| UsdGeomPointInstancer | MultiMeshInstance3D |
| UsdGeomBasisCurves | (deferred - no direct Godot equivalent) |
| UsdGeomPoints | (deferred - GPUParticles3D or custom mesh) |

### 4.2 Material Mapping (UsdPreviewSurface → StandardMaterial3D)

| USD Parameter | Godot Property | Notes |
|--------------|---------------|-------|
| diffuseColor | albedo_color | Direct mapping |
| metallic | metallic | Direct mapping |
| roughness | roughness | Direct mapping |
| emissiveColor | emission | Direct mapping |
| opacity | albedo_color.a | Sets transparency mode |
| ior | refraction_index | When using refraction |
| clearcoat | clearcoat | Direct mapping |
| clearcoatRoughness | clearcoat_roughness | Direct mapping |
| normal | normal_map | Direct mapping |
| occlusion | ao_texture | Direct mapping |
| displacement | (deferred) | Godot has limited displacement support |
| specularColor | (approximated) | Converted to metallic workflow |
| useSpecularWorkflow | (handled) | Convert specular → metallic if needed |

### 4.3 Coordinate System

USD default is **Z-up, right-handed**. Godot is **Y-up, right-handed**.

Conversion at import:
- Rotate scene root by -90 degrees around X axis
- OR per-prim transform conversion: swap Y↔Z, negate as needed
- Respect `upAxis` metadata in USD stage (can be "Y" already)

### 4.4 Units

USD uses **centimeters** by default (`metersPerUnit = 0.01`). Godot uses **meters**.
Apply scale factor: `godot_position = usd_position * metersPerUnit`

---

## 5. Build System Integration

### 5.1 config.py

```python
def can_build(env, platform):
    return not env["disable_3d"]

def configure(env):
    pass

def get_doc_classes():
    return [
        "USDDocument",
        "USDState",
        "USDNode",
        "USDMesh",
        "USDMaterial",
        "USDSkeleton",
        "USDAnimation",
        "USDLight",
        "USDCamera",
        "EditorSceneFormatImporterUSD",
    ]

def get_doc_path():
    return "doc_classes"
```

### 5.2 SCsub

```python
#!/usr/bin/env python
from misc.utility.scons_hints import *

Import("env")
Import("env_modules")

env_usd = env_modules.Clone()

# tinyusdz thirdparty source
thirdparty_obj = []
thirdparty_dir = "#thirdparty/tinyusdz/"
thirdparty_sources = [thirdparty_dir + "tinyusdz.cc"]

env_usd.Prepend(CPPPATH=[thirdparty_dir])

env_thirdparty = env_usd.Clone()
env_thirdparty.disable_warnings()
env_thirdparty.add_source_files(thirdparty_obj, thirdparty_sources)
env.modules_sources += thirdparty_obj

# Module source files
module_obj = []
env_usd.add_source_files(module_obj, "*.cpp")

SConscript("structures/SCsub")

if env.editor_build:
    SConscript("editor/SCsub")

env.modules_sources += module_obj
env.Depends(module_obj, thirdparty_obj)
```

### 5.3 tinyusdz Integration

[tinyusdz](https://github.com/syoyo/tinyusdz) is a dependency-free C++14 USD reader/writer:
- Single-file compilation (`tinyusdz.cc` + `tinyusdz.hh`)
- Reads `.usda`, `.usdc`, `.usdz` without external dependencies
- No TBB, no Boost, no Python required
- Fits Godot's `thirdparty/` vendor pattern perfectly
- Supports UsdGeom, UsdShade (UsdPreviewSurface), UsdSkel, UsdLux schemas

---

## 6. Implementation Phases

### Phase 1: Core Import (This PR)
- Module skeleton (config.py, register_types, SCsub)
- tinyusdz integration in `thirdparty/`
- USDDocument + USDState core classes
- Mesh import (UsdGeomMesh → ImporterMesh)
- Transform hierarchy (UsdGeomXform → Node3D)
- Material import (UsdPreviewSurface → StandardMaterial3D)
- Texture loading
- Coordinate system conversion (Z-up → Y-up)
- Unit conversion (cm → m)
- EditorSceneFormatImporterUSD registration
- Basic import options

### Phase 2: Extended Import (Future)
- Skeletal animation (UsdSkel → Skeleton3D + AnimationPlayer)
- Transform animation (time-sampled xforms → Animation tracks)
- Light import (UsdLux → Light3D variants)
- Camera import (UsdGeomCamera → Camera3D)
- Point instancing (UsdGeomPointInstancer → MultiMeshInstance3D)
- Variant selection in import options
- Payload handling

### Phase 3: Export (Future)
- USDDocument::append_from_scene() + write_to_filesystem()
- Scene tree → USD prim hierarchy
- StandardMaterial3D → UsdPreviewSurface
- Mesh/Skeleton/Animation export

### Phase 4: Advanced (Future)
- USD composition arcs (references, payloads)
- MaterialX support
- Subdivision surface evaluation
- Basis curves
- USD layer editing integration

---

## 7. Testing Strategy

- Unit tests for coordinate/unit conversion functions
- Import tests with reference USD files (Kitchen Set, simple geometry, animated characters)
- Material mapping verification (diffuse, metallic, roughness, normal maps)
- Skeleton/animation import correctness
- Memory and performance benchmarks for large scenes
- Comparison with existing GodotOpenUSD addon output
