# Godot Engine Runtime Patterns

How objects communicate, respond to changes, and interact at runtime in C++.

---

## 1. Signal Connections with callable_mp

### The Core Pattern

`callable_mp(object, &Class::method)` creates a type-safe Callable from a C++ method pointer. This is the **primary** way to connect signals in C++.

```cpp
// Connect a signal
mesh->connect_changed(callable_mp(this, &MeshInstance3D::_mesh_changed));

// Equivalent using signal name
mesh->connect(SNAME("changed"), callable_mp(this, &MeshInstance3D::_mesh_changed));

// Using SceneStringName for common signals
window->connect(SceneStringName(visibility_changed),
    callable_mp(this, &CanvasItem::_window_visibility_changed));
```

### Disconnect Pattern

**Always check before disconnecting** to avoid errors:
```cpp
if (mesh.is_valid() && mesh->is_connected(SNAME("changed"),
        callable_mp(this, &MeshInstance3D::_mesh_changed))) {
    mesh->disconnect(SNAME("changed"),
        callable_mp(this, &MeshInstance3D::_mesh_changed));
}
```

Or more concisely when replacing a value:
```cpp
void MyNode::set_mesh(const Ref<Mesh> &p_mesh) {
    if (mesh == p_mesh) {
        return;
    }
    // Disconnect old
    if (mesh.is_valid()) {
        mesh->disconnect_changed(callable_mp(this, &MyNode::_mesh_changed));
    }
    mesh = p_mesh;
    // Connect new
    if (mesh.is_valid()) {
        mesh->connect_changed(callable_mp(this, &MyNode::_mesh_changed));
    }
}
```

### Binding Arguments

Use `.bind()` to capture additional arguments:
```cpp
// Bind an extra argument to the callback
parent->connect(SNAME("child_order_changed"),
    callable_mp(viewport, &Viewport::canvas_parent_mark_dirty).bind(parent));

// Bind with connection flags
parent->connect(SNAME("child_order_changed"),
    callable_mp(viewport, &Viewport::canvas_parent_mark_dirty).bind(parent),
    CONNECT_REFERENCE_COUNTED);
```

### Deferred Calls

Use `.call_deferred()` to defer execution to the main thread:
```cpp
// Defer a method call
callable_mp(this, &ThemeDB::_sort_theme_items).call_deferred();

// Defer with arguments
callable_mp(this, &ProjectSettings::_emit_changed).call_deferred();

// Defer from a worker thread (thread-safe way to modify scene tree)
callable_mp(node, &Node::add_child).call_deferred(new_child, false, 0);
```

### callable_mp vs Callable

| | `callable_mp` | `Callable` |
|---|---|---|
| **Type safety** | Compile-time checked | Runtime resolved |
| **Use case** | C++ signal connections, deferred calls | GDScript interop, dynamic dispatch |
| **Syntax** | `callable_mp(obj, &Class::method)` | `Callable(obj, "method_name")` |
| **Performance** | Direct function pointer | Virtual dispatch |

**Rule:** Always prefer `callable_mp` in C++ code. Use `Callable(obj, "method")` only when the method name comes from script or is dynamic.

### Static Method Variant

```cpp
// callable_mp_static for free/static functions
callable_mp_static(&MyClass::static_method);
```

### Connection Flags

```cpp
CONNECT_DEFERRED          // Call on next frame, not immediately
CONNECT_ONE_SHOT          // Auto-disconnect after first call
CONNECT_REFERENCE_COUNTED // Track reference count (prevents dangling)
```

---

## 2. Dynamic Properties

### _get / _set / _get_property_list

Override these for properties that aren't statically defined (e.g., properties that depend on runtime state).

**Header:**
```cpp
class MyNode : public Node {
    GDCLASS(MyNode, Node);

protected:
    bool _set(const StringName &p_name, const Variant &p_value);
    bool _get(const StringName &p_name, Variant &r_ret) const;
    void _get_property_list(List<PropertyInfo> *p_list) const;
```

**Implementation:**
```cpp
bool MyNode::_set(const StringName &p_name, const Variant &p_value) {
    // Handle "parameters/my_param" style properties
    if (p_name.begins_with("parameters/")) {
        String param_name = p_name.get_slicec('/', 1);
        parameters[param_name] = p_value;
        return true;  // Return true = property was handled
    }
    return false;  // Return false = not our property, pass to parent
}

bool MyNode::_get(const StringName &p_name, Variant &r_ret) const {
    if (p_name.begins_with("parameters/")) {
        String param_name = p_name.get_slicec('/', 1);
        if (parameters.has(param_name)) {
            r_ret = parameters[param_name];
            return true;
        }
    }
    return false;
}

void MyNode::_get_property_list(List<PropertyInfo> *p_list) const {
    // Add dynamic properties to the inspector
    for (const KeyValue<String, Variant> &E : parameters) {
        p_list->push_back(PropertyInfo(Variant::FLOAT, "parameters/" + E.key,
            PROPERTY_HINT_RANGE, "0,1,0.01"));
    }
}
```

### _validate_property

Override to modify property hints/visibility dynamically in the inspector:

```cpp
void MyNode::_validate_property(PropertyInfo &p_property) const {
    // Make a property read-only when playing
    if (p_property.name == "frame" && is_playing) {
        p_property.usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY;
    }

    // Build enum hint dynamically
    if (p_property.name == "animation") {
        p_property.hint = PROPERTY_HINT_ENUM;
        p_property.hint_string = ""; // Build from available animations
        for (const StringName &name : animation_list) {
            if (!p_property.hint_string.is_empty()) {
                p_property.hint_string += ",";
            }
            p_property.hint_string += String(name);
        }
    }

    // Hide a property based on mode
    if (p_property.name == "advanced_option" && mode != MODE_ADVANCED) {
        p_property.usage = PROPERTY_USAGE_NO_EDITOR;
    }
}
```

### _property_can_revert / _property_get_revert

Provide "revert to default" in the inspector:

```cpp
bool MyNode::_property_can_revert(const StringName &p_name) const {
    if (p_name == "speed") {
        return true;  // Show revert arrow in inspector
    }
    return false;
}

bool MyNode::_property_get_revert(const StringName &p_name, Variant &r_property) const {
    if (p_name == "speed") {
        r_property = 1.0;  // Default value to revert to
        return true;
    }
    return false;
}
```

---

## 3. Change Notification

### emit_changed() — Resource data changed

Call in Resource setters when the **data content** changes:
```cpp
void MyResource::set_value(float p_value) {
    if (value == p_value) {
        return;  // No change, skip notification
    }
    value = p_value;
    emit_changed();  // Notifies all dependents (materials, meshes, etc.)
}
```

**When to call:**
- After modifying any data that affects how the resource is used
- After adding/removing tracks, entries, items
- NOT for metadata-only changes (name, path)

### notify_property_list_changed() — Property structure changed

Call when the **set of visible properties** changes:
```cpp
void MyNode::set_mode(Mode p_mode) {
    mode = p_mode;
    notify_property_list_changed();  // Inspector will re-query _get_property_list
}

void MyResource::add_item(const String &p_name) {
    items[p_name] = default_value;
    notify_property_list_changed();  // New property appears in inspector
    emit_changed();                   // Data also changed
}
```

### update_gizmos() — Visual editor handles changed

```cpp
void MyNode3D::set_size(float p_size) {
    size = p_size;
    update_gizmos();  // Refresh editor gizmo display
}
```

---

## 4. Type Casting

### Object::cast_to<T>()

The safe downcast pattern in Godot:
```cpp
// Always null-check after cast
Node3D *n3d = Object::cast_to<Node3D>(p_object);
ERR_FAIL_NULL(n3d);
n3d->set_position(Vector3());

// In conditional context
if (Node3D *n3d = Object::cast_to<Node3D>(p_object)) {
    n3d->set_position(Vector3());
}

// Multiple cast attempts
if (MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(node)) {
    // Handle mesh instance
} else if (Light3D *light = Object::cast_to<Light3D>(node)) {
    // Handle light
}
```

**Never** use `dynamic_cast` or C-style casts. `Object::cast_to` uses Godot's own RTTI which is faster and works with the class registration system.

---

## 5. Conditional Compilation Guards

### Guard Reference

```cpp
#ifdef TOOLS_ENABLED      // Editor build only (not export templates)
    // Gizmos, inspector helpers, editor-only methods
    // _edit_set_state(), _edit_get_state()
    // EditorPlugin code
    // Configuration warnings
#endif

#ifdef DEBUG_ENABLED       // Debug builds (editor + debug export)
    // Extra validation, debug drawing
    // Detailed error messages
#endif

#ifdef DEV_ENABLED         // Developer builds (extra paranoia)
    // DEV_ASSERT checks
    // Internal consistency checks
#endif

#ifndef DISABLE_DEPRECATED // Backward compatibility code
    // .compat.inc includes
    // Old method signatures
#endif

#ifndef PHYSICS_3D_DISABLED  // 3D physics available
    // PhysicsServer3D usage
    // 3D collision/body nodes
#endif

#ifndef PHYSICS_2D_DISABLED  // 2D physics available
    // PhysicsServer2D usage
    // 2D collision/body nodes
#endif

// Combined guard for shared physics code
#if !defined(PHYSICS_2D_DISABLED) || !defined(PHYSICS_3D_DISABLED)
    // PhysicsMaterial and shared physics types
#endif
```

### What Goes Where

| Guard | Contents |
|-------|----------|
| `TOOLS_ENABLED` | Editor gizmos, `_edit_*` methods, export plugins, inspector customization, configuration warnings, editor-only signals |
| `DEBUG_ENABLED` | Extra validation in hot paths, debug visualization, verbose error messages |
| `DEV_ENABLED` | Internal assertions (`DEV_ASSERT`), consistency checks that are too expensive for normal debug builds |
| `DISABLE_DEPRECATED` | `.compat.inc` includes, old method wrappers. Code inside `#ifndef DISABLE_DEPRECATED` provides backward compat |

---

## 6. Editor Undo/Redo

### Basic Pattern

Every editor action that modifies scene state must be undoable:

```cpp
void MyEditorPlugin::_do_something(Node *p_node) {
    EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

    // 1. Start action with human-readable name (use TTR for translation)
    undo_redo->create_action(TTR("Do Something"));

    // 2. Record the "do" operation
    undo_redo->add_do_method(p_node, "set_position", new_position);
    undo_redo->add_do_property(p_node, "visible", true);

    // 3. Record the "undo" operation (current state)
    undo_redo->add_undo_method(p_node, "set_position", p_node->get_position());
    undo_redo->add_undo_property(p_node, "visible", p_node->is_visible());

    // 4. Optionally request redraw
    undo_redo->add_do_method(viewport, "queue_redraw");
    undo_redo->add_undo_method(viewport, "queue_redraw");

    // 5. Commit (executes the "do" methods immediately)
    undo_redo->commit_action();
}
```

### Batch Operations

```cpp
undo_redo->create_action(TTR("Move Nodes"));
for (Node *node : selected_nodes) {
    CanvasItem *ci = Object::cast_to<CanvasItem>(node);
    if (!ci) {
        continue;
    }
    undo_redo->add_do_method(ci, "set_position", new_positions[ci]);
    undo_redo->add_undo_method(ci, "set_position", ci->get_position());
}
undo_redo->commit_action();
```

### Methods Available

```cpp
undo_redo->add_do_method(obj, "method", args...);    // Call method on do
undo_redo->add_undo_method(obj, "method", args...);  // Call method on undo
undo_redo->add_do_property(obj, "prop", value);       // Set property on do
undo_redo->add_undo_property(obj, "prop", value);     // Set property on undo
undo_redo->add_do_reference(obj);                      // Prevent deletion during undo
undo_redo->add_undo_reference(obj);                    // Prevent deletion during redo
```

---

## 7. Common connect/disconnect Lifecycle

### In _notification

The typical pattern for connecting signals during tree lifecycle:

```cpp
void MyNode::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            // Connect to parent/viewport/tree signals
            get_viewport()->connect(SNAME("size_changed"),
                callable_mp(this, &MyNode::_viewport_resized));
        } break;

        case NOTIFICATION_EXIT_TREE: {
            // Disconnect (cleanup)
            if (get_viewport()) {
                get_viewport()->disconnect(SNAME("size_changed"),
                    callable_mp(this, &MyNode::_viewport_resized));
            }
        } break;
    }
}
```

### In Setters (Resource connections)

```cpp
void MyNode::set_texture(const Ref<Texture2D> &p_texture) {
    if (texture == p_texture) {
        return;
    }

    // Disconnect from old resource
    if (texture.is_valid()) {
        texture->disconnect_changed(callable_mp(this, &MyNode::_texture_changed));
    }

    texture = p_texture;

    // Connect to new resource
    if (texture.is_valid()) {
        texture->connect_changed(callable_mp(this, &MyNode::_texture_changed));
    }

    queue_redraw();  // or update_gizmos() for 3D
}
```

### In Constructor/Destructor

For owned child nodes:

```cpp
MyNode::MyNode() {
    timer = memnew(Timer);
    timer->connect("timeout", callable_mp(this, &MyNode::_on_timeout));
    add_child(timer, false, INTERNAL_MODE_FRONT);
}

// No need to disconnect in destructor — child is freed with parent
```

---

## 8. queue_redraw and Deferred Visual Updates

### 2D Nodes
```cpp
void MyControl::set_color(const Color &p_color) {
    color = p_color;
    queue_redraw();  // Triggers NOTIFICATION_DRAW on next frame
}

void MyControl::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_DRAW: {
            draw_rect(Rect2(Point2(), get_size()), color);
        } break;
    }
}
```

### 3D Nodes
```cpp
void MyNode3D::set_size(float p_size) {
    size = p_size;
    // For visual changes, update the server-side resource
    RS::get_singleton()->instance_set_custom_aabb(get_instance(), AABB(...));
    // For editor gizmos
    update_gizmos();
}
```

---

## 9. Configuration Warnings

Provide editor warnings when a node is misconfigured:

```cpp
// In header:
GDVIRTUAL0RC(Vector<String>, _get_configuration_warnings)

// In source (override from Node):
PackedStringArray MyNode::get_configuration_warnings() const {
    PackedStringArray warnings = Node::get_configuration_warnings();

    if (mesh.is_null()) {
        warnings.push_back(RTR("No mesh assigned. This node will not display anything."));
    }

    if (!get_parent() || !Object::cast_to<Node3D>(get_parent())) {
        warnings.push_back(RTR("This node must be a child of a Node3D."));
    }

    return warnings;
}
```

Call `update_configuration_warnings()` when the condition changes:
```cpp
void MyNode::set_mesh(const Ref<Mesh> &p_mesh) {
    mesh = p_mesh;
    update_configuration_warnings();  // Refresh yellow triangle in editor
}
```
