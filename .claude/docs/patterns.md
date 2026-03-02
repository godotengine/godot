# Godot Engine Code Patterns

Copy-paste-ready templates for common tasks. All patterns are derived from existing code.

---

## 1. Adding a New Node Type

### Header (`scene/3d/my_node.h`)
```cpp
/**************************************************************************/
/*  my_node.h                                                             */
/**************************************************************************/
/* ... copyright block ... */
/**************************************************************************/

#pragma once

#include "scene/3d/node_3d.h"

class MyNode : public Node3D {
	GDCLASS(MyNode, Node3D);

public:
	enum MyEnum {
		MODE_A,
		MODE_B,
		MODE_C,
	};

private:
	float speed = 1.0;
	MyEnum mode = MODE_A;
	Ref<Mesh> mesh;

	void _mesh_changed();
	void _update_internal();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_speed(float p_speed);
	float get_speed() const;

	void set_mode(MyEnum p_mode);
	MyEnum get_mode() const;

	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	GDVIRTUAL1(_on_update, double)

	MyNode();
	~MyNode();
};

VARIANT_ENUM_CAST(MyNode::MyEnum);
```

### Source (`scene/3d/my_node.cpp`)
```cpp
/**************************************************************************/
/*  my_node.cpp                                                           */
/**************************************************************************/
/* ... copyright block ... */
/**************************************************************************/

#include "my_node.h"

void MyNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Connect to viewport/tree signals here
		} break;
		case NOTIFICATION_PROCESS: {
			GDVIRTUAL_CALL(_on_update, get_process_delta_time());
		} break;
		case NOTIFICATION_EXIT_TREE: {
			// Disconnect from viewport/tree signals here
		} break;
	}
}

void MyNode::_mesh_changed() {
	// Respond to resource data change (e.g., update visuals)
}

void MyNode::set_speed(float p_speed) {
	speed = p_speed;
}

float MyNode::get_speed() const {
	return speed;
}

void MyNode::set_mode(MyEnum p_mode) {
	mode = p_mode;
	_update_internal();
}

MyNode::MyEnum MyNode::get_mode() const {
	return mode;
}

void MyNode::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh == p_mesh) {
		return;
	}
	// Disconnect from old resource's changed signal
	if (mesh.is_valid()) {
		mesh->disconnect_changed(callable_mp(this, &MyNode::_mesh_changed));
	}
	mesh = p_mesh;
	// Connect to new resource's changed signal
	if (mesh.is_valid()) {
		mesh->connect_changed(callable_mp(this, &MyNode::_mesh_changed));
	}
}

Ref<Mesh> MyNode::get_mesh() const {
	return mesh;
}

void MyNode::_update_internal() {
	// Private implementation
}

void MyNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_speed", "speed"), &MyNode::set_speed);
	ClassDB::bind_method(D_METHOD("get_speed"), &MyNode::get_speed);
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &MyNode::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &MyNode::get_mode);
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MyNode::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MyNode::get_mesh);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed", PROPERTY_HINT_RANGE, "0,100,0.1,or_greater"), "set_speed", "get_speed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Mode A,Mode B,Mode C"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");

	BIND_ENUM_CONSTANT(MODE_A);
	BIND_ENUM_CONSTANT(MODE_B);
	BIND_ENUM_CONSTANT(MODE_C);

	ADD_SIGNAL(MethodInfo("mode_changed", PropertyInfo(Variant::INT, "new_mode")));

	GDVIRTUAL_BIND(_on_update, "delta");
}

MyNode::MyNode() {
	set_process(true);
}

MyNode::~MyNode() {
}
```

### Registration
In the appropriate `register_types.cpp`:
```cpp
#include "my_node.h"

void initialize_my_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(MyNode);
	}
}
```

---

## 2. Adding a New Resource Type

```cpp
// my_resource.h
#pragma once
#include "core/io/resource.h"

class MyResource : public Resource {
	GDCLASS(MyResource, Resource);

	float value = 0.0;
	String label;

protected:
	static void _bind_methods();

public:
	void set_value(float p_value);
	float get_value() const;
	void set_label(const String &p_label);
	String get_label() const;
};
```

```cpp
// my_resource.cpp
#include "my_resource.h"

void MyResource::set_value(float p_value) {
	if (value == p_value) {
		return;
	}
	value = p_value;
	emit_changed();
}

float MyResource::get_value() const {
	return value;
}

void MyResource::set_label(const String &p_label) {
	if (label == p_label) {
		return;
	}
	label = p_label;
	emit_changed();
}

String MyResource::get_label() const {
	return label;
}

void MyResource::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_value", "value"), &MyResource::set_value);
	ClassDB::bind_method(D_METHOD("get_value"), &MyResource::get_value);
	ClassDB::bind_method(D_METHOD("set_label", "label"), &MyResource::set_label);
	ClassDB::bind_method(D_METHOD("get_label"), &MyResource::get_label);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "value", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label"), "set_label", "get_label");
}
```

Key difference from Node: Resources call `emit_changed()` in setters to notify dependents.

---

## 3. Adding a New Module

### `modules/my_module/config.py`
```python
def can_build(env, platform):
    return True

def configure(env):
    pass

def get_doc_classes():
    return [
        "MyClass",
    ]

def get_doc_path():
    return "doc_classes"
```

### `modules/my_module/SCsub`
```python
#!/usr/bin/env python
from misc.utility.scons_hints import *

Import("env")
Import("env_modules")

env_my_module = env_modules.Clone()

module_obj = []
env_my_module.add_source_files(module_obj, "*.cpp")

if env.editor_build:
    env_my_module.add_source_files(module_obj, "editor/*.cpp")

env.modules_sources += module_obj
```

### `modules/my_module/register_types.h`
```cpp
#pragma once
#include "modules/register_module_types.h"

void initialize_my_module_module(ModuleInitializationLevel p_level);
void uninitialize_my_module_module(ModuleInitializationLevel p_level);
```

### `modules/my_module/register_types.cpp`
```cpp
#include "register_types.h"
#include "my_class.h"

void initialize_my_module_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(MyClass);
	}
}

void uninitialize_my_module_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
```

---

## 4. Adding an Editor Plugin

```cpp
// editor/plugins/my_editor_plugin.h
#pragma once
#include "editor/plugins/editor_plugin.h"

class MyEditorPlugin : public EditorPlugin {
	GDCLASS(MyEditorPlugin, EditorPlugin);

protected:
	static void _bind_methods();

public:
	virtual String get_name() const override { return "MyPlugin"; }

	MyEditorPlugin();
	~MyEditorPlugin();
};
```

Register with:
```cpp
#ifdef TOOLS_ENABLED
if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
    EditorPlugins::add_by_type<MyEditorPlugin>();
}
#endif
```

---

## 5. Adding an Inspector Plugin

```cpp
class MyInspectorPlugin : public EditorInspectorPlugin {
	GDCLASS(MyInspectorPlugin, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type,
		const String &p_name, const PropertyHint p_hint,
		const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage,
		const bool p_wide) override;
	virtual void parse_end(Object *p_object) override;
};
```

---

## 6. Writing Tests

### Test File (`tests/scene/test_my_node.h`)
```cpp
/**************************************************************************/
/*  test_my_node.h                                                        */
/**************************************************************************/
/* ... copyright block ... */
/**************************************************************************/

#pragma once

#include "tests/test_macros.h"
#include "scene/3d/my_node.h"

namespace TestMyNode {

TEST_CASE("[MyNode] Default values") {
	MyNode *node = memnew(MyNode);
	CHECK(node->get_speed() == doctest::Approx(1.0));
	CHECK(node->get_mode() == MyNode::MODE_A);
	CHECK(node->get_mesh().is_null());
	memdelete(node);
}

TEST_CASE("[MyNode] Set and get speed") {
	MyNode *node = memnew(MyNode);
	node->set_speed(5.0);
	CHECK(node->get_speed() == doctest::Approx(5.0));
	memdelete(node);
}

TEST_CASE("[MyNode] Set mode emits no crash") {
	MyNode *node = memnew(MyNode);
	node->set_mode(MyNode::MODE_B);
	CHECK(node->get_mode() == MyNode::MODE_B);
	node->set_mode(MyNode::MODE_C);
	CHECK(node->get_mode() == MyNode::MODE_C);
	memdelete(node);
}

TEST_CASE("[MyNode] Mesh assignment") {
	MyNode *node = memnew(MyNode);
	Ref<Mesh> mesh;
	mesh.instantiate();
	node->set_mesh(mesh);
	CHECK(node->get_mesh() == mesh);
	memdelete(node);
}

} // namespace TestMyNode
```

### Key Testing Patterns
- Namespace: `TestClassName`
- Tags in brackets: `[ClassName]` in TEST_CASE name
- Use `memnew`/`memdelete` for test objects (not `new`/`delete`)
- Use `doctest::Approx()` for float comparison
- `ERR_PRINT_OFF` / `ERR_PRINT_ON` to suppress expected errors
- Include test header in `tests/test_main.cpp`

---

## 7. Documentation XML

### `doc/classes/MyNode.xml`
```xml
<?xml version="1.0" encoding="UTF-8" ?>
<class name="MyNode" inherits="Node3D" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="../class.xsd">
	<brief_description>
		A custom node that does something.
	</brief_description>
	<description>
		Detailed description of what MyNode does and how to use it.
		Use [member speed] to control the rate. Use [method set_mode] to switch behavior.
		See also [Node3D] for transform properties.
	</description>
	<tutorials>
	</tutorials>
	<methods>
		<method name="set_speed">
			<return type="void" />
			<param index="0" name="speed" type="float" />
			<description>
				Sets the speed value.
			</description>
		</method>
		<method name="get_speed" qualifiers="const">
			<return type="float" />
			<description>
				Returns the current speed.
			</description>
		</method>
	</methods>
	<members>
		<member name="speed" type="float" setter="set_speed" getter="get_speed" default="1.0">
			The speed at which this node operates.
		</member>
		<member name="mode" type="int" setter="set_mode" getter="get_mode" enum="MyNode.MyEnum" default="0">
			The operating mode.
		</member>
		<member name="mesh" type="Mesh" setter="set_mesh" getter="get_mesh">
			The mesh resource used by this node.
		</member>
	</members>
	<signals>
		<signal name="mode_changed">
			<param index="0" name="new_mode" type="int" />
			<description>
				Emitted when the mode changes.
			</description>
		</signal>
	</signals>
	<constants>
		<constant name="MODE_A" value="0" enum="MyEnum">
			First mode.
		</constant>
		<constant name="MODE_B" value="1" enum="MyEnum">
			Second mode.
		</constant>
		<constant name="MODE_C" value="2" enum="MyEnum">
			Third mode.
		</constant>
	</constants>
</class>
```

### XML Markup Reference
- `[ClassName]` — link to class
- `[method method_name]` — link to method
- `[member member_name]` — link to property
- `[signal signal_name]` — link to signal
- `[constant CONSTANT_NAME]` — link to constant
- `[enum ClassName.EnumName]` — link to enum
- `[code]inline_code[/code]` — inline code
- `[b]bold[/b]`, `[i]italic[/i]`
- `[param param_name]` — reference to parameter

---

## 8. Vararg Method Binding

For methods that take variable arguments (like `rpc`):

```cpp
{
	MethodInfo mi;
	mi.name = "my_vararg_method";
	mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "method_name"));
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "my_vararg_method",
		&MyClass::_my_vararg_method_bind, mi);
}
```

---

## 9. Indexed Property Binding

When multiple properties share one getter/setter with an index:

```cpp
ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_left", PROPERTY_HINT_RANGE, "0,1,0.001"),
	"_set_anchor", "get_anchor", SIDE_LEFT);
ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_top", PROPERTY_HINT_RANGE, "0,1,0.001"),
	"_set_anchor", "get_anchor", SIDE_TOP);
ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_right", PROPERTY_HINT_RANGE, "0,1,0.001"),
	"_set_anchor", "get_anchor", SIDE_RIGHT);
ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_bottom", PROPERTY_HINT_RANGE, "0,1,0.001"),
	"_set_anchor", "get_anchor", SIDE_BOTTOM);
```

---

## 10. Theme Item Binding (GUI nodes)

```cpp
BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, normal);
BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, pressed);
BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_color);
BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, Button, font);
BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, Button, font_size);
BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, Button, h_separation);
```

These bind to the `ThemeCache` struct declared in the header.
