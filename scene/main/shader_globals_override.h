#ifndef SHADER_GLOBALS_OVERRIDE_H
#define SHADER_GLOBALS_OVERRIDE_H

#include "scene/3d/node_3d.h"

class ShaderGlobalsOverride : public Node {

	GDCLASS(ShaderGlobalsOverride, Node);

	struct Override {
		bool in_use = false;
		Variant override;
	};

	StringName *_remap(const StringName &p_name) const;

	bool active;
	mutable HashMap<StringName, Override> overrides;
	mutable HashMap<StringName, StringName> param_remaps;

	void _activate();

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	static void _bind_methods();

public:
	String get_configuration_warning() const;

	ShaderGlobalsOverride();
};

#endif // SHADER_GLOBALS_OVERRIDE_H
