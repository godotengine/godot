#ifndef SHADERTYPES_H
#define SHADERTYPES_H

#include "shader_language.h"
#include "servers/visual_server.h"
class ShaderTypes {


	struct Type {

		 Map< StringName, Map<StringName,ShaderLanguage::DataType> > functions;
		 Set<String> modes;
	};

	Map<VS::ShaderMode,Type> shader_modes;

	static ShaderTypes *singleton;
public:
	static ShaderTypes *get_singleton() { return singleton; }

	const Map< StringName, Map<StringName,ShaderLanguage::DataType> >& get_functions(VS::ShaderMode p_mode);
	const Set<String>& get_modes(VS::ShaderMode p_mode);

	ShaderTypes();
};

#endif // SHADERTYPES_H
