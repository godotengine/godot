/*************************************************************************/
/*  shader.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef SHADER_H
#define SHADER_H

#include "resource.h"
#include "io/resource_loader.h"

class Shader : public Resource {

	OBJ_TYPE(Shader,Resource);
	OBJ_SAVE_TYPE( Shader );
	RES_BASE_EXTENSION("shd");
	RID shader;

	Dictionary _get_code();
	void _set_code(const Dictionary& p_string);


	// hack the name of performance
	// shaders keep a list of ShaderMaterial -> VisualServer name translations, to make
	// convertion fast and save memory.
	mutable bool params_cache_dirty;
	mutable Map<StringName,StringName> params_cache; //map a shader param to a material param..

protected:


	static void _bind_methods();
public:
	enum Mode {

		MODE_MATERIAL,
		MODE_CANVAS_ITEM,
		MODE_POST_PROCESS
	};

	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_code( const String& p_vertex, const String& p_fragment, const String& p_light,int p_fragment_ofs=0,int p_light_ofs=0);
	String get_vertex_code() const;
	String get_fragment_code() const;
	String get_light_code() const;

	void get_param_list(List<PropertyInfo> *p_params) const;
	bool has_param(const StringName& p_param) const;

	virtual RID get_rid() const;

	Shader();
	~Shader();

};

VARIANT_ENUM_CAST( Shader::Mode );



class ResourceFormatLoaderShader : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path,const String& p_original_path="");
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};



#endif // SHADER_H
