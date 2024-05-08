/* WARNING, THIS FILE WAS GENERATED, DO NOT EDIT */
#ifndef VERTEX_FRAGMENT_GLSL_GEN_HGLES3_GLES3
#define VERTEX_FRAGMENT_GLSL_GEN_HGLES3_GLES3


#include "drivers/gles3/shader_gles3.h"


class VertexFragmentShaderGLES3 : public ShaderGLES3 {

public:

	enum ShaderVariant {
		MODE_NINEPATCH,
	};

	enum Specializations {
		DISABLE_LIGHTING=1,
	};

	_FORCE_INLINE_ bool version_bind_shader(RID p_version,ShaderVariant p_variant,uint64_t p_specialization=0) { return _version_bind_shader(p_version,p_variant,p_specialization); }

protected:

	virtual void _init() override {

		static const char **_uniform_strings=nullptr;
		static const char* _variant_defines[]={
			"#define USE_NINEPATCH",
		};

		static TexUnitPair *_texunit_pairs=nullptr;
		static UBOPair *_ubo_pairs=nullptr;
		static Specialization _spec_pairs[]={
			{"DISABLE_LIGHTING",false},
		};

		static const Feedback* _feedbacks=nullptr;
		static const char _vertex_code[]={
10,112,114,101,99,105,115,105,111,110,32,104,105,103,104,112,32,102,108,111,97,116,59,10,112,114,101,99,105,115,105,111,110,32,104,105,103,104,112,32,105,110,116,59,10,10,108,97,121,111,117,116,40,108,111,99,97,116,105,111,110,32,61,32,48,41,32,105,110,32,104,105,103,104,112,32,118,101,99,51,32,118,101,114,116,101,120,59,10,10,111,117,116,32,104,105,103,104,112,32,118,101,99,52,32,112,111,115,105,116,105,111,110,95,105,110,116,101,114,112,59,10,10,118,111,105,100,32,109,97,105,110,40,41,32,123,10,9,112,111,115,105,116,105,111,110,95,105,110,116,101,114,112,32,61,32,118,101,99,52,40,118,101,114,116,101,120,46,120,44,49,44,48,44,49,41,59,10,125,10,10,		0};

		static const char _fragment_code[]={
10,112,114,101,99,105,115,105,111,110,32,104,105,103,104,112,32,102,108,111,97,116,59,10,112,114,101,99,105,115,105,111,110,32,104,105,103,104,112,32,105,110,116,59,10,10,105,110,32,104,105,103,104,112,32,118,101,99,52,32,112,111,115,105,116,105,111,110,95,105,110,116,101,114,112,59,10,10,118,111,105,100,32,109,97,105,110,40,41,32,123,10,9,104,105,103,104,112,32,102,108,111,97,116,32,100,101,112,116,104,32,61,32,40,40,112,111,115,105,116,105,111,110,95,105,110,116,101,114,112,46,122,32,47,32,112,111,115,105,116,105,111,110,95,105,110,116,101,114,112,46,119,41,32,43,32,49,46,48,41,59,10,9,102,114,97,103,95,99,111,108,111,114,32,61,32,118,101,99,52,40,100,101,112,116,104,41,59,10,125,10,		0};

		_setup(_vertex_code,_fragment_code,"VertexFragmentShaderGLES3",0,_uniform_strings,0,_ubo_pairs,0,_feedbacks,0,_texunit_pairs,1,_spec_pairs,1,_variant_defines);
	}

};

#endif
