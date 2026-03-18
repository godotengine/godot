#ifndef SHADER_COMPILATION_TYPES_H
#define SHADER_COMPILATION_TYPES_H

#include "core/templates/rid.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"

namespace TileShaderCompilation {

static constexpr int DEFAULT_VARIANT_INDEX = 0;

struct ShaderVariant {
	String label;
	int index = DEFAULT_VARIANT_INDEX;
	Vector<String> defines;
};

struct CompilationResult {
	RID shader;
	RID pipeline;
	String error_message;

	bool is_valid() const { return shader.is_valid(); }
};

} // namespace TileShaderCompilation

#endif // SHADER_COMPILATION_TYPES_H
