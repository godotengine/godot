/* WARNING, THIS FILE WAS GENERATED, DO NOT EDIT */
#ifndef COMPUTE_GLSL_GEN_H_RD
#define COMPUTE_GLSL_GEN_H_RD

#include "servers/rendering/renderer_rd/shader_rd.h"

class ComputeShaderRD : public ShaderRD {

public:

	ComputeShaderRD() {

		static const char _compute_code[] = {
R"<!>(
#version 450

#VERSION_DEFINES

#define BLOCK_SIZE 8

#define M_PI 3.14159265359

void main() {
	uint t = BLOCK_SIZE + 1;
}
)<!>"
		};
		setup(nullptr, nullptr, _compute_code, "ComputeShaderRD");
	}
};

#endif
