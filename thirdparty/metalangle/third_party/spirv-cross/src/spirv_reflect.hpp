/*
 * Copyright 2018-2020 Bradley Austin Davis
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_REFLECT_HPP
#define SPIRV_CROSS_REFLECT_HPP

#include "spirv_glsl.hpp"
#include <utility>

namespace simple_json
{
class Stream;
}

namespace SPIRV_CROSS_NAMESPACE
{
class CompilerReflection : public CompilerGLSL
{
	using Parent = CompilerGLSL;

public:
	explicit CompilerReflection(std::vector<uint32_t> spirv_)
	    : Parent(std::move(spirv_))
	{
		options.vulkan_semantics = true;
	}

	CompilerReflection(const uint32_t *ir_, size_t word_count)
	    : Parent(ir_, word_count)
	{
		options.vulkan_semantics = true;
	}

	explicit CompilerReflection(const ParsedIR &ir_)
	    : CompilerGLSL(ir_)
	{
		options.vulkan_semantics = true;
	}

	explicit CompilerReflection(ParsedIR &&ir_)
	    : CompilerGLSL(std::move(ir_))
	{
		options.vulkan_semantics = true;
	}

	void set_format(const std::string &format);
	std::string compile() override;

private:
	static std::string execution_model_to_str(spv::ExecutionModel model);

	void emit_entry_points();
	void emit_types();
	void emit_resources();
	void emit_specialization_constants();

	void emit_type(const SPIRType &type, bool &emitted_open_tag);
	void emit_type_member(const SPIRType &type, uint32_t index);
	void emit_type_member_qualifiers(const SPIRType &type, uint32_t index);
	void emit_type_array(const SPIRType &type);
	void emit_resources(const char *tag, const SmallVector<Resource> &resources);

	std::string to_member_name(const SPIRType &type, uint32_t index) const;

	std::shared_ptr<simple_json::Stream> json_stream;
};

} // namespace SPIRV_CROSS_NAMESPACE

#endif
