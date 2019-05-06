/*
 * Copyright 2015-2019 Arm Limited
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

#ifndef SPIRV_CROSS_CPP_HPP
#define SPIRV_CROSS_CPP_HPP

#include "spirv_glsl.hpp"
#include <utility>

namespace SPIRV_CROSS_NAMESPACE
{
class CompilerCPP : public CompilerGLSL
{
public:
	explicit CompilerCPP(std::vector<uint32_t> spirv_)
	    : CompilerGLSL(std::move(spirv_))
	{
	}

	CompilerCPP(const uint32_t *ir_, size_t word_count)
	    : CompilerGLSL(ir_, word_count)
	{
	}

	explicit CompilerCPP(const ParsedIR &ir_)
	    : CompilerGLSL(ir_)
	{
	}

	explicit CompilerCPP(ParsedIR &&ir_)
	    : CompilerGLSL(std::move(ir_))
	{
	}

	std::string compile() override;

	// Sets a custom symbol name that can override
	// spirv_cross_get_interface.
	//
	// Useful when several shader interfaces are linked
	// statically into the same binary.
	void set_interface_name(std::string name)
	{
		interface_name = std::move(name);
	}

private:
	void emit_header() override;
	void emit_c_linkage();
	void emit_function_prototype(SPIRFunction &func, const Bitset &return_flags) override;

	void emit_resources();
	void emit_buffer_block(const SPIRVariable &type) override;
	void emit_push_constant_block(const SPIRVariable &var) override;
	void emit_interface_block(const SPIRVariable &type);
	void emit_block_chain(SPIRBlock &block);
	void emit_uniform(const SPIRVariable &var) override;
	void emit_shared(const SPIRVariable &var);
	void emit_block_struct(SPIRType &type);
	std::string variable_decl(const SPIRType &type, const std::string &name, uint32_t id) override;

	std::string argument_decl(const SPIRFunction::Parameter &arg);

	SmallVector<std::string> resource_registrations;
	std::string impl_type;
	std::string resource_type;
	uint32_t shared_counter = 0;

	std::string interface_name;
};
} // namespace SPIRV_CROSS_NAMESPACE

#endif
