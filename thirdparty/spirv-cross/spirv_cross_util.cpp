/*
 * Copyright 2015-2021 Arm Limited
 * SPDX-License-Identifier: Apache-2.0 OR MIT
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

/*
 * At your option, you may choose to accept this material under either:
 *  1. The Apache License, Version 2.0, found at <http://www.apache.org/licenses/LICENSE-2.0>, or
 *  2. The MIT License, found at <http://opensource.org/licenses/MIT>.
 */

#include "spirv_cross_util.hpp"
#include "spirv_common.hpp"

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;

namespace spirv_cross_util
{
void rename_interface_variable(Compiler &compiler, const SmallVector<Resource> &resources, uint32_t location,
                               const std::string &name)
{
	for (auto &v : resources)
	{
		if (!compiler.has_decoration(v.id, spv::DecorationLocation))
			continue;

		auto loc = compiler.get_decoration(v.id, spv::DecorationLocation);
		if (loc != location)
			continue;

		auto &type = compiler.get_type(v.base_type_id);

		// This is more of a friendly variant. If we need to rename interface variables, we might have to rename
		// structs as well and make sure all the names match up.
		if (type.basetype == SPIRType::Struct)
		{
			compiler.set_name(v.base_type_id, join("SPIRV_Cross_Interface_Location", location));
			for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
				compiler.set_member_name(v.base_type_id, i, join("InterfaceMember", i));
		}

		compiler.set_name(v.id, name);
	}
}

void inherit_combined_sampler_bindings(Compiler &compiler)
{
	auto &samplers = compiler.get_combined_image_samplers();
	for (auto &s : samplers)
	{
		if (compiler.has_decoration(s.image_id, spv::DecorationDescriptorSet))
		{
			uint32_t set = compiler.get_decoration(s.image_id, spv::DecorationDescriptorSet);
			compiler.set_decoration(s.combined_id, spv::DecorationDescriptorSet, set);
		}

		if (compiler.has_decoration(s.image_id, spv::DecorationBinding))
		{
			uint32_t binding = compiler.get_decoration(s.image_id, spv::DecorationBinding);
			compiler.set_decoration(s.combined_id, spv::DecorationBinding, binding);
		}
	}
}
} // namespace spirv_cross_util
