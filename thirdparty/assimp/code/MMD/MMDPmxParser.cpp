/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
copyright notice, this list of conditions and the
following disclaimer.

* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other
materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
contributors may be used to endorse or promote products
derived from this software without specific prior
written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/
#include <utility>
#include "MMDPmxParser.h"
#include <assimp/StringUtils.h>
#ifdef ASSIMP_USE_HUNTER
#  include <utf8/utf8.h>
#else
#  include "../contrib/utf8cpp/source/utf8.h"
#endif
#include <assimp/Exceptional.h>

namespace pmx
{
	int ReadIndex(std::istream *stream, int size)
	{
		switch (size)
		{
		case 1:
			uint8_t tmp8;
			stream->read((char*) &tmp8, sizeof(uint8_t));
			if (255 == tmp8)
			{
				return -1;
			}
			else {
				return (int) tmp8;
			}
		case 2:
			uint16_t tmp16;
			stream->read((char*) &tmp16, sizeof(uint16_t));
			if (65535 == tmp16)
			{
				return -1;
			}
			else {
				return (int) tmp16;
			}
		case 4:
			int tmp32;
			stream->read((char*) &tmp32, sizeof(int));
			return tmp32;
		default:
			return -1;
		}
	}

	std::string ReadString(std::istream *stream, uint8_t encoding)
	{
		int size;
		stream->read((char*) &size, sizeof(int));
		std::vector<char> buffer;
		if (size == 0)
		{
			return std::string("");
		}
		buffer.reserve(size);
		stream->read((char*) buffer.data(), size);
		if (encoding == 0)
		{
			// UTF16 to UTF8
			const uint16_t* sourceStart = (uint16_t*)buffer.data();
			const unsigned int targetSize = size * 3; // enough to encode
			char *targetStart = new char[targetSize];
            std::memset(targetStart, 0, targetSize * sizeof(char));
            
            utf8::utf16to8( sourceStart, sourceStart + size/2, targetStart );

			std::string result(targetStart);
            delete [] targetStart;
			return result;
		}
		else
		{
			// the name is already UTF8
			return std::string((const char*)buffer.data(), size);
		}
	}

	void PmxSetting::Read(std::istream *stream)
	{
		uint8_t count;
		stream->read((char*) &count, sizeof(uint8_t));
		if (count < 8)
		{
			throw DeadlyImportError("MMD: invalid size");
		}
		stream->read((char*) &encoding, sizeof(uint8_t));
		stream->read((char*) &uv, sizeof(uint8_t));
		stream->read((char*) &vertex_index_size, sizeof(uint8_t));
		stream->read((char*) &texture_index_size, sizeof(uint8_t));
		stream->read((char*) &material_index_size, sizeof(uint8_t));
		stream->read((char*) &bone_index_size, sizeof(uint8_t));
		stream->read((char*) &morph_index_size, sizeof(uint8_t));
		stream->read((char*) &rigidbody_index_size, sizeof(uint8_t));
		uint8_t temp;
		for (int i = 8; i < count; i++)
		{
			stream->read((char*)&temp, sizeof(uint8_t));
		}
	}

	void PmxVertexSkinningBDEF1::Read(std::istream *stream, PmxSetting *setting)
	{
		this->bone_index = ReadIndex(stream, setting->bone_index_size);
	}

	void PmxVertexSkinningBDEF2::Read(std::istream *stream, PmxSetting *setting)
	{
		this->bone_index1 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index2 = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*) &this->bone_weight, sizeof(float));
	}

	void PmxVertexSkinningBDEF4::Read(std::istream *stream, PmxSetting *setting)
	{
		this->bone_index1 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index2 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index3 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index4 = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*) &this->bone_weight1, sizeof(float));
		stream->read((char*) &this->bone_weight2, sizeof(float));
		stream->read((char*) &this->bone_weight3, sizeof(float));
		stream->read((char*) &this->bone_weight4, sizeof(float));
	}

	void PmxVertexSkinningSDEF::Read(std::istream *stream, PmxSetting *setting)
	{
		this->bone_index1 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index2 = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*) &this->bone_weight, sizeof(float));
		stream->read((char*) this->sdef_c, sizeof(float) * 3);
		stream->read((char*) this->sdef_r0, sizeof(float) * 3);
		stream->read((char*) this->sdef_r1, sizeof(float) * 3);
	}

	void PmxVertexSkinningQDEF::Read(std::istream *stream, PmxSetting *setting)
	{
		this->bone_index1 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index2 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index3 = ReadIndex(stream, setting->bone_index_size);
		this->bone_index4 = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*) &this->bone_weight1, sizeof(float));
		stream->read((char*) &this->bone_weight2, sizeof(float));
		stream->read((char*) &this->bone_weight3, sizeof(float));
		stream->read((char*) &this->bone_weight4, sizeof(float));
	}

	void PmxVertex::Read(std::istream *stream, PmxSetting *setting)
	{
		stream->read((char*) this->position, sizeof(float) * 3);
		stream->read((char*) this->normal, sizeof(float) * 3);
		stream->read((char*) this->uv, sizeof(float) * 2);
		for (int i = 0; i < setting->uv; ++i)
		{
			stream->read((char*) this->uva[i], sizeof(float) * 4);
		}
		stream->read((char*) &this->skinning_type, sizeof(PmxVertexSkinningType));
		switch (this->skinning_type)
		{
		case PmxVertexSkinningType::BDEF1:
			this->skinning = mmd::make_unique<PmxVertexSkinningBDEF1>();
			break;
		case PmxVertexSkinningType::BDEF2:
			this->skinning = mmd::make_unique<PmxVertexSkinningBDEF2>();
			break;
		case PmxVertexSkinningType::BDEF4:
			this->skinning = mmd::make_unique<PmxVertexSkinningBDEF4>();
			break;
		case PmxVertexSkinningType::SDEF:
			this->skinning = mmd::make_unique<PmxVertexSkinningSDEF>();
			break;
		case PmxVertexSkinningType::QDEF:
			this->skinning = mmd::make_unique<PmxVertexSkinningQDEF>();
			break;
		default:
			throw "invalid skinning type";
		}
		this->skinning->Read(stream, setting);
		stream->read((char*) &this->edge, sizeof(float));
	}

	void PmxMaterial::Read(std::istream *stream, PmxSetting *setting)
	{
		this->material_name = ReadString(stream, setting->encoding);
		this->material_english_name = ReadString(stream, setting->encoding);
		stream->read((char*) this->diffuse, sizeof(float) * 4);
		stream->read((char*) this->specular, sizeof(float) * 3);
		stream->read((char*) &this->specularlity, sizeof(float));
		stream->read((char*) this->ambient, sizeof(float) * 3);
		stream->read((char*) &this->flag, sizeof(uint8_t));
		stream->read((char*) this->edge_color, sizeof(float) * 4);
		stream->read((char*) &this->edge_size, sizeof(float));
		this->diffuse_texture_index = ReadIndex(stream, setting->texture_index_size);
		this->sphere_texture_index = ReadIndex(stream, setting->texture_index_size);
		stream->read((char*) &this->sphere_op_mode, sizeof(uint8_t));
		stream->read((char*) &this->common_toon_flag, sizeof(uint8_t));
		if (this->common_toon_flag)
		{
			stream->read((char*) &this->toon_texture_index, sizeof(uint8_t));
		}
		else {
			this->toon_texture_index = ReadIndex(stream, setting->texture_index_size);
		}
		this->memo = ReadString(stream, setting->encoding);
		stream->read((char*) &this->index_count, sizeof(int));
	}

	void PmxIkLink::Read(std::istream *stream, PmxSetting *setting)
	{
		this->link_target = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*) &this->angle_lock, sizeof(uint8_t));
		if (angle_lock == 1)
		{
			stream->read((char*) this->max_radian, sizeof(float) * 3);
			stream->read((char*) this->min_radian, sizeof(float) * 3);
		}
	}

	void PmxBone::Read(std::istream *stream, PmxSetting *setting)
	{
		this->bone_name = ReadString(stream, setting->encoding);
		this->bone_english_name = ReadString(stream, setting->encoding);
		stream->read((char*) this->position, sizeof(float) * 3);
		this->parent_index = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*) &this->level, sizeof(int));
		stream->read((char*) &this->bone_flag, sizeof(uint16_t));
		if (this->bone_flag & 0x0001) {
			this->target_index = ReadIndex(stream, setting->bone_index_size);
		}
		else {
			stream->read((char*)this->offset, sizeof(float) * 3);
		}
		if (this->bone_flag & (0x0100 | 0x0200)) {
			this->grant_parent_index = ReadIndex(stream, setting->bone_index_size);
			stream->read((char*) &this->grant_weight, sizeof(float));
		}
		if (this->bone_flag & 0x0400) {
			stream->read((char*)this->lock_axis_orientation, sizeof(float) * 3);
		}
		if (this->bone_flag & 0x0800) {
			stream->read((char*)this->local_axis_x_orientation, sizeof(float) * 3);
			stream->read((char*)this->local_axis_y_orientation, sizeof(float) * 3);
		}
		if (this->bone_flag & 0x2000) {
			stream->read((char*) &this->key, sizeof(int));
		}
		if (this->bone_flag & 0x0020) {
			this->ik_target_bone_index = ReadIndex(stream, setting->bone_index_size);
			stream->read((char*) &ik_loop, sizeof(int));
			stream->read((char*) &ik_loop_angle_limit, sizeof(float));
			stream->read((char*) &ik_link_count, sizeof(int));
			this->ik_links = mmd::make_unique<PmxIkLink []>(ik_link_count);
			for (int i = 0; i < ik_link_count; i++) {
				ik_links[i].Read(stream, setting);
			}
		}
	}

	void PmxMorphVertexOffset::Read(std::istream *stream, PmxSetting *setting)
	{
		this->vertex_index = ReadIndex(stream, setting->vertex_index_size);
		stream->read((char*)this->position_offset, sizeof(float) * 3);
	}

	void PmxMorphUVOffset::Read(std::istream *stream, PmxSetting *setting)
	{
		this->vertex_index = ReadIndex(stream, setting->vertex_index_size);
		stream->read((char*)this->uv_offset, sizeof(float) * 4);
	}

	void PmxMorphBoneOffset::Read(std::istream *stream, PmxSetting *setting)
	{
		this->bone_index = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*)this->translation, sizeof(float) * 3);
		stream->read((char*)this->rotation, sizeof(float) * 4);
	}

	void PmxMorphMaterialOffset::Read(std::istream *stream, PmxSetting *setting)
	{
		this->material_index = ReadIndex(stream, setting->material_index_size);
		stream->read((char*) &this->offset_operation, sizeof(uint8_t));
		stream->read((char*)this->diffuse, sizeof(float) * 4);
		stream->read((char*)this->specular, sizeof(float) * 3);
		stream->read((char*) &this->specularity, sizeof(float));
		stream->read((char*)this->ambient, sizeof(float) * 3);
		stream->read((char*)this->edge_color, sizeof(float) * 4);
		stream->read((char*) &this->edge_size, sizeof(float));
		stream->read((char*)this->texture_argb, sizeof(float) * 4);
		stream->read((char*)this->sphere_texture_argb, sizeof(float) * 4);
		stream->read((char*)this->toon_texture_argb, sizeof(float) * 4);
	}

	void PmxMorphGroupOffset::Read(std::istream *stream, PmxSetting *setting)
	{
		this->morph_index = ReadIndex(stream, setting->morph_index_size);
		stream->read((char*) &this->morph_weight, sizeof(float));
	}

	void PmxMorphFlipOffset::Read(std::istream *stream, PmxSetting *setting)
	{
		this->morph_index = ReadIndex(stream, setting->morph_index_size);
		stream->read((char*) &this->morph_value, sizeof(float));
	}

	void PmxMorphImplusOffset::Read(std::istream *stream, PmxSetting *setting)
	{
		this->rigid_body_index = ReadIndex(stream, setting->rigidbody_index_size);
		stream->read((char*) &this->is_local, sizeof(uint8_t));
		stream->read((char*)this->velocity, sizeof(float) * 3);
		stream->read((char*)this->angular_torque, sizeof(float) * 3);
	}

	void PmxMorph::Read(std::istream *stream, PmxSetting *setting)
	{
		this->morph_name = ReadString(stream, setting->encoding);
		this->morph_english_name = ReadString(stream, setting->encoding);
		stream->read((char*) &category, sizeof(MorphCategory));
		stream->read((char*) &morph_type, sizeof(MorphType));
		stream->read((char*) &this->offset_count, sizeof(int));
		switch (this->morph_type)
		{
		case MorphType::Group:
			group_offsets = mmd::make_unique<PmxMorphGroupOffset []>(this->offset_count);
			for (int i = 0; i < offset_count; i++)
			{
				group_offsets[i].Read(stream, setting);
			}
			break;
		case MorphType::Vertex:
			vertex_offsets = mmd::make_unique<PmxMorphVertexOffset []>(this->offset_count);
			for (int i = 0; i < offset_count; i++)
			{
				vertex_offsets[i].Read(stream, setting);
			}
			break;
		case MorphType::Bone:
			bone_offsets = mmd::make_unique<PmxMorphBoneOffset []>(this->offset_count);
			for (int i = 0; i < offset_count; i++)
			{
				bone_offsets[i].Read(stream, setting);
			}
			break;
		case MorphType::Matrial:
			material_offsets = mmd::make_unique<PmxMorphMaterialOffset []>(this->offset_count);
			for (int i = 0; i < offset_count; i++)
			{
				material_offsets[i].Read(stream, setting);
			}
			break;
		case MorphType::UV:
		case MorphType::AdditionalUV1:
		case MorphType::AdditionalUV2:
		case MorphType::AdditionalUV3:
		case MorphType::AdditionalUV4:
			uv_offsets = mmd::make_unique<PmxMorphUVOffset []>(this->offset_count);
			for (int i = 0; i < offset_count; i++)
			{
				uv_offsets[i].Read(stream, setting);
			}
			break;
		default:
            throw DeadlyImportError("MMD: unknown morth type");
		}
	}

	void PmxFrameElement::Read(std::istream *stream, PmxSetting *setting)
	{
		stream->read((char*) &this->element_target, sizeof(uint8_t));
		if (this->element_target == 0x00)
		{
			this->index = ReadIndex(stream, setting->bone_index_size);
		}
		else {
			this->index = ReadIndex(stream, setting->morph_index_size);
		}
	}

	void PmxFrame::Read(std::istream *stream, PmxSetting *setting)
	{
		this->frame_name = ReadString(stream, setting->encoding);
		this->frame_english_name = ReadString(stream, setting->encoding);
		stream->read((char*) &this->frame_flag, sizeof(uint8_t));
		stream->read((char*) &this->element_count, sizeof(int));
		this->elements = mmd::make_unique<PmxFrameElement []>(this->element_count);
		for (int i = 0; i < this->element_count; i++)
		{
			this->elements[i].Read(stream, setting);
		}
	}

	void PmxRigidBody::Read(std::istream *stream, PmxSetting *setting)
	{
		this->girid_body_name = ReadString(stream, setting->encoding);
		this->girid_body_english_name = ReadString(stream, setting->encoding);
		this->target_bone = ReadIndex(stream, setting->bone_index_size);
		stream->read((char*) &this->group, sizeof(uint8_t));
		stream->read((char*) &this->mask, sizeof(uint16_t));
		stream->read((char*) &this->shape, sizeof(uint8_t));
		stream->read((char*) this->size, sizeof(float) * 3);
		stream->read((char*) this->position, sizeof(float) * 3);
		stream->read((char*) this->orientation, sizeof(float) * 3);
		stream->read((char*) &this->mass, sizeof(float));
		stream->read((char*) &this->move_attenuation, sizeof(float));
		stream->read((char*) &this->rotation_attenuation, sizeof(float));
		stream->read((char*) &this->repulsion, sizeof(float));
		stream->read((char*) &this->friction, sizeof(float));
		stream->read((char*) &this->physics_calc_type, sizeof(uint8_t));
	}

	void PmxJointParam::Read(std::istream *stream, PmxSetting *setting)
	{
		this->rigid_body1 = ReadIndex(stream, setting->rigidbody_index_size);
		this->rigid_body2 = ReadIndex(stream, setting->rigidbody_index_size);
		stream->read((char*) this->position, sizeof(float) * 3);
		stream->read((char*) this->orientaiton, sizeof(float) * 3);
		stream->read((char*) this->move_limitation_min, sizeof(float) * 3);
		stream->read((char*) this->move_limitation_max, sizeof(float) * 3);
		stream->read((char*) this->rotation_limitation_min, sizeof(float) * 3);
		stream->read((char*) this->rotation_limitation_max, sizeof(float) * 3);
		stream->read((char*) this->spring_move_coefficient, sizeof(float) * 3);
		stream->read((char*) this->spring_rotation_coefficient, sizeof(float) * 3);
	}

	void PmxJoint::Read(std::istream *stream, PmxSetting *setting)
	{
		this->joint_name = ReadString(stream, setting->encoding);
		this->joint_english_name = ReadString(stream, setting->encoding);
		stream->read((char*) &this->joint_type, sizeof(uint8_t));
		this->param.Read(stream, setting);
	}

	void PmxAncherRigidBody::Read(std::istream *stream, PmxSetting *setting)
	{
		this->related_rigid_body = ReadIndex(stream, setting->rigidbody_index_size);
		this->related_vertex = ReadIndex(stream, setting->vertex_index_size);
		stream->read((char*) &this->is_near, sizeof(uint8_t));
	}

    void PmxSoftBody::Read(std::istream * /*stream*/, PmxSetting * /*setting*/)
	{
		std::cerr << "Not Implemented Exception" << std::endl;
        throw DeadlyImportError("MMD: Not Implemented Exception");
    }

	void PmxModel::Init()
	{
		this->version = 0.0f;
		this->model_name.clear();
		this->model_english_name.clear();
		this->model_comment.clear();
		this->model_english_comment.clear();
		this->vertex_count = 0;
		this->vertices = nullptr;
		this->index_count = 0;
		this->indices = nullptr;
		this->texture_count = 0;
		this->textures = nullptr;
		this->material_count = 0;
		this->materials = nullptr;
		this->bone_count = 0;
		this->bones = nullptr;
		this->morph_count = 0;
		this->morphs = nullptr;
		this->frame_count = 0;
		this->frames = nullptr;
		this->rigid_body_count = 0;
		this->rigid_bodies = nullptr;
		this->joint_count = 0;
		this->joints = nullptr;
		this->soft_body_count = 0;
		this->soft_bodies = nullptr;
	}

	void PmxModel::Read(std::istream *stream)
	{
		char magic[4];
		stream->read((char*) magic, sizeof(char) * 4);
		if (magic[0] != 0x50 || magic[1] != 0x4d || magic[2] != 0x58 || magic[3] != 0x20)
		{
			std::cerr << "invalid magic number." << std::endl;
      throw DeadlyImportError("MMD: invalid magic number.");
    }
		stream->read((char*) &version, sizeof(float));
		if (version != 2.0f && version != 2.1f)
		{
			std::cerr << "this is not ver2.0 or ver2.1 but " << version << "." << std::endl;
            throw DeadlyImportError("MMD: this is not ver2.0 or ver2.1 but " + to_string(version));
    }
		this->setting.Read(stream);

		this->model_name = ReadString(stream, setting.encoding);
		this->model_english_name = ReadString(stream, setting.encoding);
		this->model_comment = ReadString(stream, setting.encoding);
		this->model_english_comment = ReadString(stream, setting.encoding);

		// read vertices
		stream->read((char*) &vertex_count, sizeof(int));
		this->vertices = mmd::make_unique<PmxVertex []>(vertex_count);
		for (int i = 0; i < vertex_count; i++)
		{
			vertices[i].Read(stream, &setting);
		}

		// read indices
		stream->read((char*) &index_count, sizeof(int));
		this->indices = mmd::make_unique<int []>(index_count);
		for (int i = 0; i < index_count; i++)
		{
			this->indices[i] = ReadIndex(stream, setting.vertex_index_size);
		}

		// read texture names
		stream->read((char*) &texture_count, sizeof(int));
		this->textures = mmd::make_unique<std::string []>(texture_count);
		for (int i = 0; i < texture_count; i++)
		{
			this->textures[i] = ReadString(stream, setting.encoding);
		}

		// read materials
		stream->read((char*) &material_count, sizeof(int));
		this->materials = mmd::make_unique<PmxMaterial []>(material_count);
		for (int i = 0; i < material_count; i++)
		{
			this->materials[i].Read(stream, &setting);
		}

		// read bones
		stream->read((char*) &this->bone_count, sizeof(int));
		this->bones = mmd::make_unique<PmxBone []>(this->bone_count);
		for (int i = 0; i < this->bone_count; i++)
		{
			this->bones[i].Read(stream, &setting);
		}

		// read morphs
		stream->read((char*) &this->morph_count, sizeof(int));
		this->morphs = mmd::make_unique<PmxMorph []>(this->morph_count);
		for (int i = 0; i < this->morph_count; i++)
		{
			this->morphs[i].Read(stream, &setting);
		}

		// read display frames
		stream->read((char*) &this->frame_count, sizeof(int));
		this->frames = mmd::make_unique<PmxFrame []>(this->frame_count);
		for (int i = 0; i < this->frame_count; i++)
		{
			this->frames[i].Read(stream, &setting);
		}

		// read rigid bodies
		stream->read((char*) &this->rigid_body_count, sizeof(int));
		this->rigid_bodies = mmd::make_unique<PmxRigidBody []>(this->rigid_body_count);
		for (int i = 0; i < this->rigid_body_count; i++)
		{
			this->rigid_bodies[i].Read(stream, &setting);
		}

		// read joints
		stream->read((char*) &this->joint_count, sizeof(int));
		this->joints = mmd::make_unique<PmxJoint []>(this->joint_count);
		for (int i = 0; i < this->joint_count; i++)
		{
			this->joints[i].Read(stream, &setting);
		}
	}
}
