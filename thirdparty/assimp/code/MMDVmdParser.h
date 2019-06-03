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
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <ostream>
#include "MMDCpp14.h"

namespace vmd
{
	class VmdBoneFrame
	{
	public:
		std::string name;
		int frame;
		float position[3];
		float orientation[4];
		char interpolation[4][4][4];

		void Read(std::istream* stream)
		{
			char buffer[15];
			stream->read((char*) buffer, sizeof(char)*15);
			name = std::string(buffer);
			stream->read((char*) &frame, sizeof(int));
			stream->read((char*) position, sizeof(float)*3);
			stream->read((char*) orientation, sizeof(float)*4);
			stream->read((char*) interpolation, sizeof(char) * 4 * 4 * 4);
		}

		void Write(std::ostream* stream)
		{
			stream->write((char*)name.c_str(), sizeof(char) * 15);
			stream->write((char*)&frame, sizeof(int));
			stream->write((char*)position, sizeof(float) * 3);
			stream->write((char*)orientation, sizeof(float) * 4);
			stream->write((char*)interpolation, sizeof(char) * 4 * 4 * 4);
		}
	};

	class VmdFaceFrame
	{
	public:
		std::string face_name;
		float weight;
		uint32_t frame;

		void Read(std::istream* stream)
		{
			char buffer[15];
			stream->read((char*) &buffer, sizeof(char) * 15);
			face_name = std::string(buffer);
			stream->read((char*) &frame, sizeof(int));
			stream->read((char*) &weight, sizeof(float));
		}

		void Write(std::ostream* stream)
		{
			stream->write((char*)face_name.c_str(), sizeof(char) * 15);
			stream->write((char*)&frame, sizeof(int));
			stream->write((char*)&weight, sizeof(float));
		}
	};

	class VmdCameraFrame
	{
	public:
		int frame;
		float distance;
		float position[3];
		float orientation[3];
		char interpolation[6][4];
		float angle;
		char unknown[3];

		void Read(std::istream *stream)
		{
			stream->read((char*) &frame, sizeof(int));
			stream->read((char*) &distance, sizeof(float));
			stream->read((char*) position, sizeof(float) * 3);
			stream->read((char*) orientation, sizeof(float) * 3);
			stream->read((char*) interpolation, sizeof(char) * 24);
			stream->read((char*) &angle, sizeof(float));
			stream->read((char*) unknown, sizeof(char) * 3);
		}

		void Write(std::ostream *stream)
		{
			stream->write((char*)&frame, sizeof(int));
			stream->write((char*)&distance, sizeof(float));
			stream->write((char*)position, sizeof(float) * 3);
			stream->write((char*)orientation, sizeof(float) * 3);
			stream->write((char*)interpolation, sizeof(char) * 24);
			stream->write((char*)&angle, sizeof(float));
			stream->write((char*)unknown, sizeof(char) * 3);
		}
	};

	class VmdLightFrame
	{
	public:
		int frame;
		float color[3];
		float position[3];

		void Read(std::istream* stream)
		{
			stream->read((char*) &frame, sizeof(int));
			stream->read((char*) color, sizeof(float) * 3);
			stream->read((char*) position, sizeof(float) * 3);
		}

		void Write(std::ostream* stream)
		{
			stream->write((char*)&frame, sizeof(int));
			stream->write((char*)color, sizeof(float) * 3);
			stream->write((char*)position, sizeof(float) * 3);
		}
	};

	class VmdIkEnable
	{
	public:
		std::string ik_name;
		bool enable;
	};

	class VmdIkFrame
	{
	public:
		int frame;
		bool display;
		std::vector<VmdIkEnable> ik_enable;

		void Read(std::istream *stream)
		{
			char buffer[20];
			stream->read((char*) &frame, sizeof(int));
			stream->read((char*) &display, sizeof(uint8_t));
			int ik_count;
			stream->read((char*) &ik_count, sizeof(int));
			ik_enable.resize(ik_count);
			for (int i = 0; i < ik_count; i++)
			{
				stream->read(buffer, 20);
				ik_enable[i].ik_name = std::string(buffer);
				stream->read((char*) &ik_enable[i].enable, sizeof(uint8_t));
			}
		}

		void Write(std::ostream *stream)
		{
			stream->write((char*)&frame, sizeof(int));
			stream->write((char*)&display, sizeof(uint8_t));
			int ik_count = static_cast<int>(ik_enable.size());
			stream->write((char*)&ik_count, sizeof(int));
			for (int i = 0; i < ik_count; i++)
			{
				const VmdIkEnable& ik_enable = this->ik_enable.at(i);
				stream->write(ik_enable.ik_name.c_str(), 20);
				stream->write((char*)&ik_enable.enable, sizeof(uint8_t));
			}
		}
	};

	class VmdMotion
	{
	public:
		std::string model_name;
		int version;
		std::vector<VmdBoneFrame> bone_frames;
		std::vector<VmdFaceFrame> face_frames;
		std::vector<VmdCameraFrame> camera_frames;
		std::vector<VmdLightFrame> light_frames;
		std::vector<VmdIkFrame> ik_frames;

		static std::unique_ptr<VmdMotion> LoadFromFile(char const *filename)
		{
			std::ifstream stream(filename, std::ios::binary);
			auto result = LoadFromStream(&stream);
			stream.close();
			return result;
		}

		static std::unique_ptr<VmdMotion> LoadFromStream(std::ifstream *stream)
		{

			char buffer[30];
			auto result = mmd::make_unique<VmdMotion>();

			// magic and version
			stream->read((char*) buffer, 30);
			if (strncmp(buffer, "Vocaloid Motion Data", 20))
			{
				std::cerr << "invalid vmd file." << std::endl;
				return nullptr;
			}
			result->version = std::atoi(buffer + 20);

			// name
			stream->read(buffer, 20);
			result->model_name = std::string(buffer);

			// bone frames
			int bone_frame_num;
			stream->read((char*) &bone_frame_num, sizeof(int));
			result->bone_frames.resize(bone_frame_num);
			for (int i = 0; i < bone_frame_num; i++)
			{
				result->bone_frames[i].Read(stream);
			}

			// face frames
			int face_frame_num;
			stream->read((char*) &face_frame_num, sizeof(int));
			result->face_frames.resize(face_frame_num);
			for (int i = 0; i < face_frame_num; i++)
			{
				result->face_frames[i].Read(stream);
			}

			// camera frames
			int camera_frame_num;
			stream->read((char*) &camera_frame_num, sizeof(int));
			result->camera_frames.resize(camera_frame_num);
			for (int i = 0; i < camera_frame_num; i++)
			{
				result->camera_frames[i].Read(stream);
			}

			// light frames
			int light_frame_num;
			stream->read((char*) &light_frame_num, sizeof(int));
			result->light_frames.resize(light_frame_num);
			for (int i = 0; i < light_frame_num; i++)
			{
				result->light_frames[i].Read(stream);
			}

			// unknown2
			stream->read(buffer, 4);

			// ik frames
			if (stream->peek() != std::ios::traits_type::eof())
			{
				int ik_num;
				stream->read((char*) &ik_num, sizeof(int));
				result->ik_frames.resize(ik_num);
				for (int i = 0; i < ik_num; i++)
				{
					result->ik_frames[i].Read(stream);
				}
			}

			if (stream->peek() != std::ios::traits_type::eof())
			{
				std::cerr << "vmd stream has unknown data." << std::endl;
			}

			return result;
		}

        bool SaveToFile(const std::u16string& /*filename*/)
		{
			// TODO: How to adapt u16string to string?
			/*
			std::ofstream stream(filename.c_str(), std::ios::binary);
			auto result = SaveToStream(&stream);
			stream.close();
			return result;
			*/
			return false;
		}

		bool SaveToStream(std::ofstream *stream)
		{
			std::string magic = "Vocaloid Motion Data 0002\0";
			magic.resize(30);

			// magic and version
			stream->write(magic.c_str(), 30);

			// name
			stream->write(model_name.c_str(), 20);

			// bone frames
			const int bone_frame_num = static_cast<int>(bone_frames.size());
			stream->write(reinterpret_cast<const char*>(&bone_frame_num), sizeof(int));
			for (int i = 0; i < bone_frame_num; i++)
			{
				bone_frames[i].Write(stream);
			}

			// face frames
			const int face_frame_num = static_cast<int>(face_frames.size());
			stream->write(reinterpret_cast<const char*>(&face_frame_num), sizeof(int));
			for (int i = 0; i < face_frame_num; i++)
			{
				face_frames[i].Write(stream);
			}

			// camera frames
			const int camera_frame_num = static_cast<int>(camera_frames.size());
			stream->write(reinterpret_cast<const char*>(&camera_frame_num), sizeof(int));
			for (int i = 0; i < camera_frame_num; i++)
			{
				camera_frames[i].Write(stream);
			}

			// light frames
			const int light_frame_num = static_cast<int>(light_frames.size());
			stream->write(reinterpret_cast<const char*>(&light_frame_num), sizeof(int));
			for (int i = 0; i < light_frame_num; i++)
			{
				light_frames[i].Write(stream);
			}

			// self shadow datas
			const int self_shadow_num = 0;
			stream->write(reinterpret_cast<const char*>(&self_shadow_num), sizeof(int));

			// ik frames
			const int ik_num = static_cast<int>(ik_frames.size());
			stream->write(reinterpret_cast<const char*>(&ik_num), sizeof(int));
			for (int i = 0; i < ik_num; i++)
			{
				ik_frames[i].Write(stream);
			}

			return true;
		}
	};
}
