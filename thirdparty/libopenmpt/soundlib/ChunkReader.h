/*
 * ChunkReader.h
 * -------------
 * Purpose: An extended FileReader to read Iff-like chunk-based file structures.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../common/FileReader.h"

#include <vector>


OPENMPT_NAMESPACE_BEGIN


class ChunkReader : public FileReader
{
public:

	template <typename Tbyte> ChunkReader(mpt::span<Tbyte> bytedata) : FileReader(bytedata) { }
	ChunkReader(const FileReader &other) : FileReader(other) { }
	ChunkReader(FileReader &&other) : FileReader(std::move(other)) { }

	template<typename T>
	class Item
	{
	private:
		T chunkHeader;
		FileReader chunkData;

	public:
		Item(const T &header, FileReader &&data) : chunkHeader(header), chunkData(std::move(data)) { }
		Item(const Item<T> &) = default;
		Item(Item<T> &&) noexcept = default;

		const T &GetHeader() const { return chunkHeader; }
		const FileReader &GetData() const { return chunkData; }
	};

	template<typename T>
	class ChunkList : public std::vector<Item<T>>
	{
	public:
		typedef decltype(T().GetID()) id_type;

		// Check if the list contains a given chunk.
		bool ChunkExists(id_type id) const
		{
			return std::find_if(this->cbegin(), this->cend(), [&id](const Item<T> &item) { return item.GetHeader().GetID() == id; }) != this->cend();
		}

		// Retrieve the first chunk with a given ID.
		FileReader GetChunk(id_type id) const
		{
			auto item = std::find_if(this->cbegin(), this->cend(), [&id](const Item<T> &item) { return item.GetHeader().GetID() == id; });
			if(item != this->cend())
				return item->GetData();
			return FileReader();
		}

		// Retrieve all chunks with a given ID.
		std::vector<FileReader> GetAllChunks(id_type id) const
		{
			std::vector<FileReader> result;
			for(const auto &item : *this)
			{
				if(item.GetHeader().GetID() == id)
				{
					result.push_back(item.GetData());
				}
			}
			return result;
		}
	};

	// Read a single "T" chunk.
	// T is required to have the methods GetID() and GetLength().
	// GetLength() must return the chunk size in bytes, and GetID() the chunk ID.
	template<typename T>
	Item<T> GetNextChunk(off_t padding)
	{
		T chunkHeader;
		off_t dataSize = 0;
		if(Read(chunkHeader))
		{
			dataSize = chunkHeader.GetLength();
		}
		Item<T> resultItem(chunkHeader, ReadChunk(dataSize));

		// Skip padding bytes
		if(padding != 0 && dataSize % padding != 0)
		{
			Skip(padding - (dataSize % padding));
		}

		return resultItem;
	}

	// Read a series of "T" chunks until the end of file is reached.
	// T is required to have the methods GetID() and GetLength().
	// GetLength() must return the chunk size in bytes, and GetID() the chunk ID.
	template<typename T>
	ChunkList<T> ReadChunks(off_t padding)
	{
		ChunkList<T> result;
		while(CanRead(sizeof(T)))
		{
			result.push_back(GetNextChunk<T>(padding));
		}

		return result;
	}

	// Read a series of "T" chunks until a given chunk ID is found.
	// T is required to have the methods GetID() and GetLength().
	// GetLength() must return the chunk size in bytes, and GetID() the chunk ID.
	template<typename T>
	ChunkList<T> ReadChunksUntil(off_t padding, decltype(T().GetID()) stopAtID)
	{
		ChunkList<T> result;
		while(CanRead(sizeof(T)))
		{
			result.push_back(GetNextChunk<T>(padding));
			if(result.back().GetHeader().GetID() == stopAtID)
			{
				break;
			}
		}

		return result;
	}

};


OPENMPT_NAMESPACE_END
