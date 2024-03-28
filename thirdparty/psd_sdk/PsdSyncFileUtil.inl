// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

namespace fileUtil
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	inline T ReadFromFile(Ref<StreamPeerBuffer> reader)
	{
		T value = 0;
		reader->get_data((uint8_t *)&value, sizeof(T));
		return value;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	inline T ReadFromFileBE(Ref<StreamPeerBuffer> reader)
	{
		T value = ReadFromFile<T>(reader);
		value = endianUtil::BigEndianToNative(value);
		return value;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	inline void WriteToFile(Ref<StreamPeerBuffer> writer, const T& data)
	{
		writer->put_data((const uint8_t *)&data, sizeof(T));
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	inline void WriteToFileBE(Ref<StreamPeerBuffer> writer, const T& data)
	{
		const T dataBE = endianUtil::NativeToBigEndian(data);
		writer->put_data((const uint8_t *)&dataBE, sizeof(T));
	}
}
