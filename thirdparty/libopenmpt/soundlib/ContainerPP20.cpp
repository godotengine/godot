/*
 * ContainerPP20.cpp
 * -----------------
 * Purpose: Handling of PowerPack PP20 compressed modules
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#include "../common/FileReader.h"
#include "Container.h"
#include "Sndfile.h"

#include <stdexcept>


OPENMPT_NAMESPACE_BEGIN


//#define MMCMP_LOG


struct PPBITBUFFER
{
	uint32 bitcount;
	uint32 bitbuffer;
	const uint8 *pStart;
	const uint8 *pSrc;

	uint32 GetBits(uint32 n);
};


uint32 PPBITBUFFER::GetBits(uint32 n)
{
	uint32 result = 0;

	for (uint32 i=0; i<n; i++)
	{
		if (!bitcount)
		{
			bitcount = 8;
			if (pSrc != pStart) pSrc--;
			bitbuffer = *pSrc;
		}
		result = (result<<1) | (bitbuffer&1);
		bitbuffer >>= 1;
		bitcount--;
	}
	return result;
}


static bool PP20_DoUnpack(const uint8 *pSrc, uint32 nSrcLen, uint8 *pDst, uint32 nDstLen)
{
	PPBITBUFFER BitBuffer;
	uint32 nBytesLeft;

	BitBuffer.pStart = pSrc;
	BitBuffer.pSrc = pSrc + nSrcLen - 4;
	BitBuffer.bitbuffer = 0;
	BitBuffer.bitcount = 0;
	BitBuffer.GetBits(pSrc[nSrcLen-1]);
	nBytesLeft = nDstLen;
	while (nBytesLeft > 0)
	{
		if (!BitBuffer.GetBits(1))
		{
			uint32 n = 1;
			while (n < nBytesLeft)
			{
				uint32 code = BitBuffer.GetBits(2);
				n += code;
				if (code != 3) break;
			}
			LimitMax(n, nBytesLeft);
			for (uint32 i=0; i<n; i++)
			{
				pDst[--nBytesLeft] = (uint8)BitBuffer.GetBits(8);
			}
			if (!nBytesLeft) break;
		}
		{
			uint32 n = BitBuffer.GetBits(2)+1;
			if(n < 1 || n-1 >= nSrcLen) return false;
			uint32 nbits = pSrc[n-1];
			uint32 nofs;
			if (n==4)
			{
				nofs = BitBuffer.GetBits( (BitBuffer.GetBits(1)) ? nbits : 7 );
				while (n < nBytesLeft)
				{
					uint32 code = BitBuffer.GetBits(3);
					n += code;
					if (code != 7) break;
				}
			} else
			{
				nofs = BitBuffer.GetBits(nbits);
			}
			LimitMax(n, nBytesLeft);
			for (uint32 i=0; i<=n; i++)
			{
				pDst[nBytesLeft-1] = (nBytesLeft+nofs < nDstLen) ? pDst[nBytesLeft+nofs] : 0;
				if (!--nBytesLeft) break;
			}
		}
	}
	return true;
}


struct PP20header
{
	char    magic[4];       // "PP20"
	uint8be efficiency[4];
};

MPT_BINARY_STRUCT(PP20header, 8)


static bool ValidateHeader(const PP20header &hdr)
{
	if(std::memcmp(hdr.magic, "PP20", 4) != 0)
	{
		return false;
	}
	if(hdr.efficiency[0] < 9 || hdr.efficiency[0] > 15
		|| hdr.efficiency[1] < 9 || hdr.efficiency[1] > 15
		|| hdr.efficiency[2] < 9 || hdr.efficiency[2] > 15
		|| hdr.efficiency[3] < 9 || hdr.efficiency[3] > 15)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderPP20(MemoryFileReader file, const uint64 *pfilesize)
{
	PP20header hdr;
	if(!file.ReadStruct(hdr))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(hdr))
	{
		return ProbeFailure;
	}
	MPT_UNREFERENCED_PARAMETER(pfilesize);
	return ProbeSuccess;
}


bool UnpackPP20(std::vector<ContainerItem> &containerItems, FileReader &file, ContainerLoadingFlags loadFlags)
{
	file.Rewind();
	containerItems.clear();

	PP20header hdr;
	if(!file.ReadStruct(hdr))
	{
		return false;
	}
	if(!ValidateHeader(hdr))
	{
		return false;
	}
	if(loadFlags == ContainerOnlyVerifyHeader)
	{
		return true;
	}

	if(!file.CanRead(4))
	{
		return false;
	}

	containerItems.emplace_back();
	containerItems.back().data_cache = mpt::make_unique<std::vector<char> >();
	std::vector<char> & unpackedData = *(containerItems.back().data_cache);

	FileReader::off_t length = file.GetLength();
	if(!Util::TypeCanHoldValue<uint32>(length)) return false;
	// Length word must be aligned
	if((length % 2u) != 0)
		return false;

	file.Seek(length - 4);
	uint32 dstLen = 0;
	dstLen |= file.ReadUint8() << 16;
	dstLen |= file.ReadUint8() << 8;
	dstLen |= file.ReadUint8() << 0;
	if(dstLen == 0) return false;
	try
	{
		unpackedData.resize(dstLen);
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
		return false;
	}
	file.Seek(4);
	bool result = PP20_DoUnpack(file.GetRawData<uint8>(), static_cast<uint32>(length - 4), mpt::byte_cast<uint8 *>(unpackedData.data()), dstLen);

	if(result)
	{
		containerItems.back().file = FileReader(mpt::byte_cast<mpt::const_byte_span>(mpt::as_span(unpackedData)));
	}

	return result;
}


OPENMPT_NAMESPACE_END
