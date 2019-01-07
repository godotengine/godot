
#ifndef B3_LAUNCHER_CL_H
#define B3_LAUNCHER_CL_H

#include "b3BufferInfoCL.h"
#include "Bullet3Common/b3MinMax.h"
#include "b3OpenCLArray.h"
#include <stdio.h>

#define B3_DEBUG_SERIALIZE_CL

#ifdef _WIN32
#pragma warning(disable : 4996)
#endif
#define B3_CL_MAX_ARG_SIZE 16
B3_ATTRIBUTE_ALIGNED16(struct)
b3KernelArgData
{
	int m_isBuffer;
	int m_argIndex;
	int m_argSizeInBytes;
	int m_unusedPadding;
	union {
		cl_mem m_clBuffer;
		unsigned char m_argData[B3_CL_MAX_ARG_SIZE];
	};
};

class b3LauncherCL
{
	cl_command_queue m_commandQueue;
	cl_kernel m_kernel;
	int m_idx;

	b3AlignedObjectArray<b3KernelArgData> m_kernelArguments;
	int m_serializationSizeInBytes;
	bool m_enableSerialization;

	const char* m_name;

public:
	b3AlignedObjectArray<b3OpenCLArray<unsigned char>*> m_arrays;

	b3LauncherCL(cl_command_queue queue, cl_kernel kernel, const char* name);

	virtual ~b3LauncherCL();

	void setBuffer(cl_mem clBuffer);

	void setBuffers(b3BufferInfoCL* buffInfo, int n);

	int getSerializationBufferSize() const
	{
		return m_serializationSizeInBytes;
	}

	int deserializeArgs(unsigned char* buf, int bufSize, cl_context ctx);

	inline int validateResults(unsigned char* goldBuffer, int goldBufferCapacity, cl_context ctx);

	int serializeArguments(unsigned char* destBuffer, int destBufferCapacity);

	int getNumArguments() const
	{
		return m_kernelArguments.size();
	}

	b3KernelArgData getArgument(int index)
	{
		return m_kernelArguments[index];
	}

	void serializeToFile(const char* fileName, int numWorkItems);

	template <typename T>
	inline void setConst(const T& consts)
	{
		int sz = sizeof(T);
		b3Assert(sz <= B3_CL_MAX_ARG_SIZE);

		if (m_enableSerialization)
		{
			b3KernelArgData kernelArg;
			kernelArg.m_argIndex = m_idx;
			kernelArg.m_isBuffer = 0;
			T* destArg = (T*)kernelArg.m_argData;
			*destArg = consts;
			kernelArg.m_argSizeInBytes = sizeof(T);
			m_kernelArguments.push_back(kernelArg);
			m_serializationSizeInBytes += sizeof(b3KernelArgData);
		}

		cl_int status = clSetKernelArg(m_kernel, m_idx++, sz, &consts);
		b3Assert(status == CL_SUCCESS);
	}

	inline void launch1D(int numThreads, int localSize = 64)
	{
		launch2D(numThreads, 1, localSize, 1);
	}

	inline void launch2D(int numThreadsX, int numThreadsY, int localSizeX, int localSizeY)
	{
		size_t gRange[3] = {1, 1, 1};
		size_t lRange[3] = {1, 1, 1};
		lRange[0] = localSizeX;
		lRange[1] = localSizeY;
		gRange[0] = b3Max((size_t)1, (numThreadsX / lRange[0]) + (!(numThreadsX % lRange[0]) ? 0 : 1));
		gRange[0] *= lRange[0];
		gRange[1] = b3Max((size_t)1, (numThreadsY / lRange[1]) + (!(numThreadsY % lRange[1]) ? 0 : 1));
		gRange[1] *= lRange[1];

		cl_int status = clEnqueueNDRangeKernel(m_commandQueue,
											   m_kernel, 2, NULL, gRange, lRange, 0, 0, 0);
		if (status != CL_SUCCESS)
		{
			printf("Error: OpenCL status = %d\n", status);
		}
		b3Assert(status == CL_SUCCESS);
	}

	void enableSerialization(bool serialize)
	{
		m_enableSerialization = serialize;
	}
};

#endif  //B3_LAUNCHER_CL_H
