
#ifndef B3_PREFIX_SCAN_CL_H
#define B3_PREFIX_SCAN_CL_H

#include "b3OpenCLArray.h"
#include "b3BufferInfoCL.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

class b3PrefixScanCL
{
	enum
	{
		BLOCK_SIZE = 128
	};

	//	Option m_option;

	cl_command_queue m_commandQueue;

	cl_kernel m_localScanKernel;
	cl_kernel m_blockSumKernel;
	cl_kernel m_propagationKernel;

	b3OpenCLArray<unsigned int>* m_workBuffer;

public:
	b3PrefixScanCL(cl_context ctx, cl_device_id device, cl_command_queue queue, int size = 0);

	virtual ~b3PrefixScanCL();

	void execute(b3OpenCLArray<unsigned int>& src, b3OpenCLArray<unsigned int>& dst, int n, unsigned int* sum = 0);
	void executeHost(b3AlignedObjectArray<unsigned int>& src, b3AlignedObjectArray<unsigned int>& dst, int n, unsigned int* sum = 0);
};

#endif  //B3_PREFIX_SCAN_CL_H
