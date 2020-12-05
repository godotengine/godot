
#ifndef B3_PREFIX_SCAN_CL_H
#define B3_PREFIX_SCAN_CL_H

#include "b3OpenCLArray.h"
#include "b3BufferInfoCL.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Vector3.h"

class b3PrefixScanFloat4CL
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

	b3OpenCLArray<b3Vector3>* m_workBuffer;

public:
	b3PrefixScanFloat4CL(cl_context ctx, cl_device_id device, cl_command_queue queue, int size = 0);

	virtual ~b3PrefixScanFloat4CL();

	void execute(b3OpenCLArray<b3Vector3>& src, b3OpenCLArray<b3Vector3>& dst, int n, b3Vector3* sum = 0);
	void executeHost(b3AlignedObjectArray<b3Vector3>& src, b3AlignedObjectArray<b3Vector3>& dst, int n, b3Vector3* sum);
};

#endif  //B3_PREFIX_SCAN_CL_H
