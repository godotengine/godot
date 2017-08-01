#include "b3FillCL.h"
#include "Bullet3OpenCL/Initialize/b3OpenCLUtils.h"
#include "b3BufferInfoCL.h"
#include "b3LauncherCL.h"

#define FILL_CL_PROGRAM_PATH "src/Bullet3OpenCL/ParallelPrimitives/kernels/FillKernels.cl"

#include "kernels/FillKernelsCL.h"

b3FillCL::b3FillCL(cl_context ctx, cl_device_id device, cl_command_queue queue)
:m_commandQueue(queue)
{
	const char* kernelSource = fillKernelsCL;
	cl_int pErrNum;
	const char* additionalMacros = "";

	cl_program fillProg = b3OpenCLUtils::compileCLProgramFromString( ctx, device, kernelSource, &pErrNum,additionalMacros, FILL_CL_PROGRAM_PATH);
	b3Assert(fillProg);

	m_fillIntKernel = b3OpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "FillIntKernel", &pErrNum, fillProg,additionalMacros );
	b3Assert(m_fillIntKernel);

	m_fillUnsignedIntKernel = b3OpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "FillUnsignedIntKernel", &pErrNum, fillProg,additionalMacros );
	b3Assert(m_fillIntKernel);

	m_fillFloatKernel = b3OpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "FillFloatKernel", &pErrNum, fillProg,additionalMacros );
	b3Assert(m_fillFloatKernel);

	

	m_fillKernelInt2 = b3OpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "FillInt2Kernel", &pErrNum, fillProg,additionalMacros );
	b3Assert(m_fillKernelInt2);
	
}

b3FillCL::~b3FillCL()
{
	clReleaseKernel(m_fillKernelInt2);
	clReleaseKernel(m_fillIntKernel);
	clReleaseKernel(m_fillUnsignedIntKernel);
	clReleaseKernel(m_fillFloatKernel);

}

void b3FillCL::execute(b3OpenCLArray<float>& src, const float value, int n, int offset)
{
	b3Assert( n>0 );

	{
		b3LauncherCL launcher( m_commandQueue, m_fillFloatKernel,"m_fillFloatKernel" );
		launcher.setBuffer( src.getBufferCL());
		launcher.setConst( n );
		launcher.setConst( value );
		launcher.setConst( offset);

		launcher.launch1D( n );
	}
}

void b3FillCL::execute(b3OpenCLArray<int>& src, const int value, int n, int offset)
{
	b3Assert( n>0 );
	

	{
		b3LauncherCL launcher( m_commandQueue, m_fillIntKernel ,"m_fillIntKernel");
		launcher.setBuffer(src.getBufferCL());
		launcher.setConst( n);
		launcher.setConst( value);
		launcher.setConst( offset);
		launcher.launch1D( n );
	}
}


void b3FillCL::execute(b3OpenCLArray<unsigned int>& src, const unsigned int value, int n, int offset)
{
	b3Assert( n>0 );

	{
		b3BufferInfoCL bInfo[] = { b3BufferInfoCL( src.getBufferCL() ) };

		b3LauncherCL launcher( m_commandQueue, m_fillUnsignedIntKernel,"m_fillUnsignedIntKernel" );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
		launcher.setConst( n );
        launcher.setConst(value);
		launcher.setConst(offset);

		launcher.launch1D( n );
	}
}

void b3FillCL::executeHost(b3AlignedObjectArray<b3Int2> &src, const b3Int2 &value, int n, int offset)
{
	for (int i=0;i<n;i++)
	{
		src[i+offset]=value;
	}
}

void b3FillCL::executeHost(b3AlignedObjectArray<int> &src, const int value, int n, int offset)
{
	for (int i=0;i<n;i++)
	{
		src[i+offset]=value;
	}
}

void b3FillCL::execute(b3OpenCLArray<b3Int2> &src, const b3Int2 &value, int n, int offset)
{
	b3Assert( n>0 );
	

	{
		b3BufferInfoCL bInfo[] = { b3BufferInfoCL( src.getBufferCL() ) };

		b3LauncherCL launcher(m_commandQueue, m_fillKernelInt2,"m_fillKernelInt2");
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
		launcher.setConst(n);
		launcher.setConst(value);
		launcher.setConst(offset);

		//( constBuffer );
		launcher.launch1D( n );
	}
}
