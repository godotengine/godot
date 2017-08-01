#ifndef B3_FILL_CL_H
#define B3_FILL_CL_H

#include "b3OpenCLArray.h"
#include "Bullet3Common/b3Scalar.h"

#include "Bullet3Common/shared/b3Int2.h"
#include "Bullet3Common/shared/b3Int4.h"


class b3FillCL
{
	
	cl_command_queue	m_commandQueue;
	
	cl_kernel			m_fillKernelInt2;
	cl_kernel			m_fillIntKernel;
	cl_kernel			m_fillUnsignedIntKernel;
	cl_kernel			m_fillFloatKernel;

	public:
		
		struct b3ConstData
		{
			union
			{
				b3Int4 m_data;
				b3UnsignedInt4 m_UnsignedData;
			};
			int m_offset;
			int m_n;
			int m_padding[2];
		};

protected:

public:

		b3FillCL(cl_context ctx, cl_device_id device, cl_command_queue queue);

		virtual ~b3FillCL();

		void execute(b3OpenCLArray<unsigned int>& src, const unsigned int value, int n, int offset = 0);
	
		void execute(b3OpenCLArray<int>& src, const int value, int n, int offset = 0);

		void execute(b3OpenCLArray<float>& src, const float value, int n, int offset = 0);

		void execute(b3OpenCLArray<b3Int2>& src, const b3Int2& value, int n, int offset = 0);

		void executeHost(b3AlignedObjectArray<b3Int2> &src, const b3Int2 &value, int n, int offset);

		void executeHost(b3AlignedObjectArray<int> &src, const int value, int n, int offset);

	//	void execute(b3OpenCLArray<b3Int4>& src, const b3Int4& value, int n, int offset = 0);

};
		
		
		
	

#endif //B3_FILL_CL_H
