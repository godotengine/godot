
#ifndef B3_BUFFER_INFO_CL_H
#define B3_BUFFER_INFO_CL_H

#include "b3OpenCLArray.h"


struct b3BufferInfoCL
{
	//b3BufferInfoCL(){}

//	template<typename T>
	b3BufferInfoCL(cl_mem buff, bool isReadOnly = false): m_clBuffer(buff), m_isReadOnly(isReadOnly){}

	cl_mem m_clBuffer;
	bool m_isReadOnly;
};

#endif //B3_BUFFER_INFO_CL_H
