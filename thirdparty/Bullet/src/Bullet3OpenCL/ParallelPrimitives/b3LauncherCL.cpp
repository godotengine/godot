#include "b3LauncherCL.h"

bool gDebugLauncherCL = false;
    
b3LauncherCL::b3LauncherCL(cl_command_queue queue, cl_kernel kernel, const char* name)
:m_commandQueue(queue),
m_kernel(kernel),
m_idx(0),
m_enableSerialization(false),
m_name(name)
{
	if (gDebugLauncherCL)
	{
		static int counter = 0;
		printf("[%d] Prepare to launch OpenCL kernel %s\n", counter++, name);
	}

      m_serializationSizeInBytes = sizeof(int);
}
    
b3LauncherCL::~b3LauncherCL()
  {
      for (int i=0;i<m_arrays.size();i++)
      {
		  delete (m_arrays[i]);
      }

	  m_arrays.clear();
	  if (gDebugLauncherCL)
	  {
		static int counter = 0;
		printf("[%d] Finished launching OpenCL kernel %s\n", counter++,m_name);
	  }
  }

void b3LauncherCL::setBuffer( cl_mem clBuffer)
{
		if (m_enableSerialization)
		{
			b3KernelArgData kernelArg;
			kernelArg.m_argIndex = m_idx;
			kernelArg.m_isBuffer = 1;
			kernelArg.m_clBuffer = clBuffer;
		
			cl_mem_info param_name = CL_MEM_SIZE;
			size_t param_value;
			size_t sizeInBytes = sizeof(size_t);
			size_t actualSizeInBytes;
			cl_int err;
			err = clGetMemObjectInfo (	kernelArg.m_clBuffer,
									  param_name,
									  sizeInBytes,
									  &param_value,
									  &actualSizeInBytes);
			
			b3Assert( err == CL_SUCCESS );
			kernelArg.m_argSizeInBytes = param_value;
			
			m_kernelArguments.push_back(kernelArg);
			m_serializationSizeInBytes+= sizeof(b3KernelArgData);
			m_serializationSizeInBytes+=param_value;
            }
            cl_int status = clSetKernelArg( m_kernel, m_idx++, sizeof(cl_mem), &clBuffer);
		b3Assert( status == CL_SUCCESS );
}


void b3LauncherCL::setBuffers( b3BufferInfoCL* buffInfo, int n )
{
	for(int i=0; i<n; i++)
	{
		if (m_enableSerialization)
		{
			b3KernelArgData kernelArg;
			kernelArg.m_argIndex = m_idx;
			kernelArg.m_isBuffer = 1;
			kernelArg.m_clBuffer = buffInfo[i].m_clBuffer;
		
			cl_mem_info param_name = CL_MEM_SIZE;
			size_t param_value;
			size_t sizeInBytes = sizeof(size_t);
			size_t actualSizeInBytes;
			cl_int err;
			err = clGetMemObjectInfo (	kernelArg.m_clBuffer,
									  param_name,
									  sizeInBytes,
									  &param_value,
									  &actualSizeInBytes);
			
			b3Assert( err == CL_SUCCESS );
			kernelArg.m_argSizeInBytes = param_value;
			
			m_kernelArguments.push_back(kernelArg);
			m_serializationSizeInBytes+= sizeof(b3KernelArgData);
			m_serializationSizeInBytes+=param_value;
            }
            cl_int status = clSetKernelArg( m_kernel, m_idx++, sizeof(cl_mem), &buffInfo[i].m_clBuffer);
		b3Assert( status == CL_SUCCESS );
        }
}

struct b3KernelArgDataUnaligned
{
    int m_isBuffer;
    int m_argIndex;
    int m_argSizeInBytes;
	int m_unusedPadding;
    union
    {
        cl_mem m_clBuffer;
        unsigned char m_argData[B3_CL_MAX_ARG_SIZE];
    };
    
};
#include <string.h>



int b3LauncherCL::deserializeArgs(unsigned char* buf, int bufSize, cl_context ctx)
{
    int index=0;
    
    int numArguments = *(int*) &buf[index];
    index+=sizeof(int);
    
    for (int i=0;i<numArguments;i++)
    {
        b3KernelArgDataUnaligned* arg = (b3KernelArgDataUnaligned*)&buf[index];

        index+=sizeof(b3KernelArgData);
        if (arg->m_isBuffer)
        {
            b3OpenCLArray<unsigned char>* clData = new b3OpenCLArray<unsigned char>(ctx,m_commandQueue, arg->m_argSizeInBytes);
            clData->resize(arg->m_argSizeInBytes);
            
            clData->copyFromHostPointer(&buf[index], arg->m_argSizeInBytes);
            
            arg->m_clBuffer = clData->getBufferCL();
            
            m_arrays.push_back(clData);
            
            cl_int status = clSetKernelArg( m_kernel, m_idx++, sizeof(cl_mem), &arg->m_clBuffer);
		b3Assert( status == CL_SUCCESS );
            index+=arg->m_argSizeInBytes;
        } else 
        {
            cl_int status = clSetKernelArg( m_kernel, m_idx++, arg->m_argSizeInBytes, &arg->m_argData);
		b3Assert( status == CL_SUCCESS );
        }
		b3KernelArgData b;
		memcpy(&b,arg,sizeof(b3KernelArgDataUnaligned));
	m_kernelArguments.push_back(b);
    }
m_serializationSizeInBytes = index;
    return index;
}

int b3LauncherCL::validateResults(unsigned char* goldBuffer, int goldBufferCapacity, cl_context ctx)
  {
	 int index=0;
      
      int numArguments = *(int*) &goldBuffer[index];
      index+=sizeof(int);

	if (numArguments != m_kernelArguments.size())
	{
		printf("failed validation: expected %d arguments, found %d\n",numArguments, m_kernelArguments.size());
		return -1;
	}
      
      for (int ii=0;ii<numArguments;ii++)
      {
          b3KernelArgData* argGold = (b3KernelArgData*)&goldBuffer[index];

		if (m_kernelArguments[ii].m_argSizeInBytes != argGold->m_argSizeInBytes)
		{
			printf("failed validation: argument %d sizeInBytes expected: %d, found %d\n",ii, argGold->m_argSizeInBytes, m_kernelArguments[ii].m_argSizeInBytes);
			return -2;
		}

		{
			int expected = argGold->m_isBuffer;
			int found = m_kernelArguments[ii].m_isBuffer;

			if (expected != found)
			{
				printf("failed validation: argument %d isBuffer expected: %d, found %d\n",ii,expected, found);
				return -3;
			}
		}
		index+=sizeof(b3KernelArgData);

		if (argGold->m_isBuffer)
          {

			unsigned char* memBuf= (unsigned char*) malloc(m_kernelArguments[ii].m_argSizeInBytes);
			unsigned char* goldBuf = &goldBuffer[index];
			for (int j=0;j<m_kernelArguments[j].m_argSizeInBytes;j++)
			{
				memBuf[j] = 0xaa;
			}

			cl_int status = 0;
			status = clEnqueueReadBuffer( m_commandQueue, m_kernelArguments[ii].m_clBuffer, CL_TRUE, 0, m_kernelArguments[ii].m_argSizeInBytes,
                                           memBuf, 0,0,0 );
              b3Assert( status==CL_SUCCESS );
              clFinish(m_commandQueue);

			for (int b=0;b<m_kernelArguments[ii].m_argSizeInBytes;b++)
			{
				int expected = goldBuf[b];
				int found = memBuf[b];
				if (expected != found)
				{
					printf("failed validation: argument %d OpenCL data at byte position %d expected: %d, found %d\n",
						ii, b, expected, found);
					return -4;
				}
			}

              
              index+=argGold->m_argSizeInBytes;
          } else 
          {
			
			//compare content
			for (int b=0;b<m_kernelArguments[ii].m_argSizeInBytes;b++)
			{
				int expected = argGold->m_argData[b];
				int found =m_kernelArguments[ii].m_argData[b];
				if (expected != found)
				{
					printf("failed validation: argument %d const data at byte position %d expected: %d, found %d\n",
						ii, b, expected, found);
					return -5;
				}
			}

          }
      }
      return index;

}

int b3LauncherCL::serializeArguments(unsigned char* destBuffer, int destBufferCapacity)
{
//initialize to known values
for (int i=0;i<destBufferCapacity;i++)
	destBuffer[i] = 0xec;

    assert(destBufferCapacity>=m_serializationSizeInBytes);
    
    //todo: use the b3Serializer for this to allow for 32/64bit, endianness etc        
    int numArguments = m_kernelArguments.size();
    int curBufferSize = 0;
    int* dest = (int*)&destBuffer[curBufferSize];
    *dest = numArguments;
    curBufferSize += sizeof(int);
    
    
    
    for (int i=0;i<this->m_kernelArguments.size();i++)
    {
        b3KernelArgData* arg = (b3KernelArgData*) &destBuffer[curBufferSize];
        *arg = m_kernelArguments[i];
        curBufferSize+=sizeof(b3KernelArgData);
        if (arg->m_isBuffer==1)
        {
            //copy the OpenCL buffer content
            cl_int status = 0;
            status = clEnqueueReadBuffer( m_commandQueue, arg->m_clBuffer, 0, 0, arg->m_argSizeInBytes,
                                         &destBuffer[curBufferSize], 0,0,0 );
            b3Assert( status==CL_SUCCESS );
            clFinish(m_commandQueue);
            curBufferSize+=arg->m_argSizeInBytes;
        }
        
    }
    return curBufferSize;
}

void b3LauncherCL::serializeToFile(const char* fileName, int numWorkItems)
{
	int num = numWorkItems;
	int buffSize = getSerializationBufferSize();
	unsigned char* buf = new unsigned char[buffSize+sizeof(int)];
	for (int i=0;i<buffSize+1;i++)
	{
		unsigned char* ptr = (unsigned char*)&buf[i];
		*ptr = 0xff;
	}
//	int actualWrite = serializeArguments(buf,buffSize);
              
//	unsigned char* cptr = (unsigned char*)&buf[buffSize];
//            printf("buf[buffSize] = %d\n",*cptr);
              
	assert(buf[buffSize]==0xff);//check for buffer overrun
	int* ptr = (int*)&buf[buffSize];
              
	*ptr = num;
              
	FILE* f = fopen(fileName,"wb");
	fwrite(buf,buffSize+sizeof(int),1,f);
	fclose(f);

	delete[] buf;
}		

