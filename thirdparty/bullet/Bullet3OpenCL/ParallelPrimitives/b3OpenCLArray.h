#ifndef B3_OPENCL_ARRAY_H
#define B3_OPENCL_ARRAY_H

#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"

template <typename T>
class b3OpenCLArray
{
	size_t m_size;
	size_t m_capacity;
	cl_mem m_clBuffer;

	cl_context m_clContext;
	cl_command_queue m_commandQueue;

	bool m_ownsMemory;

	bool m_allowGrowingCapacity;

	void deallocate()
	{
		if (m_clBuffer && m_ownsMemory)
		{
			clReleaseMemObject(m_clBuffer);
		}
		m_clBuffer = 0;
		m_capacity = 0;
	}

	b3OpenCLArray<T>& operator=(const b3OpenCLArray<T>& src);

	B3_FORCE_INLINE size_t allocSize(size_t size)
	{
		return (size ? size * 2 : 1);
	}

public:
	b3OpenCLArray(cl_context ctx, cl_command_queue queue, size_t initialCapacity = 0, bool allowGrowingCapacity = true)
		: m_size(0), m_capacity(0), m_clBuffer(0), m_clContext(ctx), m_commandQueue(queue), m_ownsMemory(true), m_allowGrowingCapacity(true)
	{
		if (initialCapacity)
		{
			reserve(initialCapacity);
		}
		m_allowGrowingCapacity = allowGrowingCapacity;
	}

	///this is an error-prone method with no error checking, be careful!
	void setFromOpenCLBuffer(cl_mem buffer, size_t sizeInElements)
	{
		deallocate();
		m_ownsMemory = false;
		m_allowGrowingCapacity = false;
		m_clBuffer = buffer;
		m_size = sizeInElements;
		m_capacity = sizeInElements;
	}

	// we could enable this assignment, but need to make sure to avoid accidental deep copies
	//	b3OpenCLArray<T>& operator=(const b3AlignedObjectArray<T>& src)
	//	{
	//		copyFromArray(src);
	//		return *this;
	//	}

	cl_mem getBufferCL() const
	{
		return m_clBuffer;
	}

	virtual ~b3OpenCLArray()
	{
		deallocate();
		m_size = 0;
		m_capacity = 0;
	}

	B3_FORCE_INLINE bool push_back(const T& _Val, bool waitForCompletion = true)
	{
		bool result = true;
		size_t sz = size();
		if (sz == capacity())
		{
			result = reserve(allocSize(size()));
		}
		copyFromHostPointer(&_Val, 1, sz, waitForCompletion);
		m_size++;
		return result;
	}

	B3_FORCE_INLINE T forcedAt(size_t n) const
	{
		b3Assert(n >= 0);
		b3Assert(n < capacity());
		T elem;
		copyToHostPointer(&elem, 1, n, true);
		return elem;
	}

	B3_FORCE_INLINE T at(size_t n) const
	{
		b3Assert(n >= 0);
		b3Assert(n < size());
		T elem;
		copyToHostPointer(&elem, 1, n, true);
		return elem;
	}

	B3_FORCE_INLINE bool resize(size_t newsize, bool copyOldContents = true)
	{
		bool result = true;
		size_t curSize = size();

		if (newsize < curSize)
		{
			//leave the OpenCL memory for now
		}
		else
		{
			if (newsize > size())
			{
				result = reserve(newsize, copyOldContents);
			}

			//leave new data uninitialized (init in debug mode?)
			//for (size_t i=curSize;i<newsize;i++) ...
		}

		if (result)
		{
			m_size = newsize;
		}
		else
		{
			m_size = 0;
		}
		return result;
	}

	B3_FORCE_INLINE size_t size() const
	{
		return m_size;
	}

	B3_FORCE_INLINE size_t capacity() const
	{
		return m_capacity;
	}

	B3_FORCE_INLINE bool reserve(size_t _Count, bool copyOldContents = true)
	{
		bool result = true;
		// determine new minimum length of allocated storage
		if (capacity() < _Count)
		{  // not enough room, reallocate

			if (m_allowGrowingCapacity)
			{
				cl_int ciErrNum;
				//create a new OpenCL buffer
				size_t memSizeInBytes = sizeof(T) * _Count;
				cl_mem buf = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, memSizeInBytes, NULL, &ciErrNum);
				if (ciErrNum != CL_SUCCESS)
				{
					b3Error("OpenCL out-of-memory\n");
					_Count = 0;
					result = false;
				}
//#define B3_ALWAYS_INITIALIZE_OPENCL_BUFFERS
#ifdef B3_ALWAYS_INITIALIZE_OPENCL_BUFFERS
				unsigned char* src = (unsigned char*)malloc(memSizeInBytes);
				for (size_t i = 0; i < memSizeInBytes; i++)
					src[i] = 0xbb;
				ciErrNum = clEnqueueWriteBuffer(m_commandQueue, buf, CL_TRUE, 0, memSizeInBytes, src, 0, 0, 0);
				b3Assert(ciErrNum == CL_SUCCESS);
				clFinish(m_commandQueue);
				free(src);
#endif  //B3_ALWAYS_INITIALIZE_OPENCL_BUFFERS

				if (result)
				{
					if (copyOldContents)
						copyToCL(buf, size());
				}

				//deallocate the old buffer
				deallocate();

				m_clBuffer = buf;

				m_capacity = _Count;
			}
			else
			{
				//fail: assert and
				b3Assert(0);
				deallocate();
				result = false;
			}
		}
		return result;
	}

	void copyToCL(cl_mem destination, size_t numElements, size_t firstElem = 0, size_t dstOffsetInElems = 0) const
	{
		if (numElements <= 0)
			return;

		b3Assert(m_clBuffer);
		b3Assert(destination);

		//likely some error, destination is same as source
		b3Assert(m_clBuffer != destination);

		b3Assert((firstElem + numElements) <= m_size);

		cl_int status = 0;

		b3Assert(numElements > 0);
		b3Assert(numElements <= m_size);

		size_t srcOffsetBytes = sizeof(T) * firstElem;
		size_t dstOffsetInBytes = sizeof(T) * dstOffsetInElems;

		status = clEnqueueCopyBuffer(m_commandQueue, m_clBuffer, destination,
									 srcOffsetBytes, dstOffsetInBytes, sizeof(T) * numElements, 0, 0, 0);

		b3Assert(status == CL_SUCCESS);
	}

	void copyFromHost(const b3AlignedObjectArray<T>& srcArray, bool waitForCompletion = true)
	{
		size_t newSize = srcArray.size();

		bool copyOldContents = false;
		resize(newSize, copyOldContents);
		if (newSize)
			copyFromHostPointer(&srcArray[0], newSize, 0, waitForCompletion);
	}

	void copyFromHostPointer(const T* src, size_t numElems, size_t destFirstElem = 0, bool waitForCompletion = true)
	{
		b3Assert(numElems + destFirstElem <= capacity());

		if (numElems + destFirstElem)
		{
			cl_int status = 0;
			size_t sizeInBytes = sizeof(T) * numElems;
			status = clEnqueueWriteBuffer(m_commandQueue, m_clBuffer, 0, sizeof(T) * destFirstElem, sizeInBytes,
										  src, 0, 0, 0);
			b3Assert(status == CL_SUCCESS);
			if (waitForCompletion)
				clFinish(m_commandQueue);
		}
		else
		{
			b3Error("copyFromHostPointer invalid range\n");
		}
	}

	void copyToHost(b3AlignedObjectArray<T>& destArray, bool waitForCompletion = true) const
	{
		destArray.resize(this->size());
		if (size())
			copyToHostPointer(&destArray[0], size(), 0, waitForCompletion);
	}

	void copyToHostPointer(T* destPtr, size_t numElem, size_t srcFirstElem = 0, bool waitForCompletion = true) const
	{
		b3Assert(numElem + srcFirstElem <= capacity());

		if (numElem + srcFirstElem <= capacity())
		{
			cl_int status = 0;
			status = clEnqueueReadBuffer(m_commandQueue, m_clBuffer, 0, sizeof(T) * srcFirstElem, sizeof(T) * numElem,
										 destPtr, 0, 0, 0);
			b3Assert(status == CL_SUCCESS);

			if (waitForCompletion)
				clFinish(m_commandQueue);
		}
		else
		{
			b3Error("copyToHostPointer invalid range\n");
		}
	}

	void copyFromOpenCLArray(const b3OpenCLArray& src)
	{
		size_t newSize = src.size();
		resize(newSize);
		if (size())
		{
			src.copyToCL(m_clBuffer, size());
		}
	}
};

#endif  //B3_OPENCL_ARRAY_H
