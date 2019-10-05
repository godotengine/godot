
#ifndef B3_RESIZABLE_POOL_H
#define B3_RESIZABLE_POOL_H

#include "Bullet3Common/b3AlignedObjectArray.h"

enum
{
	B3_POOL_HANDLE_TERMINAL_FREE = -1,
	B3_POOL_HANDLE_TERMINAL_USED = -2
};

template <typename U>
struct b3PoolBodyHandle : public U
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	int m_nextFreeHandle;
	void setNextFree(int next)
	{
		m_nextFreeHandle = next;
	}
	int getNextFree() const
	{
		return m_nextFreeHandle;
	}
};

template <typename T>
class b3ResizablePool
{
protected:
	b3AlignedObjectArray<T> m_bodyHandles;
	int m_numUsedHandles;   // number of active handles
	int m_firstFreeHandle;  // free handles list

	T* getHandleInternal(int handle)
	{
		return &m_bodyHandles[handle];
	}
	const T* getHandleInternal(int handle) const
	{
		return &m_bodyHandles[handle];
	}

public:
	b3ResizablePool()
	{
		initHandles();
	}

	virtual ~b3ResizablePool()
	{
		exitHandles();
	}
	///handle management

	int getNumHandles() const
	{
		return m_bodyHandles.size();
	}

	void getUsedHandles(b3AlignedObjectArray<int>& usedHandles) const
	{
		for (int i = 0; i < m_bodyHandles.size(); i++)
		{
			if (m_bodyHandles[i].getNextFree() == B3_POOL_HANDLE_TERMINAL_USED)
			{
				usedHandles.push_back(i);
			}
		}
	}

	T* getHandle(int handle)
	{
		b3Assert(handle >= 0);
		b3Assert(handle < m_bodyHandles.size());
		if ((handle < 0) || (handle >= m_bodyHandles.size()))
		{
			return 0;
		}

		if (m_bodyHandles[handle].getNextFree() == B3_POOL_HANDLE_TERMINAL_USED)
		{
			return &m_bodyHandles[handle];
		}
		return 0;
	}
	const T* getHandle(int handle) const
	{
		b3Assert(handle >= 0);
		b3Assert(handle < m_bodyHandles.size());
		if ((handle < 0) || (handle >= m_bodyHandles.size()))
		{
			return 0;
		}

		if (m_bodyHandles[handle].getNextFree() == B3_POOL_HANDLE_TERMINAL_USED)
		{
			return &m_bodyHandles[handle];
		}
		return 0;
	}

	void increaseHandleCapacity(int extraCapacity)
	{
		int curCapacity = m_bodyHandles.size();
		//b3Assert(curCapacity == m_numUsedHandles);
		int newCapacity = curCapacity + extraCapacity;
		m_bodyHandles.resize(newCapacity);

		{
			for (int i = curCapacity; i < newCapacity; i++)
				m_bodyHandles[i].setNextFree(i + 1);

			m_bodyHandles[newCapacity - 1].setNextFree(-1);
		}
		m_firstFreeHandle = curCapacity;
	}
	void initHandles()
	{
		m_numUsedHandles = 0;
		m_firstFreeHandle = -1;

		increaseHandleCapacity(1);
	}

	void exitHandles()
	{
		m_bodyHandles.resize(0);
		m_firstFreeHandle = -1;
		m_numUsedHandles = 0;
	}

	int allocHandle()
	{
		b3Assert(m_firstFreeHandle >= 0);

		int handle = m_firstFreeHandle;
		m_firstFreeHandle = getHandleInternal(handle)->getNextFree();
		m_numUsedHandles++;

		if (m_firstFreeHandle < 0)
		{
			//int curCapacity = m_bodyHandles.size();
			int additionalCapacity = m_bodyHandles.size();
			increaseHandleCapacity(additionalCapacity);

			getHandleInternal(handle)->setNextFree(m_firstFreeHandle);
		}
		getHandleInternal(handle)->setNextFree(B3_POOL_HANDLE_TERMINAL_USED);
		getHandleInternal(handle)->clear();
		return handle;
	}

	void freeHandle(int handle)
	{
		b3Assert(handle >= 0);

		if (m_bodyHandles[handle].getNextFree() == B3_POOL_HANDLE_TERMINAL_USED)
		{
			getHandleInternal(handle)->clear();
			getHandleInternal(handle)->setNextFree(m_firstFreeHandle);
			m_firstFreeHandle = handle;
			m_numUsedHandles--;
		}
	}
};
///end handle management

#endif  //B3_RESIZABLE_POOL_H
