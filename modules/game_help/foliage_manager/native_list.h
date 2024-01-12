#ifndef NATIVE_LIST_H
#define NATIVE_LIST_H
#include <atomic>
#include "core/object/class_db.h"
#include "core/templates/safe_refcount.h"
namespace Foliage
{
    
    template<typename T> 
    struct NativeList
	{
		T* data = nullptr;
		SafeNumeric<int64_t> listCount;

		int64_t memorySize = 0;
		bool isSafeAlocal = false;
		int Length() const 
        {
             return MIN(listCount.get(),memorySize);
        }
		NativeList(int memSize = 0)
		{
			data = nullptr;
			listCount = 0;
			memorySize = 0;
			isSafeAlocal = false;
			if (memSize > 0)
			{
				AutoResize(memSize, memSize);
			}
		}
        ~NativeList()
        {
            Dispose();
        }
		void SetMemory(T* memory, int offset, int list_count, int memory_size)
		{
			data = memory + offset;
			listCount.set(list_count);
			memorySize = memory_size;
			isSafeAlocal = false;
		}
		void Init(int memSize = 0)
		{
			data = nullptr;
			listCount.set(0);
			memorySize = 0;
			isSafeAlocal = false;
			if (memSize > 0)
			{
				AutoResize(memSize, memSize);
			}
		}
		/// <summary>
		/// 自动重置大小
		/// </summary>
        void AutoResize(int length, int next_length)
		{
			if (memorySize < length)
			{
				if (next_length < length)
				{
					//Debug.LogError("参数设置错误，next_length 要大于length ！");
					next_length = length;
				}

				long newsize = sizeof(T) * (long)next_length;
				auto new_data = UnsafeUtility.Malloc(newsize, 8, Allocator.Persistent);
				if (new_data == null)
				{
					throw new Exception($"内存非陪错误！无法分配【{newsize}】 内存");
				}
				UnsafeUtility.MemClear(new_data, newsize);
				if (data != null)
				{
					UnsafeUtility.MemCpy(new_data, GetUnsafePtr(), UnsafeUtility.SizeOf<T>() * listCount);
					UnsafeUtility.Free(data, Allocator.Persistent);
				}
				memorySize = next_length;
				data = (T*)new_data;
				isSafeAlocal = true;
			}
		}
        T* GetUnsafePtr(int index = 0)
		{
			if (index > listCount)
			{
				//Debug.LogError($"参数获取错误，index[{index}] 要小于listCount[{listCount}]！");
				return nullptr;
			}
			return &data[index];
		}
        void SetValue(int index, in T value)
		{
			if (index > listCount)
			{
				//Debug.LogError($"参数设置错误，index[{index}] 要小于listCount[{listCount}]！");
				return;
			}
			data[index] = value;
		}
		void RemoveRange(int start, int count)
		{
			if (count <= 0 || start < 0)
			{
				//Debug.LogError($"小贼别跑，这里有毒：{Length} start:{start} count:{count}！");
				return;
			}
			if (start + count > Length())
			{
				//Debug.LogError($"移除的数据太多了，当前长度：{Length} start:{start} count:{count}！");
				listCount.set(start);
				return;
			}
			else if (start + count == Length())
			{
				listCount.set(start);
				return;
			}
			//int bs = start + count;
			//int index = 0;
			//for(int i = bs; i < listCount; ++i,++index)
			//{
			//	UnsafeUtility.MemCpy(&data[start + index], &data[bs + index], UnsafeUtility.SizeOf<T>());
			//}
			int last_index = start + count;
			int copy_count = listCount.get() - last_index;
			T* temp = (T*)allocal(sizeof(T) * copy_count);
			memcpy(temp, &data[last_index], sizeof(T) * copy_count);
			memcpy(&data[start], temp, sizeof(T) * copy_count);
		}
		/// <summary>
		/// 获取指针地址
		/// </summary>
		/// <returns></returns>
        void* GetUnsafeReadOnlyPtr(int index = 0)
		{
			return GetUnsafePtr(index);
		}
        void ZeroMemory()
		{
			auto ptr = GetUnsafePtr();
			if (ptr != nullptr)
			{
				memset(ptr,0, sizeof(T) * listCount);
			}
		}
        void MemSet(uint8_t value)
		{
			auto ptr = GetUnsafePtr();
			if (ptr != nullptr)
			{
				memset(ptr,0, sizeof(T) * listCount);
			}
		}
        void AddRef(T& _data)
		{
			AutoResize(listCount + 1, listCount + 1 + 200);

			memcpy(&data[listCount], & _data, sizeof(T));
			listCount.add(1);
		}
		/// <summary>
		/// 多线程并发写入，不支持动态分配内存，需要预先分配好内存
		/// </summary>
		/// <param name="_data"></param>
        void Thread_Add(ref T _data)
		{
			auto idx = listCount.add(1) - 1;
			if (idx < memorySize)
			{
				memcpy(&data[idx], & _data, sizeof(T));
			}
		}
		/// <summary>
		/// 多线程并发写入，不支持动态分配内存，需要预先分配好内存
		/// </summary>
		/// <param name="_data"></param>
        void Thread_AddRange(T* _data, int count)
		{
			auto idx = listCount.Add( count) - count;
			if (idx + count <= memorySize)
			{
				memcpy(&data[idx], _data, sizeof(T) * count);
			}
		}


        void Add(T _data)
		{
			AutoResize(listCount + 1, listCount + 1 + 200);

			memcpy(&data[listCount], &_data, sizeof(T));
			listCount += 1;
		}
        void AddRange(T* _data, int count)
		{
			AutoResize(listCount + count, listCount + count + 200);

			memcpy(&data[listCount], _data, sizeof(T) * count);
			listCount += count;

		}
		/// <summary>
		/// Sets the length of this list, increasing the capacity if necessary.
		/// </summary>
		/// <remarks>Does not clear newly allocated bytes.</remarks>
		/// <param name="length">The new length of this list.</param>
        void ResizeUninitialized(int length, int addCount = 0)
		{
			AutoResize(length, length + addCount);
			listCount = length;
		}
        void CompactMemory()
		{
			if (data != nullptr)
			{
				if (isSafeAlocal)
				{
					UnsafeUtility.Free(data, Allocator.Persistent);
				}
				data = nullptr;
				listCount.set(0);
				memorySize = 0;
			}

		}
        void Clear()
		{
			listCount = 0;
		}
        void Dispose()
		{
			if (data != nullptr)
			{
				if (isSafeAlocal)
				{
					UnsafeUtility.Free(data, Allocator.Persistent);
				}
				data = null;
				listCount = 0;
				memorySize = 0;
			}
		}
	};
	
}

#endif