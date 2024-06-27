/**************************************************************************/
/*  memory.cpp                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "memory.h"
#include "mutex.h"

#include "core/error/error_macros.h"
#include "core/templates/safe_refcount.h"

#include <stdio.h>
#include <stdlib.h>
#include "thirdparty/embree/common/sys/mutex.h"
// #include "thirdparty/embree/common/sys/platform.h"
// #include "thirdparty/embree/common/simd/simd.h"
// #include "thirdparty/embree/common/math/vec2.h"
// #include "thirdparty/embree/common/math/vec3.h"
// #include "thirdparty/embree/common/math/vec4.h"
// #include "thirdparty/embree/common/math/vec2fa.h"
// #include "thirdparty/embree/common/math/vec3fa.h"

using MutexSys = embree::MutexSys;

void *operator new(size_t p_size, const char *p_description) {
	return Memory::alloc_static(p_size, false);
}

void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size)) {
	return p_allocfunc(p_size);
}

#ifdef _MSC_VER
void operator delete(void *p_mem, const char *p_description) {
	CRASH_NOW_MSG("Call to placement delete should not happen.");
}

void operator delete(void *p_mem, void *(*p_allocfunc)(size_t p_size)) {
	CRASH_NOW_MSG("Call to placement delete should not happen.");
}

void operator delete(void *p_mem, void *p_pointer, size_t check, const char *p_description) {
	CRASH_NOW_MSG("Call to placement delete should not happen.");
}
#endif

#ifdef DEBUG_ENABLED
SafeNumeric<uint64_t> Memory::mem_usage;
SafeNumeric<uint64_t> Memory::max_usage;
#endif

SafeNumeric<uint64_t> Memory::alloc_count;
#define SAMLL_MEMORY_MANAGER 1


template <int SIZE_COUNT>
struct SmallMemoryBuffer
{
	uint64_t size_or_next = 0;
	uint8_t data[SIZE_COUNT];
};
template <int SIZE_COUNT>
struct SmallMemory
{
	void * alloc_mem(int p_count) {
		{			
			void * ret = nullptr;
			mutex.lock();
			if(buffers != nullptr)
			{
				SmallMemoryBuffer<SIZE_COUNT> *b = buffers;
				buffers = (SmallMemoryBuffer<SIZE_COUNT> *)b->size_or_next;
				free_count.decrement();
				ret = b->data;
				b->size_or_next = p_count;
			}
			mutex.unlock();
			if(ret != nullptr)
			{
				return ret;
			}
		}
		SmallMemoryBuffer<SIZE_COUNT> *b = (SmallMemoryBuffer<SIZE_COUNT> *)malloc(sizeof(SmallMemoryBuffer<SIZE_COUNT>));
		if(b == nullptr)
		{
			return nullptr;
		}
		b->size_or_next = p_count;
		return b->data;
		
	}
	void free_mem(void *p_ptr) {
		if(p_ptr == nullptr)
		{
			return;
		}

		uint8_t* mem = (uint8_t*)p_ptr;
		mem -= sizeof(uint64_t);
		if(free_count.get() > 500)
		{
			free(mem);
			return;
		}
		SmallMemoryBuffer<SIZE_COUNT>* b = (SmallMemoryBuffer<SIZE_COUNT> *) mem;
		mutex.lock();
		b->size_or_next = (uint64_t)buffers;
		buffers = b;
		mutex.unlock();
		free_count.increment();
	}
	 
	MutexSys mutex;
	SmallMemoryBuffer<SIZE_COUNT> * buffers = nullptr;
	SafeNumeric<uint64_t> free_count;
};

struct SmallMemoryManager
{
	#define SAMLL_MEMORY_MEMBER(count) SmallMemory<count> small_memory_## count
	#define SAMLL_MEMORY_MEMBER_BREAH(value,count) if(value <= count) {return small_memory_## count.alloc_mem(count);}
	#define SAMLL_MEMORY_MEMBER_BREAH_FREE(ptr,value,count) if(value <= count) {small_memory_## count.free_mem(ptr);}
	#define SAMLL_MEMORY_MEMBER_BREAH_INDEX(value,count,index) if(value <= count) {return index;}

	void* alloc_mem(int p_count) {
		if(p_count <= 0)
			return nullptr;
		SAMLL_MEMORY_MEMBER_BREAH(p_count, 20)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 40)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 80)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 120)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 160)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 180)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 256)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 320)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 480)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 512)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 800)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 1024)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 1500)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 1800)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 2048)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 3000)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 5000)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 7000)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 8000)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 12000)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 15000)
		else SAMLL_MEMORY_MEMBER_BREAH(p_count, 20000)
		else
		{
			uint8_t *mem = (uint8_t *)malloc(p_count + sizeof(uint64_t));
			if(mem == nullptr)
			{
				return nullptr;
			}
			*(uint64_t *)mem = p_count;
			return mem + sizeof(uint64_t);
			
		}

	}
	void free_mem(void *p_ptr) {
		if(p_ptr == nullptr)
		{
			return;
		}
		uint64_t p_count = get_buffer_count(p_ptr);
		SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count,20)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count,40)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count,80)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 120)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 160)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 180)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 256)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 320)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 480)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 512)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 800)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 1024)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 1500)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 1800)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 2048)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 3000)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 5000)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 7000)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 8000)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 12000)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 15000)
		else SAMLL_MEMORY_MEMBER_BREAH_FREE(p_ptr,p_count, 20000)
		else
		{
			uint8_t* mem = (uint8_t*)p_ptr - sizeof(uint64_t);
			free(mem);
		}



	}
	_FORCE_INLINE_ uint64_t get_buffer_count(void* p_ptr)
	{
		uint8_t *mem = (uint8_t *)p_ptr - sizeof(uint64_t);
		return *(uint64_t *)(mem);
	}
	_FORCE_INLINE_ int get_buffer_type_index(void* p_ptr)
	{
		uint8_t *mem = (uint8_t *)p_ptr - sizeof(uint64_t);
		uint64_t count = *(uint64_t *)(mem);
		return get_buffer_type_index(count);
	}

	int get_buffer_type_index(int p_count)
	{
		SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 20,0)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count,40,1)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count,80,2)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 120,3)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 160,4)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 180,5)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 256,6)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 320,7)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 480,8)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 512,9)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 800,10)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 1024,11)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 1500,12)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 1800,13)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 2048,14)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 3000,15)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 5000,16)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 7000,17)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 8000,18)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 12000,19)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 15000,20)
		else SAMLL_MEMORY_MEMBER_BREAH_INDEX(p_count, 20000,21)
		
		return -1;
	}




	SAMLL_MEMORY_MEMBER(20);
	SAMLL_MEMORY_MEMBER(40);
	SAMLL_MEMORY_MEMBER(80);
	SAMLL_MEMORY_MEMBER(120);
	SAMLL_MEMORY_MEMBER(160);
	SAMLL_MEMORY_MEMBER(180);
	SAMLL_MEMORY_MEMBER(256);
	SAMLL_MEMORY_MEMBER(320);
	SAMLL_MEMORY_MEMBER(480);
	SAMLL_MEMORY_MEMBER(512);
	SAMLL_MEMORY_MEMBER(800);
	SAMLL_MEMORY_MEMBER(1024);
	SAMLL_MEMORY_MEMBER(1500);
	SAMLL_MEMORY_MEMBER(1800);
	SAMLL_MEMORY_MEMBER(2048);
	SAMLL_MEMORY_MEMBER(3000);
	SAMLL_MEMORY_MEMBER(5000);
	SAMLL_MEMORY_MEMBER(7000);
	SAMLL_MEMORY_MEMBER(8000);
	SAMLL_MEMORY_MEMBER(12000);
	SAMLL_MEMORY_MEMBER(15000);
	SAMLL_MEMORY_MEMBER(20000);

	#undef SAMLL_MEMORY_MEMBER
	#undef SAMLL_MEMORY_MEMBER_BREAH
	#undef SAMLL_MEMORY_MEMBER_BREAH_FREE
	#undef SAMLL_MEMORY_MEMBER_BREAH_INDEX

};


static _FORCE_INLINE_ SmallMemoryManager& get_small_memory_manager()
{
	static SmallMemoryManager* s_manager = new(malloc(sizeof(SmallMemoryManager))) SmallMemoryManager();
	return *s_manager;
}


// void Memory::memset(void *p_ptr, int p_value, size_t p_size)
// {
	
// 	embree::vfloat8 t16(embree::zero);
// 	if(p_ptr == nullptr || p_size <= 0)
// 	{
// 		return;
// 	}
// 	uint8_t* dest = (uint8_t*)p_ptr;
// #if defined (__AVX512F__)
// 	while (p_size > 64)
// 	{
// 		new(dest) embree::vfloat16(embree::zero);
// 		p_size -= 64;
// 		dest += 64;
// 	}
// #endif

// #if defined(__AVX__)
// 	while(p_size > 32)
// 	{
// 		embree::vfloat8::storeu((void*)dest, (embree::vfloat8&)t16);
// 		p_size -= 32;
// 		dest += 32;
// 	}
// #else
// 	while(p_size > 16)
// 	{
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;
// 	}
// #endif

// 	switch (p_size)
// 	{
// 	case 31:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24


// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 28

// 		dest[0] = 0; // 29
// 		dest[1] = 0; // 30
// 		dest[2] = 0; // 31
// 		break;
// 	case 30:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 28

// 		dest[0] = 0; // 29
// 		dest[1] = 0; // 30
// 		break;
// 	case 29:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 28

// 		dest[0] = 0; // 29
// 		break;
// 	case 28:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 28

// 		break;
// 	case 27:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24

// 		dest[0] = 0; // 25
// 		dest[1] = 0; // 26
// 		dest[2] = 0; // 27

// 		break;
// 	case 26:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24

// 		dest[0] = 0; // 25
// 		dest[1] = 0; // 26
// 		break;


// 	case 25:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24

// 		dest[0] = 0; // 25

// 		break;

// 	case 24:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8; // 24

// 		break;

// 	case 23:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 20

// 		dest[0] = 0; // 21
// 		dest[1] = 0; // 22
// 		dest[2] = 0; // 23
// 		break;

// 	case 22:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 20

// 		dest[0] = 0; // 21
// 		dest[1] = 0; // 22
// 		break;

// 	case 21:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 20

// 		dest[0] = 0; // 21
// 		break;


// 	case 20:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 20

// 		break;

// 	case 19:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		dest[0] = 0; // 17
// 		dest[1] = 0; // 18
// 		dest[2] = 0; // 19
// 		break;

// 	case 18:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		dest[0] = 0; // 17
// 		dest[1] = 0; // 18
// 		break;

// 	case 17:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		dest[0] = 0; // 17
// 		break;

// 	case 16:
// 		embree::vfloat4::storeu((void*)dest, (embree::vfloat4&)t16);
// 		p_size -= 16;
// 		dest += 16;// 16

// 		break;

// 	case 15:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8
		
// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 12

// 		dest[4] = 0; // 13
// 		dest[5] = 0; // 14
// 		dest[6] = 0; // 15
// 		break;

// 	case 14:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 12

// 		dest[4] = 0; // 13
// 		dest[5] = 0; // 14
// 		break;

// 	case 13:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 12

// 		dest[4] = 0; // 13
// 		break;

// 	case 12:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8

// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 12

// 		break;

// 	case 11:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8

// 		dest[0] = 0; // 9
// 		dest[1] = 0; // 10
// 		dest[2] = 0; // 11
// 		break;

// 	case 10:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8

// 		dest[0] = 0; // 9
// 		dest[1] = 0; // 10
// 		break;

// 	case 9:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8

// 		dest[0] = 0; // 9
// 		break;

// 	case 8:
// 		*(uint64_t*)dest = 0ULL;
// 		p_size -= 8;
// 		dest += 8;// 8

// 		break;

// 	case 7:
// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 4

// 		dest[0] = 0; // 5
// 		dest[1] = 0; // 6
// 		dest[2] = 0; // 7
// 		break;

// 	case 6:
// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 4

// 		dest[0] = 0; // 5
// 		dest[1] = 0; // 6
// 		break;

// 	case 5:
// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 4

// 		dest[0] = 0; // 5
// 		break;

// 	case 4:
// 		*(uint32_t*)dest = 0;
// 		p_size -= 4;
// 		dest += 4;// 4

// 		break;

// 	case 3:
// 		*(uint16_t*)dest = 0;
// 		p_size -= 2;
// 		dest += 2;// 2

// 		dest[0] = 0; // 3
// 		dest[1] = 0; // 4
// 		break;

// 	case 2:
// 		*(uint16_t*)dest = 0;
// 		p_size -= 2;
// 		dest += 2;// 2
// 		break;

// 	case 1:
// 		*(uint8_t*)dest = 0;
// 		p_size -= 1;
// 		dest += 1;// 1
// 		break;

// 	default:
// 		break;
// 	}

// }
// void memcpy(void *p_dest, const void *p_src, size_t p_size)
// {
// 	if(p_dest == nullptr || p_src == nullptr || p_size <= 0)
// 	{
// 		return;
// 	}
// 	uint8_t* dest = (uint8_t*)p_dest;
// 	uint8_t* src = (uint8_t*)p_src;
// #if defined (__AVX512F__)
// 	while (p_size > 64)
// 	{
// 		embree::vfloatx::storeu(dest, *(embree::vfloat16*)src);
// 		p_size -= 64;
// 		dest += 64;
// 		src += 64;
// 	}
// #endif

// #if defined(__AVX__)
// 	while (p_size > 32)
// 	{
// 		embree::vfloat8::storeu(dest, *(embree::vfloat8*)src);
// 		p_size -= 32;
// 		dest += 32;
// 		src += 32;
// 	}
// #else
// 	while (p_size > 16)
// 	{
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;
// 		src += 16;
// 	}
// #endif
// 	switch (p_size)
// 	{
// 	case 31:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24


// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 28
// 		src += 4;// 28

// 		dest[0] = src[0]; // 29
// 		dest[1] = src[1]; // 30
// 		dest[2] = src[2]; // 31
// 		break;

// 	case 30:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 28
// 		src += 4;// 28

// 		dest[0] = src[0]; // 29
// 		dest[1] = src[1]; // 30
// 		break;

// 	case 29:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 28
// 		src += 4;// 28

// 		dest[0] = src[0]; // 29
// 		break;

// 	case 28:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 28
// 		src += 4;// 28
// 		break;
// 	case 27:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24

// 		dest[0] = src[0]; // 25
// 		dest[1] = src[1]; // 26
// 		dest[2] = src[2]; // 27

// 		break;

// 	case 26:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24

// 		dest[0] = src[0]; // 25
// 		dest[1] = src[1]; // 26
// 		break;
// 	case 25:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24

// 		dest[0] = src[0]; // 25
// 		break;

// 	case 24:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 24
// 		src += 8; // 24
// 		break;

// 	case 23:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 20
// 		src += 4;// 20

// 		dest[0] = src[0]; // 21
// 		dest[1] = src[1]; // 22
// 		dest[2] = src[2]; // 23
// 		break;

// 	case 22:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 20
// 		src += 4;// 20

// 		dest[0] = src[0]; // 21
// 		dest[1] = src[1]; // 22
// 		break;

// 	case 21:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 20
// 		src += 4;// 20

// 		dest[0] = src[0]; // 21
// 		break;
// 	case 20:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 20
// 		src += 4;// 20
// 		break;
		
// 	case 19:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16
		
// 		dest[0] = src[0]; // 17
// 		dest[1] = src[1]; // 18
// 		dest[2] = src[2]; // 19
// 		break;
		
// 	case 18:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16
		
// 		dest[0] = src[0]; // 17
// 		dest[1] = src[1]; // 18
// 		break;
		
// 	case 17:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16
		
// 		dest[0] = src[0]; // 17
// 		break;

// 	case 16:
// 		embree::vfloat4::storeu((void*)dest, *(embree::vfloat4*)src);
// 		p_size -= 16;
// 		dest += 16;// 16
// 		src += 16;// 16
// 		break;

// 	case 15:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 12
// 		src += 4;// 12

// 		dest[0] = src[0]; // 13
// 		dest[1] = src[1]; // 14
// 		dest[2] = src[2]; // 15
// 		break;

// 	case 14:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 12
// 		src += 4;// 12

// 		dest[0] = src[0]; // 13
// 		dest[1] = src[1]; // 14
// 		break;

// 	case 13:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 12
// 		src += 4;// 12

// 		dest[0] = src[0]; // 13
// 		break;

// 	case 12:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8

// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 12
// 		src += 4;// 12
// 		break;

// 	case 11:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8

// 		dest[0] = src[0]; // 9
// 		dest[1] = src[1]; // 10
// 		dest[2] = src[2]; // 11
// 		break;

// 	case 10:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8

// 		dest[0] = src[0]; // 9
// 		dest[1] = src[1]; // 10
// 		break;

// 	case 9:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8

// 		dest[0] = src[0]; // 9
// 		break;


// 	case 8:
// 		*(uint64_t*)dest = *(uint64_t*)src;
// 		p_size -= 8;
// 		dest += 8; // 8
// 		src += 8; // 8
// 		break;


// 	case 7:
// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 4
// 		src += 4;// 4

// 		dest[0] = src[0]; // 5
// 		dest[1] = src[1]; // 6
// 		dest[2] = src[2]; // 7
// 		break;

// 	case 6:
// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 4
// 		src += 4;// 4

// 		dest[0] = src[0]; // 5
// 		dest[1] = src[1]; // 6
// 		break;

// 	case 5:
// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 4
// 		src += 4;// 4

// 		dest[0] = src[0]; // 5
// 		break;


// 	case 4:
// 		*(uint32_t*)dest = *(uint32_t*)src;
// 		p_size -= 4;
// 		p_size -= 4;
// 		dest += 4;// 4
// 		src += 4;// 4
// 		break;

// 	case 3:
// 		*(uint16_t*)dest = *(uint16_t*)src;
// 		p_size -= 2;
// 		p_size -= 2;
// 		dest += 2;// 2
// 		src += 2;// 2

// 		dest[0] = src[0]; // 3
// 		dest[1] = src[1]; // 4
// 		break;

// 	case 2:
// 		*(uint16_t*)dest = *(uint16_t*)src;
// 		p_size -= 2;
// 		p_size -= 2;
// 		dest += 2;// 2
// 		src += 2;// 2
// 		break;

// 	case 1:
// 		*(uint8_t*)dest = *(uint8_t*)src;
// 		p_size -= 1;
// 		p_size -= 1;
// 		dest += 1;// 1
// 		src += 1;// 1
// 		break;

// 	default:
// 		break;
// 	}
	
// }


void *Memory::alloc_static(size_t p_bytes, bool p_pad_align) {
#ifdef DEBUG_ENABLED
	bool prepad = true;
#else
	bool prepad = p_pad_align;
#endif

	//void *mem = malloc(p_bytes + (prepad ? DATA_OFFSET : 0));
	void *mem = get_small_memory_manager().alloc_mem(p_bytes);

	ERR_FAIL_NULL_V(mem, nullptr);

	alloc_count.increment();
	return mem;

// 	if (prepad) {
// 		uint8_t *s8 = (uint8_t *)mem;

// 		uint64_t *s = (uint64_t *)(s8 + SIZE_OFFSET);
// 		*s = p_bytes;

// #ifdef DEBUG_ENABLED
// 		uint64_t new_mem_usage = mem_usage.add(p_bytes);
// 		max_usage.exchange_if_greater(new_mem_usage);
// #endif
// 		return s8 + DATA_OFFSET;
// 	} else {
// 		return mem;
// 	}
}

void *Memory::realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align) {
	if (p_memory == nullptr) {
		return alloc_static(p_bytes, p_pad_align);
	}

	uint8_t *mem = (uint8_t *)p_memory;
	if (p_bytes == 0) {
		free(mem);
		return nullptr;
	} 
	else {
		auto old_count = get_small_memory_manager().get_buffer_count(mem);

		int index = get_small_memory_manager().get_buffer_type_index(old_count);
		if(index >= 0 && index == get_small_memory_manager().get_buffer_type_index(p_bytes) )
		{
			return p_memory;
		}

		void* new_mem = get_small_memory_manager().alloc_mem(p_bytes);
		memcpy(new_mem, p_memory, MIN(old_count, p_bytes));
		get_small_memory_manager().free_mem(p_memory);
		return new_mem;
	}
}

void Memory::free_static(void *p_ptr, bool p_pad_align) {
	ERR_FAIL_NULL(p_ptr);
	
	get_small_memory_manager().free_mem(p_ptr);
 	alloc_count.decrement();

// 	uint8_t *mem = (uint8_t *)p_ptr;

// #ifdef DEBUG_ENABLED
// 	bool prepad = true;
// #else
// 	bool prepad = p_pad_align;
// #endif

// 	alloc_count.decrement();

// 	if (prepad) {
// 		mem -= DATA_OFFSET;

// #ifdef DEBUG_ENABLED
// 		uint64_t *s = (uint64_t *)(mem + SIZE_OFFSET);
// 		mem_usage.sub(*s);
// #endif

// 		free(mem);
// 	} else {
// 		free(mem);
// 	}
}

uint64_t Memory::get_mem_available() {
	return -1; // 0xFFFF...
}

uint64_t Memory::get_mem_usage() {
#ifdef DEBUG_ENABLED
	return mem_usage.get();
#else
	return 0;
#endif
}

uint64_t Memory::get_mem_max_usage() {
#ifdef DEBUG_ENABLED
	return max_usage.get();
#else
	return 0;
#endif
}

_GlobalNil::_GlobalNil() {
	left = this;
	right = this;
	parent = this;
}

_GlobalNil _GlobalNilClass::_nil;
