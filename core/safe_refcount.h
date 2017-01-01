/*************************************************************************/
/*  safe_refcount.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef SAFE_REFCOUNT_H
#define SAFE_REFCOUNT_H

#include "os/mutex.h"
/* x86/x86_64 GCC */

#include "platform_config.h"


#ifdef NO_THREADS

struct SafeRefCount {

	int count;

public:

	// destroy() is called when weak_count_ drops to zero.

	bool ref() {  //true on success

		if (count==0)
			return false;
		count++;

		return true;
	}

	int refval() {  //true on success

		if (count==0)
			return 0;
		count++;
		return count;
	}

	bool unref() { // true if must be disposed of

		if (count>0)
			count--;

		return count==0;
	}

	long get() const { // nothrow

		return static_cast<int const volatile &>( count );
	}

	void init(int p_value=1) {

		count=p_value;
	};

};








#else

#if defined( PLATFORM_REFCOUNT )

#include "platform_refcount.h"


#elif defined( __GNUC__ ) && ( defined( __i386__ ) || defined( __x86_64__ ) )

#define REFCOUNT_T volatile int
#define REFCOUNT_GET_T int const volatile&

static inline int atomic_conditional_increment( volatile int * pw ) {
	// int rv = *pw;
	// if( rv != 0 ) ++*pw;
	// return rv;

	int rv, tmp;

	__asm__
	(
		"movl %0, %%eax\n\t"
		"0:\n\t"
		"test %%eax, %%eax\n\t"
		"je 1f\n\t"
		"movl %%eax, %2\n\t"
		"incl %2\n\t"
		"lock\n\t"
		"cmpxchgl %2, %0\n\t"
		"jne 0b\n\t"
		"1:":
		"=m"( *pw ), "=&a"( rv ), "=&r"( tmp ): // outputs (%0, %1, %2)
		"m"( *pw ): // input (%3)
		"cc" // clobbers
	);

	return rv;
}

static inline int atomic_decrement( volatile int *pw) {

	// return --(*pw);

	unsigned char rv;

	__asm__
	(
		"lock\n\t"
		"decl %0\n\t"
		"setne %1":
		"=m" (*pw), "=qm" (rv):
		"m" (*pw):
		"memory"
	);
	return static_cast<int>(rv);
}

/* PowerPC32/64 GCC */

#elif ( defined( __GNUC__ ) ) && ( defined( __powerpc__ ) || defined( __ppc__ ) )

#define REFCOUNT_T int
#define REFCOUNT_GET_T int const volatile&

inline int atomic_conditional_increment( int * pw )
{
    // if( *pw != 0 ) ++*pw;
    // return *pw;

    int rv;

    __asm__
    (
        "0:\n\t"
        "lwarx %1, 0, %2\n\t"
        "cmpwi %1, 0\n\t"
        "beq 1f\n\t"
        "addi %1, %1, 1\n\t"
        "1:\n\t"
        "stwcx. %1, 0, %2\n\t"
        "bne- 0b":

        "=m"( *pw ), "=&b"( rv ):
        "r"( pw ), "m"( *pw ):
        "cc"
    );

    return rv;
}


inline int atomic_decrement( int * pw )
{
    // return --*pw;

    int rv;

    __asm__ __volatile__
    (
        "sync\n\t"
        "0:\n\t"
        "lwarx %1, 0, %2\n\t"
        "addi %1, %1, -1\n\t"
        "stwcx. %1, 0, %2\n\t"
        "bne- 0b\n\t"
        "isync":

        "=m"( *pw ), "=&b"( rv ):
        "r"( pw ), "m"( *pw ):
        "memory", "cc"
    );

    return rv;
}

/* CW ARM */

#elif defined( __GNUC__ ) && ( defined( __arm__ )  )

#define REFCOUNT_T int
#define REFCOUNT_GET_T int const volatile&

inline int atomic_conditional_increment(volatile int* v)
{
   int t;
   int tmp;

   __asm__ __volatile__(
			 "1:  ldrex   %0, [%2]        \n"
			 "    cmp     %0, #0      \n"
			 "    beq     2f          \n"
			 "    add     %0, %0, #1      \n"
			 "2: \n"
			 "    strex   %1, %0, [%2]    \n"
			 "    cmp     %1, #0          \n"
			 "    bne     1b              \n"

			 : "=&r" (t), "=&r" (tmp)
			 : "r" (v)
			 : "cc", "memory");

   return t;
}


inline int atomic_decrement(volatile int* v)
{
   int t;
   int tmp;

   __asm__ __volatile__(
			 "1:  ldrex   %0, [%2]        \n"
			 "    add     %0, %0, #-1      \n"
			 "    strex   %1, %0, [%2]    \n"
			 "    cmp     %1, #0          \n"
			 "    bne     1b              \n"

			 : "=&r" (t), "=&r" (tmp)
			 : "r" (v)
			 : "cc", "memory");

   return t;
}



/* CW PPC */

#elif ( defined( __MWERKS__ ) ) && defined( __POWERPC__ )

inline long atomic_conditional_increment( register long * pw )
{
    register int a;

	asm
	{
	loop:

	lwarx   a, 0, pw
	cmpwi   a, 0
	beq     store

	addi    a, a, 1

	store:

	stwcx.  a, 0, pw
	bne-    loop
    }

    return a;
}


inline long atomic_decrement( register long * pw )
{
    register int a;

    asm {

	sync

	loop:

	lwarx   a, 0, pw
	addi    a, a, -1
	stwcx.  a, 0, pw
	bne-    loop

	isync
    }

    return a;
}

/* Any Windows (MSVC) */

#elif defined( _MSC_VER )

// made functions to not pollute namespace..

#define REFCOUNT_T long
#define REFCOUNT_GET_T long const volatile&

long atomic_conditional_increment( register long * pw );
long atomic_decrement( register long * pw );

#if 0
#elif defined( __GNUC__ ) && defined( ARMV6_ENABLED)


#endif




#else

#error This platform cannot use safe refcount, compile with NO_THREADS or implement it.

#endif



struct SafeRefCount {

  REFCOUNT_T count;

public:

	// destroy() is called when weak_count_ drops to zero.

	bool ref() {  //true on success

		return atomic_conditional_increment( &count ) != 0;
	}

	int refval() {  //true on success

		return atomic_conditional_increment( &count );
	}

	bool unref() { // true if must be disposed of

		if( atomic_decrement ( &count ) == 0 ) {
			return true;
		}

		return false;
	}

	long get() const { // nothrow

		return static_cast<REFCOUNT_GET_T>( count );
	}

	void init(int p_value=1) {

		count=p_value;
	};

};



#endif // no thread safe

#endif
