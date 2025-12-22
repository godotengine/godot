/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_CPU_H
#define SPA_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>

/** \defgroup spa_cpu CPU
 * Querying CPU properties
 */

/**
 * \addtogroup spa_cpu
 * \{
 */

/**
 * The CPU features interface
 */
#define SPA_TYPE_INTERFACE_CPU	SPA_TYPE_INFO_INTERFACE_BASE "CPU"

#define SPA_VERSION_CPU		0
struct spa_cpu { struct spa_interface iface; };

/* x86 specific */
#define SPA_CPU_FLAG_MMX		(1<<0)	/**< standard MMX */
#define SPA_CPU_FLAG_MMXEXT		(1<<1)	/**< SSE integer or AMD MMX ext */
#define SPA_CPU_FLAG_3DNOW		(1<<2)	/**< AMD 3DNOW */
#define SPA_CPU_FLAG_SSE		(1<<3)	/**< SSE */
#define SPA_CPU_FLAG_SSE2		(1<<4)	/**< SSE2 */
#define SPA_CPU_FLAG_3DNOWEXT		(1<<5)	/**< AMD 3DNowExt */
#define SPA_CPU_FLAG_SSE3		(1<<6)	/**< Prescott SSE3 */
#define SPA_CPU_FLAG_SSSE3		(1<<7)	/**< Conroe SSSE3 */
#define SPA_CPU_FLAG_SSE41		(1<<8)	/**< Penryn SSE4.1 */
#define SPA_CPU_FLAG_SSE42		(1<<9)	/**< Nehalem SSE4.2 */
#define SPA_CPU_FLAG_AESNI		(1<<10)	/**< Advanced Encryption Standard */
#define SPA_CPU_FLAG_AVX		(1<<11)	/**< AVX */
#define SPA_CPU_FLAG_XOP		(1<<12)	/**< Bulldozer XOP */
#define SPA_CPU_FLAG_FMA4		(1<<13)	/**< Bulldozer FMA4 */
#define SPA_CPU_FLAG_CMOV		(1<<14)	/**< supports cmov */
#define SPA_CPU_FLAG_AVX2		(1<<15)	/**< AVX2 */
#define SPA_CPU_FLAG_FMA3		(1<<16)	/**< Haswell FMA3 */
#define SPA_CPU_FLAG_BMI1		(1<<17)	/**< Bit Manipulation Instruction Set 1 */
#define SPA_CPU_FLAG_BMI2		(1<<18)	/**< Bit Manipulation Instruction Set 2 */
#define SPA_CPU_FLAG_AVX512		(1<<19)	/**< AVX-512 */
#define SPA_CPU_FLAG_SLOW_UNALIGNED	(1<<20)	/**< unaligned loads/stores are slow */

/* PPC specific */
#define SPA_CPU_FLAG_ALTIVEC		(1<<0)	/**< standard */
#define SPA_CPU_FLAG_VSX		(1<<1)	/**< ISA 2.06 */
#define SPA_CPU_FLAG_POWER8		(1<<2)	/**< ISA 2.07 */

/* ARM specific */
#define SPA_CPU_FLAG_ARMV5TE		(1 << 0)
#define SPA_CPU_FLAG_ARMV6		(1 << 1)
#define SPA_CPU_FLAG_ARMV6T2		(1 << 2)
#define SPA_CPU_FLAG_VFP		(1 << 3)
#define SPA_CPU_FLAG_VFPV3		(1 << 4)
#define SPA_CPU_FLAG_NEON		(1 << 5)
#define SPA_CPU_FLAG_ARMV8		(1 << 6)

#define SPA_CPU_FORCE_AUTODETECT	((uint32_t)-1)

#define SPA_CPU_VM_NONE			(0)
#define SPA_CPU_VM_OTHER		(1 << 0)
#define SPA_CPU_VM_KVM			(1 << 1)
#define SPA_CPU_VM_QEMU			(1 << 2)
#define SPA_CPU_VM_BOCHS		(1 << 3)
#define SPA_CPU_VM_XEN			(1 << 4)
#define SPA_CPU_VM_UML			(1 << 5)
#define SPA_CPU_VM_VMWARE		(1 << 6)
#define SPA_CPU_VM_ORACLE		(1 << 7)
#define SPA_CPU_VM_MICROSOFT		(1 << 8)
#define SPA_CPU_VM_ZVM			(1 << 9)
#define SPA_CPU_VM_PARALLELS		(1 << 10)
#define SPA_CPU_VM_BHYVE		(1 << 11)
#define SPA_CPU_VM_QNX			(1 << 12)
#define SPA_CPU_VM_ACRN			(1 << 13)
#define SPA_CPU_VM_POWERVM		(1 << 14)

/**
 * methods
 */
struct spa_cpu_methods {
	/** the version of the methods. This can be used to expand this
	  structure in the future */
#define SPA_VERSION_CPU_METHODS	2
	uint32_t version;

	/** get CPU flags */
	uint32_t (*get_flags) (void *object);

	/** force CPU flags, use SPA_CPU_FORCE_AUTODETECT to autodetect CPU flags */
	int (*force_flags) (void *object, uint32_t flags);

	/** get number of CPU cores */
	uint32_t (*get_count) (void *object);

	/** get maximum required alignment of data */
	uint32_t (*get_max_align) (void *object);

	/* check if running in a VM. Since:1 */
	uint32_t (*get_vm_type) (void *object);

	/* denormals will be handled as zero, either with FTZ or DAZ.
	 * Since:2 */
	int (*zero_denormals) (void *object, bool enable);
};

#define spa_cpu_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	struct spa_cpu *_c = o;						\
	spa_interface_call_res(&_c->iface,				\
			struct spa_cpu_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})
#define spa_cpu_get_flags(c)		spa_cpu_method(c, get_flags, 0)
#define spa_cpu_force_flags(c,f)	spa_cpu_method(c, force_flags, 0, f)
#define spa_cpu_get_count(c)		spa_cpu_method(c, get_count, 0)
#define spa_cpu_get_max_align(c)	spa_cpu_method(c, get_max_align, 0)
#define spa_cpu_get_vm_type(c)		spa_cpu_method(c, get_vm_type, 1)
#define spa_cpu_zero_denormals(c,e)	spa_cpu_method(c, zero_denormals, 2, e)

/** keys can be given when initializing the cpu handle */
#define SPA_KEY_CPU_FORCE		"cpu.force"		/**< force cpu flags */
#define SPA_KEY_CPU_VM_TYPE		"cpu.vm.type"		/**< force a VM type */
#define SPA_KEY_CPU_ZERO_DENORMALS	"cpu.zero.denormals"	/**< zero denormals */

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_CPU_H */
