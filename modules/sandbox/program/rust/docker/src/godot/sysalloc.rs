extern crate alloc;
use alloc::alloc::Layout;
use alloc::alloc::GlobalAlloc;
use std::arch::asm;
const NATIVE_SYSCALLS_BASE: i32 = 480;

struct SysAllocator;

/** Native-performance host-side implementations of common heap functions */

unsafe impl GlobalAlloc for SysAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret: *mut u8;
        asm!("ecall", in("a7") NATIVE_SYSCALLS_BASE + 0,
            in("a0") layout.size(), in("a1") layout.align(),
            lateout("a0") ret);
        return ret;
    }
    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ret: *mut u8;
        asm!("ecall", in("a7") NATIVE_SYSCALLS_BASE + 1,
            in("a0") layout.size(), in("a1") 1, lateout("a0") ret);
        return ret;
    }
    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, _layout: Layout, new_size: usize) -> *mut u8 {
        let ret: *mut u8;
        asm!("ecall", in("a7") NATIVE_SYSCALLS_BASE + 2,
            in("a0") ptr, in("a1") new_size, lateout("a0") ret);
        return ret;
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        asm!("ecall", in("a7") NATIVE_SYSCALLS_BASE + 3,
            in("a0") ptr, lateout("a0") _);
    }
}

#[global_allocator]
static A: SysAllocator = SysAllocator;

/** Native-performance host-side implementations of common memory functions */

#[no_mangle]
pub fn __wrap_memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8
{
	unsafe {
		asm!("ecall",
			in("a0") dest,
			in("a1") src,
			in("a2") n,
			in("a7") 485+0,
			options(nostack)
		);
	}
	return dest;
}

#[no_mangle]
pub fn __wrap_memset(s: *mut u8, c: i32, n: usize) -> *mut u8
{
	unsafe {
		asm!("ecall",
			in("a0") s,
			in("a1") c,
			in("a2") n,
			in("a7") 485+1,
			options(nostack)
		);
	}
	return s;
}

#[no_mangle]
pub fn __wrap_memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8
{
	unsafe {
		asm!("ecall",
			in("a0") dest,
			in("a1") src,
			in("a2") n,
			in("a7") 485+2,
			options(nostack)
		);
	}
	return dest;
}

#[no_mangle]
pub fn __wrap_memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32
{
	let result: i32;
	unsafe {
		asm!("ecall",
			in("a0") s1,
			in("a1") s2,
			in("a2") n,
			in("a7") 485+3,
			lateout("a0") result,
			options(nostack, readonly)
		);
	}
	return result;
}

#[no_mangle]
pub fn __wrap_strlen(s: *const u8) -> usize
{
	let result: usize;
	unsafe {
		asm!("ecall",
			in("a0") s,
			in("a7") 490,
			lateout("a0") result,
			options(nostack, readonly)
		);
	}
	return result;
}

#[no_mangle]
pub fn __wrap_strcmp(s1: *const u8, s2: *const u8) -> i32
{
	let result: i32;
	unsafe {
		asm!("ecall",
			in("a0") s1,
			in("a1") s2,
			in("a2") 4096, // MAX_STRLEN
			in("a7") 491,
			lateout("a0") result,
			options(nostack, readonly)
		);
	}
	return result;
}

#[no_mangle]
pub fn __wrap_strncmp(s1: *const u8, s2: *const u8, n: usize) -> i32
{
	let result: i32;
	unsafe {
		asm!("ecall",
			in("a0") s1,
			in("a1") s2,
			in("a2") n,
			in("a7") 491,
			lateout("a0") result,
			options(nostack, readonly)
		);
	}
	return result;
}

#[no_mangle]
pub fn fast_exit() -> ! {
	unsafe {
		asm!("ecall",
			in("a7") 93,
			options(noreturn)
		);
	}
}
