extern crate alloc;
use alloc::alloc::Layout;
use alloc::alloc::GlobalAlloc;
use std::arch::asm;
const NATIVE_SYSCALLS_BASE: i32 = 490;

struct SysAllocator;

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
