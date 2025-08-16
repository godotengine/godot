use std::arch::global_asm;
use std::arch::asm;

// A dynamic call is system call 510 with a function number in register t0
global_asm!(
	".pushsection .text",
	".global dyncall1",
	"dyncall1:",
	"li t0, 1",
	"li a7, 510",
	"ecall",
	"ret",
	".global dyncall2",
	"dyncall2:",
	"li t0, 2",
	"li a7, 510",
	"ecall",
	"ret",
	".global dyncall3",
	"dyncall3:",
	"li t0, 3",
	"li a7, 510",
	"ecall",
	"ret",
	".global dyncall4",
	"dyncall4:",
	"li t0, 4",
	"li a7, 510",
	"ecall",
	"ret",
	".global dyncall5",
	"dyncall5:",
	"li t0, 5",
	"li a7, 510",
	"ecall",
	"ret",
	".global fast_exit",
	"fast_exit:",
	"wfi",
	"j fast_exit",
	".popsection"
);
extern "C" {
	// This is the function signatures for the dynamic calls
	// You decide what the functions look like and what they do
	pub fn dyncall1(n: i32) -> i32;
	pub fn dyncall2(n: i32) -> i32;
	pub fn dyncall3();
	pub fn dyncall4(n: i32) -> i32;
	pub fn dyncall5(n: i32) -> i32;
}

#[no_mangle]
pub fn __wrap_memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8
{
	unsafe {
		asm!("ecall",
			in("a0") dest,
			in("a1") src,
			in("a2") n,
			in("a7") 495+0,
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
			in("a7") 495+1,
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
			in("a7") 495+2,
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
			in("a7") 495+3,
			lateout("a0") result,
			options(nostack, readonly)
		);
	}
	return result;
}
