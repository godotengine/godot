#![allow(dead_code)]
use core::arch::asm;
const ECALL_STRING_CREATE: u32 = 525;
const ECALL_STRING_OPS: u32 = 526;
const STRING_OP_TO_STD_STRING: u32 = 7;
const STD_STRING_SSO_SIZE: usize = 16;

#[repr(C)]
pub(self) union CapacityOrSSO {
	pub cap: usize,
	pub sso: [u8; STD_STRING_SSO_SIZE],
}

#[repr(C)]
pub struct GuestStdString {
	// 64-bit pointer + 64-bit length + 64-bit capacity OR 16-byte SSO data
	pub ptr: *const char,
	pub len: usize,
	pub(self) cap_or_sso: CapacityOrSSO,
}

impl GuestStdString
{
	pub fn new(s: &str) -> GuestStdString
	{
		if s.len() <= STD_STRING_SSO_SIZE {
			let mut sso = [0; STD_STRING_SSO_SIZE];
			for i in 0..s.len() {
				sso[i] = s.as_bytes()[i];
			}
			let v = GuestStdString { ptr: s.as_ptr() as *const char, len: s.len(), cap_or_sso: CapacityOrSSO { sso: sso } };
			return v;
		}

		let v = GuestStdString { ptr: s.as_ptr() as *const char, len: s.len(), cap_or_sso: CapacityOrSSO { cap: s.len() } };
		v
	}

	pub fn new_empty() -> GuestStdString
	{
		let v = GuestStdString { ptr: std::ptr::null(), len: 0, cap_or_sso: CapacityOrSSO { cap: 0 } };
		return v;
	}

	pub fn as_str(&self) -> &str
	{
		unsafe {
			let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr as *const u8, self.len));
			return s;
		}
	}

	pub fn as_string(&self) -> String
	{
		unsafe {
			if self.len <= STD_STRING_SSO_SIZE {
				// We want to copy self.ptr, self.len into a new String
				return std::string::String::from_utf8_unchecked(self.as_str().as_bytes().to_vec());
			}
			let s = std::string::String::from_raw_parts(self.ptr as *mut u8, self.len, self.len);
			return s;
		}
	}
}

// Create a new GodotString and return the i32 reference
// MAKE_SYSCALL(ECALL_STRING_CREATE, unsigned, sys_string_create, const char *, size_t);
pub fn godot_string_create(string: &str) -> i32 {
	unsafe {
		let mut gs_ref;
		asm!("ecall",
			in("a7") ECALL_STRING_CREATE,
			in("a0") string.as_ptr(),
			in("a1") string.len(),
			lateout("a0") gs_ref,
		);
		gs_ref
	}
}

pub fn godot_string_to_string(gs: i32) -> String {
	let mut std_string = GuestStdString::new_empty();
	unsafe {
		asm!("ecall",
			in("a7") ECALL_STRING_OPS,
			in("a0") STRING_OP_TO_STD_STRING,
			in("a1") gs,
			in("a2") 0,  // utf-8
			inout("a3") &mut std_string => _, // std::string
			lateout("a0") _,
		);
	}
	std_string.as_string()
}
