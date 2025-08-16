#![allow(dead_code)]
use core::arch::asm;
use core::arch::global_asm;
use core::ffi::c_void;
use crate::godot::variant;

pub struct Engine
{
}

impl Engine
{
	pub fn is_editor_hint() -> bool
	{
		let is_editor: i32;
		unsafe {
			asm!("ecall",
				in("a7") 512, // ECALL_IS_EDITOR
				lateout("a0") is_editor,
				options(nostack));
		}
		return is_editor != 0;
	}
}

#[repr(C)]
struct GuestFunctionExtra
{
	pub desc: *const u8,
	pub desc_len: usize,
	pub return_type: *const u8,
	pub return_type_len: usize,
	pub args: *const u8,
	pub args_len: usize,
}

fn register_public_api_c_func(name: &str, address: *mut c_void, return_type: &str, args: &str) {
	let description = "";
	let extra = GuestFunctionExtra {
		desc: description.as_ptr(),
		desc_len: description.len(),
		return_type: return_type.as_ptr(),
		return_type_len: return_type.len(),
		args: args.as_ptr(),
		args_len: args.len(),
	};
	unsafe {
		asm!("ecall",
			in("a7") 547, // ECALL_SANDBOX_ADD
			in("a0") 1, // SANDBOX_PUBLIC_API
			in("a1") name.as_ptr(),
			in("a2") name.len(),
			in("a3") address,
			in("a4") &extra,
			options(nostack));
	}
}

// Register a public API function with a single argument
pub fn register_public_api_func0(name: &str, func: extern "C" fn() -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}
pub fn register_public_api_func1<T>(name: &str, func: extern "C" fn(T) -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}
pub fn register_public_api_func2<T, U>(name: &str, func: extern "C" fn(T, U) -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}
pub fn register_public_api_func3<T, U, V>(name: &str, func: extern "C" fn(T, U, V) -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}
pub fn register_public_api_func4<T, U, V, W>(name: &str, func: extern "C" fn(T, U, V, W) -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}
pub fn register_public_api_func5<T, U, V, W, X>(name: &str, func: extern "C" fn(T, U, V, W, X) -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}
pub fn register_public_api_func6<T, U, V, W, X, Y>(name: &str, func: extern "C" fn(T, U, V, W, X, Y) -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}
pub fn register_public_api_func7<T, U, V, W, X, Y, Z>(name: &str, func: extern "C" fn(T, U, V, W, X, Y, Z) -> variant::Variant, return_type: &str, args: &str) {
	register_public_api_c_func(name, func as *mut c_void, return_type, args);
}

// Godot Rust API version embedded in the binary
global_asm!(
	".pushsection .comment",
	".string \"Godot Rust API v5\"",
	".popsection",
);
