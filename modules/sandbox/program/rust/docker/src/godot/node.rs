#![allow(dead_code)]
use core::arch::asm;
use core::ffi::c_char;
use crate::godot::variant::Variant;

pub struct Node
{
	address: usize,
}

impl Node
{
	pub fn new(address: usize) -> Node
	{
		Node
		{
			address: address,
		}
	}

	pub fn new_from_path(path: &str) -> Node
	{
		let address = godot_node_get(0, path.as_ptr(), path.len());
		Node::new(address)
	}

	pub fn get_address(&self) -> usize
	{
		self.address
	}

	pub fn get_node(&self, path: &str) -> Node
	{
		let node_address = godot_node_get(self.address, path.as_ptr(), path.len());
		Node::new(node_address)
	}

	pub fn call(&self, method: &str, args: &[Variant]) -> Variant
	{
		let address = self.address;
		return godot_method_call(address, method.as_ptr(), method.len(), args.as_ptr(), args.len());
	}
}

fn godot_node_get(parent_address: usize, path: *const c_char, size: usize) -> usize
{
	const SYSCALL_NODE_GET: i32 = 507;
	let new_address: usize;
	unsafe {
		asm!("ecall",
			in("a0") parent_address,
			in("a1") path,
			in("a2") size,
			in("a7") SYSCALL_NODE_GET,
			lateout("a0") new_address,
			options(nostack));
	}
	return new_address;
}

fn godot_method_call(address: usize, method: *const c_char, msize: usize, args: *const Variant, num_args: usize) -> Variant
{
	const SYSCALL_OBJ_CALLP: i32 = 506;
	// sys_obj_callp(address: usize, method: *const c_char, size: usize, deferred: bool, result: *Variant, args: *const Variant, num_args: usize) -> Variant;
	let mut result: Variant = Variant::new_nil();
	let res_ptr = &mut result as *mut Variant;
	unsafe {
		asm!("ecall",
			in("a0") address,
			in("a1") method,
			in("a2") msize,
			in("a3") 0,
			in("a4") res_ptr,
			in("a5") args,
			in("a6") num_args,
			in("a7") SYSCALL_OBJ_CALLP,
			options(nostack));
		return result;
	}
}

/* Get the current node (owner of the Sandbox) */
pub fn get_node() -> Node
{
	Node::new_from_path(".")
}

/* Get a node relative to the current node (owner of the Sandbox) */
pub fn get_node_from_path(path: &str) -> Node
{
	Node::new_from_path(path)
}
