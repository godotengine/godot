#![allow(dead_code)]
use core::arch::asm;
use crate::Variant;
use crate::VariantType;

#[repr(C)]
pub struct GodotArray {
	pub reference: i32
}

impl GodotArray {
	pub fn new() -> GodotArray {
		GodotArray {
			reference: i32::MIN
		}
	}

	pub fn create(array: &[Variant]) -> GodotArray {
		const ECALL_VREATE: i32 = 517;

		let mut var = Variant::new_nil();
		unsafe {
			asm!("ecall",
				in("a0") &mut var, // Result variant
				in("a1") VariantType::Array as i32, // Variant type
				in("a2") 1, // Array from pointer + length
				in("a3") array.as_ptr(),
				in("a4") array.len(),
				in("a7") ECALL_VREATE,
			);
		}
		GodotArray {
			reference: unsafe { var.u.i } as i32
		}
	}

	pub fn from_ref(var_ref: i32) -> GodotArray {
		GodotArray {
			reference: var_ref
		}
	}

	pub fn to_variant(&self) -> Variant {
		let mut v = Variant::new_nil();
		v.t = VariantType::Array;
		v.u.i = self.reference as i64;
		v
	}

	/* Godot Array API */
	pub fn len(&self) -> i64 {
		return self.call("size", &[]).to_integer();
	}

	pub fn empty(&self) -> bool {
		return self.len() == 0;
	}

	pub fn push_back(&self, var: &Variant) {
		// Use slice::from_ref to create a new Variant without copying
		self.call("push_back", std::slice::from_ref(var));
	}
	pub fn append(&self, var: &Variant) {
		self.push_back(var);
	}
	pub fn pop_back(&self) {
		self.call("pop_back", &[]);
	}

	pub fn clear(&self) {
		self.call("clear", &[]);
	}

	pub fn get(&self, idx: i32) -> Variant {
		const ECALL_ARRAY_AT: i32 = 522;
		let mut var = Variant::new_nil();
		unsafe {
			asm!("ecall",
				in("a0") self.reference,
				in("a1") idx,
				in("a2") &mut var,
				in("a7") ECALL_ARRAY_AT,
			);
		}
		var
	}
	pub fn set(&self, idx: i32, var: &Variant) {
		const ECALL_ARRAY_AT: i32 = 522;
		// Set: idx = -idx - 1 with signed overflow
		let set_idx = -(idx as i32) - 1;
		unsafe {
			asm!("ecall",
				in("a0") self.reference,
				in("a1") set_idx,
				in("a2") var,
				in("a7") ECALL_ARRAY_AT,
			);
		}
	}

	/* Make a method call on the string (as Variant) */
	pub fn call(&self, method: &str, args: &[Variant]) -> Variant {
		// Call the method using Variant::callp
		let var = Variant::from_ref(VariantType::Array, self.reference);
		var.call(method, &args)
	}
}
