#![allow(dead_code)]
use core::arch::asm;
use crate::Variant;
use crate::VariantType;

#[repr(C)]
pub struct GodotDictionary {
	pub reference: i32
}

impl GodotDictionary {
	pub fn new() -> GodotDictionary {
		let mut var = Variant::new_nil();
		unsafe {
			asm!("ecall",
				in("a0") &mut var,
				in("a1") VariantType::Dictionary as i32,
				in("a2") 0,   // method
				in("a7") 517, // ECALL_VCREATE
			);
		}
		GodotDictionary {
			reference: unsafe { var.u.i } as i32
		}
	}

	pub fn from_ref(var_ref: i32) -> GodotDictionary {
		GodotDictionary {
			reference: var_ref
		}
	}

	pub fn to_variant(&self) -> Variant {
		let mut v = Variant::new_nil();
		v.t = VariantType::Array;
		v.u.i = self.reference as i64;
		v
	}

	/* Godot Dictionary API */
	pub fn size(&self) -> i64 {
		return self.call("size", &[]).to_integer();
	}

	pub fn empty(&self) -> bool {
		return self.size() == 0;
	}

	pub fn clear(&self) {
		self.call("clear", &[]);
	}

	pub fn get(&self, key: &Variant) -> Variant {
		const ECALL_DICTIONARY_OPS: i32 = 524;
		let mut var = Variant::new_nil();
		unsafe {
			asm!("ecall",
				in("a0") 0, // OP_GET
				in("a1") self.reference,
				in("a2") key,
				in("a3") &mut var,
				in("a7") ECALL_DICTIONARY_OPS,
			);
		}
		var
	}
	pub fn set(&self, key: &Variant, value: &Variant) {
		const ECALL_DICTIONARY_OPS: i32 = 524;
		unsafe {
			asm!("ecall",
				in("a0") 1, // OP_SET
				in("a1") self.reference,
				in("a2") key,
				in("a3") value,
				in("a7") ECALL_DICTIONARY_OPS,
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
