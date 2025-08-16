#![allow(dead_code)]
use super::syscalls::*;
use crate::Variant;
use crate::VariantType;

#[repr(C)]
pub struct GodotString {
	pub reference: i32
}

impl GodotString {
	pub fn new() -> GodotString {
		GodotString {
			reference: i32::MIN
		}
	}

	pub fn create(s: &str) -> GodotString {
		let mut godot_string = GodotString::new();
		godot_string.reference = godot_string_create(s);
		godot_string
	}

	pub fn from_ref(var_ref: i32) -> GodotString {
		GodotString {
			reference: var_ref
		}
	}

	pub fn to_string(&self) -> String {
		godot_string_to_string(self.reference)
	}

	pub fn to_variant(&self) -> Variant {
		let mut v = Variant::new_nil();
		v.t = VariantType::String;
		v.u.i = self.reference as i64;
		v
	}

	/* Godot String API */
	pub fn length(&self) -> i64 {
		return self.call("length", &[]).to_integer();
	}

	/* Make a method call on the string (as Variant) */
	pub fn call(&self, method: &str, args: &[Variant]) -> Variant {
		// Call the method using Variant::callp
		let var = Variant::from_ref(VariantType::String, self.reference);
		var.call(method, &args)
	}
}
