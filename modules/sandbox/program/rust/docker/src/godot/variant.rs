#![allow(dead_code)]
use core::arch::asm;
use std::fmt;
use super::string::*;

#[repr(C)]
pub enum VariantType {
	Nil,
	Bool,
	Integer,
	Float,
	String,

	Vector2,
	Vector2i,
	Rect2,
	Rect2i,
	Vector3,
	Vector3i,
	Transform2D,
	Vector4,
	Vector4i,
	Plane,
	Quaterion,
	AABB,
	Basis,
	Transform3D,
	Projection,

	Color,
	StringName,
	NodePath,
	RID,
	Object,
	Callable,
	Signal,
	Dictionary,
	Array,

	// Packed arrays
	PackedByteArray,
	PackedInt32Array,
	PackedInt64Array,
	PackedFloat32Array,
	PackedFloat64Array,
	PackedStringArray,
	PackedVector2Array,
	PackedVector3Array,
	PackedColorArray,
	PackedVector4Array,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vector2 {
	pub x: f32,
	pub y: f32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vector2i {
	pub x: i32,
	pub y: i32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vector3 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vector3i {
	pub x: i32,
	pub y: i32,
	pub z: i32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Rect2 {
	pub position: Vector2,
	pub size: Vector2,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Rect2i {
	pub position: Vector2i,
	pub size: Vector2i,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vector4 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
	pub w: f32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vector4i {
	pub x: i32,
	pub y: i32,
	pub z: i32,
	pub w: i32,
}

#[repr(C)]
pub union VariantUnion {
	pub b: bool,
	pub i: i64,
	pub f: f64,
	pub v2: Vector2,
	pub v2i: Vector2i,
	pub v3: Vector3,
	pub v3i: Vector3i,
	pub r2: Rect2,
	pub r2i: Rect2i,
	pub v4: Vector4,
	pub v4i: Vector4i,
}

#[repr(C)]
pub struct Variant
{
	pub t : VariantType,
	pub u : VariantUnion,
}

impl Variant
{
	pub fn new_nil() -> Variant
	{
		let v = Variant { t: VariantType::Nil, u: VariantUnion { b: false } };
		return v;
	}

	pub fn new_bool(b: bool) -> Variant
	{
		let v = Variant { t: VariantType::Bool, u: VariantUnion { b: b } };
		return v;
	}
	pub fn new_integer(i: i64) -> Variant
	{
		let v = Variant { t: VariantType::Integer, u: VariantUnion { i: i } };
		return v;
	}
	pub fn new_float(f: f64) -> Variant
	{
		let v = Variant { t: VariantType::Float, u: VariantUnion { f: f } };
		return v;
	}
	pub fn new_vec2(v: Vector2) -> Variant
	{
		let res = Variant { t: VariantType::Vector2, u: VariantUnion { v2: v } };
		return res;
	}
	pub fn new_vec2i(v: Vector2i) -> Variant
	{
		let res = Variant { t: VariantType::Vector2i, u: VariantUnion { v2i: v } };
		return res;
	}
	pub fn new_vec3(v: Vector3) -> Variant
	{
		let res = Variant { t: VariantType::Vector3, u: VariantUnion { v3: v } };
		return res;
	}
	pub fn new_vec3i(v: Vector3i) -> Variant
	{
		let res = Variant { t: VariantType::Vector3i, u: VariantUnion { v3i: v } };
		return res;
	}
	pub fn new_rect2(r: Rect2) -> Variant
	{
		let res = Variant { t: VariantType::Rect2, u: VariantUnion { r2: r } };
		return res;
	}
	pub fn new_rect2i(r: Rect2i) -> Variant
	{
		let res = Variant { t: VariantType::Rect2i, u: VariantUnion { r2i: r } };
		return res;
	}
	pub fn new_vec4(v: Vector4) -> Variant
	{
		let res = Variant { t: VariantType::Vector4, u: VariantUnion { v4: v } };
		return res;
	}
	pub fn new_vec4i(v: Vector4i) -> Variant
	{
		let res = Variant { t: VariantType::Vector4i, u: VariantUnion { v4i: v } };
		return res;
	}
	pub fn new_string(s: &str) -> Variant
	{
		let v = Variant::internal_create_string(VariantType::String, s);
		return v;
	}
	pub fn from_ref(t: VariantType, r: i32) -> Variant
	{
		let v = Variant { t: t, u: VariantUnion { i: r.into() } };
		return v;
	}

	pub fn to_bool(&self) -> bool
	{
		match self.t {
			VariantType::Bool => {
				let b = unsafe { self.u.b };
				return b
			},
			_ => panic!("Variant is not a bool"),
		}
	}

	pub fn to_integer(&self) -> i64
	{
		match self.t {
			VariantType::Integer => {
				let i = unsafe { self.u.i };
				return i
			},
			_ => panic!("Variant is not an integer"),
		}
	}

	pub fn to_float(&self) -> f64
	{
		match self.t {
			VariantType::Float => {
				let f = unsafe { self.u.f };
				return f
			},
			_ => panic!("Variant is not a float"),
		}
	}

	pub fn to_vec2(&self) -> Vector2
	{
		match self.t {
			VariantType::Vector2 => {
				let v2 = unsafe { self.u.v2 };
				return v2
			},
			_ => panic!("Variant is not a Vector2"),
		}
	}

	pub fn to_vec2i(&self) -> Vector2i
	{
		match self.t {
			VariantType::Vector2i => {
				let v2i = unsafe { self.u.v2i };
				return v2i
			},
			_ => panic!("Variant is not a Vector2i"),
		}
	}

	pub fn to_vec3(&self) -> Vector3
	{
		match self.t {
			VariantType::Vector3 => {
				let v3 = unsafe { self.u.v3 };
				return v3
			},
			_ => panic!("Variant is not a Vector3"),
		}
	}

	pub fn to_vec3i(&self) -> Vector3i
	{
		match self.t {
			VariantType::Vector3i => {
				let v3i = unsafe { self.u.v3i };
				return v3i
			},
			_ => panic!("Variant is not a Vector3i"),
		}
	}

	pub fn to_rect2(&self) -> Rect2
	{
		match self.t {
			VariantType::Rect2 => {
				let r2 = unsafe { self.u.r2 };
				return r2
			},
			_ => panic!("Variant is not a Rect2"),
		}
	}

	pub fn to_rect2i(&self) -> Rect2i
	{
		match self.t {
			VariantType::Rect2i => {
				let r2i = unsafe { self.u.r2i };
				return r2i
			},
			_ => panic!("Variant is not a Rect2i"),
		}
	}

	pub fn to_vec4(&self) -> Vector4
	{
		match self.t {
			VariantType::Vector4 => {
				let v4 = unsafe { self.u.v4 };
				return v4
			},
			_ => panic!("Variant is not a Vector4"),
		}
	}

	pub fn to_vec4i(&self) -> Vector4i
	{
		match self.t {
			VariantType::Vector4i => {
				let v4i = unsafe { self.u.v4i };
				return v4i
			},
			_ => panic!("Variant is not a Vector4i"),
		}
	}

	pub fn to_string(&self) -> String
	{
		match self.t {
			VariantType::String => {
				let gs = GodotString::from_ref(unsafe { self.u.i } as i32);
				gs.to_string()
			},
			VariantType::StringName => {
				let gs = GodotString::from_ref(unsafe { self.u.i } as i32);
				gs.to_string()
			},
			VariantType::NodePath => {
				let gs = GodotString::from_ref(unsafe { self.u.i } as i32);
				gs.to_string()
			},
			_ => panic!("Variant is not a String"),
		}
	}

	pub fn internal_create_string(t: VariantType, s: &str) -> Variant
	{
		let gs = GodotString::create(s);
		let mut v = Variant::new_nil();
		v.t = t;
		v.u.i = gs.reference as i64;
		return v;
	}

	pub fn call(&self, method: &str, args: &[Variant]) -> Variant
	{
		const SYSCALL_VCALL: i64 = 501;
		let mut res = Variant::new_nil();
		unsafe {
			asm!("ecall",
				in("a0") self,
				in("a1") method.as_ptr(),
				in("a2") method.len(),
				in("a3") args.as_ptr(),
				in("a4") args.len(),
				in("a5") &mut res,
				in("a7") SYSCALL_VCALL,
				options(nostack));
		}
		return res;
	}
}

impl Vector2
{
	pub fn new(x: f32, y: f32) -> Vector2
	{
		Vector2 { x: x, y: y }
	}
}

impl Vector2i
{
	pub fn new(x: i32, y: i32) -> Vector2i
	{
		Vector2i { x: x, y: y }
	}
}

impl Vector3
{
	pub fn new(x: f32, y: f32, z: f32) -> Vector3
	{
		Vector3 { x: x, y: y, z: z }
	}
}

impl Vector3i
{
	pub fn new(x: i32, y: i32, z: i32) -> Vector3i
	{
		Vector3i { x: x, y: y, z: z }
	}
}

impl Rect2
{
	pub fn new(position: Vector2, size: Vector2) -> Rect2
	{
		Rect2 { position: position, size: size }
	}
}

impl Rect2i
{
	pub fn new(position: Vector2i, size: Vector2i) -> Rect2i
	{
		Rect2i { position: position, size: size }
	}
}

impl Vector4
{
	pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vector4
	{
		Vector4 { x: x, y: y, z: z, w: w }
	}
}

impl Vector4i
{
	pub fn new(x: i32, y: i32, z: i32, w: i32) -> Vector4i
	{
		Vector4i { x: x, y: y, z: z, w: w }
	}
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self.t {
			VariantType::Bool => {
        		write!(f, "{}", self.to_bool())
			},
			VariantType::Integer => {
				write!(f, "{}", self.to_integer())
			},
			VariantType::Float => {
				write!(f, "{}", self.to_float())
			},
			VariantType::Vector2 => {
				let v = self.to_vec2();
				write!(f, "({}, {})", v.x, v.y)
			},
			VariantType::Vector2i => {
				let v = self.to_vec2i();
				write!(f, "({}, {})", v.x, v.y)
			},
			VariantType::Vector3 => {
				let v = self.to_vec3();
				write!(f, "({}, {}, {})", v.x, v.y, v.z)
			},
			VariantType::Vector3i => {
				let v = self.to_vec3i();
				write!(f, "({}, {}, {})", v.x, v.y, v.z)
			},
			VariantType::Rect2 => {
				let r = self.to_rect2();
				write!(f, "({}, {},  {} {})", r.position.x, r.position.y, r.size.x, r.size.y)
			},
			VariantType::Rect2i => {
				let r = self.to_rect2i();
				write!(f, "({}, {},  {} {})", r.position.x, r.position.y, r.size.x, r.size.y)
			},
			VariantType::Vector4 => {
				let v = self.to_vec4();
				write!(f, "({}, {}, {}, {})", v.x, v.y, v.z, v.w)
			},
			VariantType::Vector4i => {
				let v = self.to_vec4i();
				write!(f, "({}, {}, {}, {})", v.x, v.y, v.z, v.w)
			},
			VariantType::String => {
				write!(f, "{}", self.to_string())
			},
			_ => {
				write!(f, "Unknown")
			},
		} // match
    } // fn
} // impl

const SYSCALL_PRINT: i64 = 500;
const SYSCALL_THROW: i64 = 511;

// The print system call takes a pointer to an array of Variant structs in A0,
// as well as the number of elements in the array in A1.
pub fn print1(v: &Variant)
{
	unsafe {
		asm!("ecall",
			in("a0") v,
			in("a1") 1,
			in("a7") SYSCALL_PRINT,
			lateout("a0") _,
			options(nostack));
	}
}
pub fn print(v: &[Variant])
{
	unsafe {
		asm!("ecall",
			in("a0") v.as_ptr(),
			in("a1") v.len(),
			in("a7") SYSCALL_PRINT,
			lateout("a0") _,
			options(nostack));
	}
}
