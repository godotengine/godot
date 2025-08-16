mod godot;
use godot::api::*;
use godot::node::*;
use godot::string::*;
use godot::variant::*;

const SPEED: f32 = 50.0;

extern "C"
fn _physics_process(delta: f64) -> Variant {
	if Engine::is_editor_hint() {
		return Variant::new_nil();
	}

	let slime = get_node();
	let sprite = get_node_from_path("AnimatedSprite2D");

	let flip_h = sprite.call("is_flipped_h", &[]);
	let direction: f32 = (flip_h.to_bool() as i32 as f32 - 0.5) * -2.0;

	let mut pos = slime.call("get_position", &[]).to_vec2();
	pos.x += direction * SPEED * delta as f32;

	slime.call("set_position", &[Variant::new_vec2(pos)]);

	let ray_right = get_node_from_path("raycast_right");
	let ray_left  = get_node_from_path("raycast_left");
	if direction > 0.0 && ray_right.call("is_colliding", &[]).to_bool() {
		sprite.call("set_flip_h", &[Variant::new_bool(true)]);
	}
	else if direction < 0.0 && ray_left.call("is_colliding", &[]).to_bool() {
		sprite.call("set_flip_h", &[Variant::new_bool(false)]);
	}

	Variant::new_nil()
}

extern "C"
fn test(val1: i32, val2: i32, val3: GodotString) -> Variant {
	let sum = val1 + val2 + val3.to_string().parse::<i32>().unwrap();
	Variant::new_integer(sum.into())
}

pub fn main() {
	let v = Variant::new_string("Hello, Rust World!");
	println!("{}", v.to_string());

	register_public_api_func1("_physics_process", _physics_process, "Variant", "f64");
	register_public_api_func3("test", test, "int", "int a, int b, String c");
}
