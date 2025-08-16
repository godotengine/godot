//
// Godot Sandbox Zig API
//
pub const V = union { b: bool, i: i64, f: f64, obj: u64, bytes: [16]u8 };
pub extern fn sys_vcall(self: *Variant, method: [*]const u8, method_len: usize, args: [*]Variant, args_len: usize, result: *Variant) void;

pub const Variant = struct {
    type: i64,
    v: V,

    pub fn nil() Variant {
        return Variant{ .type = 0, .v = V{ .b = false } };
    }

    pub fn init_bool(self: *Variant, b: bool) void {
        self.type = 1; // BOOL
        self.v.b = b;
    }

    pub fn init_int(self: *Variant, int: i64) void {
        self.type = 2; // INT
        self.v.i = int;
    }

    pub fn init_float(self: *Variant, f: f64) void {
        self.type = 3; // FLOAT
        self.v.f = f;
    }

    pub fn init_object(self: *Variant, obj: u64) void {
        self.type = 4; // OBJECT
        self.v.obj = obj;
    }

    pub fn call(self: *Variant, method: []const u8, args: []Variant) Variant {
        var result: Variant = undefined;
        sys_vcall(self, method.ptr, method.len, args.ptr, args.len, &result);
        return result;
    }
};

comptime {
    asm (
        \\.global sys_vcall;
        \\.type sys_vcall, @function;
        \\sys_vcall:
        \\  li a7, 501
        \\  ecall
        \\  ret
        \\.global fast_exit;
        \\.type fast_exit, @function;
        \\fast_exit:
        \\  .insn i SYSTEM, 0, x0, x0, 0x7ff
        \\.pushsection .comment
        \\.string "Godot Zig API v1"
        \\.popsection
    );
}
