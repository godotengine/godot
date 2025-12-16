# Sample GDScript for testing ELF64 compilation
# Usage: godot --headless --script tools/gdscript_elf64_compiler.gd tools/test_elf64_sample.gd

func add(a: int, b: int) -> int:
	return a + b

func multiply(x: int, y: int) -> int:
	return x * y

func subtract(a: int, b: int) -> int:
	return a - b

func divide(a: float, b: float) -> float:
	if b == 0.0:
		return 0.0
	return a / b
