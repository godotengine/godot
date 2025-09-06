import subprocess
import tempfile

RISCV_COMPILER = 'riscv64-linux-gnu-gcc-10'
#RISCV_COMPILER = 'riscv64-unknown-linux-gnu-gcc'

def compare_program(testname, program):
	program = program.encode('utf-8')

	with tempfile.NamedTemporaryFile() as tmp:
		result = subprocess.run(['gcc', '-x', 'c', '-static', '-o', tmp.name, '-'],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			input=program)
		tmp.file.close()

		result = subprocess.run([tmp.name],
			stdout=subprocess.PIPE)

		native = result.stdout.decode('utf-8')

	with tempfile.NamedTemporaryFile() as tmp:
		result = subprocess.run([RISCV_COMPILER, '-x', 'c', '-static', '-o', tmp.name, '-'],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			input=program)

		result = subprocess.run(['emulator/build/emulator', tmp.name],
			stdout=subprocess.PIPE)

		emu = result.stdout.decode('utf-8')

	if native != emu:
		print("Test {} mismatch!".format(testname))
		print("Native: " + native)
		print("Emulator: " + emu)
		exit(1)

compare_program("Prints Hello World",
"""#include <stdio.h>
int main() {
	printf("Hello World!\\n");
}""")

def build_IMM32_program(type, fmt, op):
	return """#include <stdio.h>
	int main() {
		volatile TYPE a = 1;
		volatile TYPE b = 1;
		printf("FMT ", a + b);
		a = -1;
		printf("FMT ", a + b);
		b = -1;
		printf("FMT ", a + b);
		a = 0xffffffff;
		b = 1;
		printf("FMT ", a + b);
		a = 0xffffffffffffffff;
		b = 0xffffffffffffffff;
		printf("FMT ", a + b);
		a = 1;
		b = -1;
		printf("FMT\\n", a + b);
	}""".replace("TYPE", type).replace("FMT", fmt).replace("+", op)


compare_program("u8 Plus", build_IMM32_program('unsigned char', '%hhu', '+'))
compare_program("u8 Minus", build_IMM32_program('unsigned char', '%hhu', '-'))
compare_program("u8 Mult", build_IMM32_program('unsigned char', '%hhu', '*'))
compare_program("u8 Div", build_IMM32_program('unsigned char', '%hhu', '/'))
compare_program("u8 And", build_IMM32_program('unsigned char', '%hhu', '&'))
compare_program("u8 Or", build_IMM32_program('unsigned char', '%hhu', '|'))
compare_program("u8 Xor", build_IMM32_program('unsigned char', '%hhu', '^'))

compare_program("i8 Plus", build_IMM32_program('char', '%hhd', '+'))
compare_program("i8 Minus", build_IMM32_program('char', '%hhd', '-'))
compare_program("i8 Mult", build_IMM32_program('char', '%hhd', '*'))
#compare_program("i8 Div", build_IMM32_program('char', '%hhd', '/'))
compare_program("i8 And", build_IMM32_program('char', '%hhd', '&'))
compare_program("i8 Or", build_IMM32_program('char', '%hhd', '|'))
compare_program("i8 Xor", build_IMM32_program('char', '%hhd', '^'))

compare_program("u16 Plus", build_IMM32_program('unsigned short', '%hu', '+'))
compare_program("u16 Minus", build_IMM32_program('unsigned short', '%hu', '-'))
compare_program("u16 Mult", build_IMM32_program('unsigned short', '%hu', '*'))
compare_program("u16 Div", build_IMM32_program('unsigned short', '%hu', '/'))
compare_program("u16 And", build_IMM32_program('unsigned short', '%hu', '&'))
compare_program("u16 Or", build_IMM32_program('unsigned short', '%hu', '|'))
compare_program("u16 Xor", build_IMM32_program('unsigned short', '%hu', '^'))

compare_program("i16 Plus", build_IMM32_program('short', '%hd', '+'))
compare_program("i16 Minus", build_IMM32_program('short', '%hd', '-'))
compare_program("i16 Mult", build_IMM32_program('short', '%hd', '*'))
compare_program("i16 Div", build_IMM32_program('short', '%hd', '/'))
compare_program("i16 And", build_IMM32_program('short', '%hd', '&'))
compare_program("i16 Or", build_IMM32_program('short', '%hd', '|'))
compare_program("i16 Xor", build_IMM32_program('short', '%hd', '^'))

compare_program("u32 Plus", build_IMM32_program('unsigned', '%u', '+'))
compare_program("u32 Minus", build_IMM32_program('unsigned', '%u', '-'))
compare_program("u32 Mult", build_IMM32_program('unsigned', '%u', '*'))
compare_program("u32 Div", build_IMM32_program('unsigned', '%u', '/'))
compare_program("u32 And", build_IMM32_program('unsigned', '%u', '&'))
compare_program("u32 Or", build_IMM32_program('unsigned', '%u', '|'))
compare_program("u32 Xor", build_IMM32_program('unsigned', '%u', '^'))

compare_program("i32 Plus", build_IMM32_program('int', '%d', '+'))
compare_program("i32 Minus", build_IMM32_program('int', '%d', '-'))
compare_program("i32 Mult", build_IMM32_program('int', '%d', '*'))
compare_program("i32 Div", build_IMM32_program('int', '%d', '/'))
compare_program("i32 And", build_IMM32_program('int', '%d', '&'))
compare_program("i32 Or", build_IMM32_program('int', '%d', '|'))
compare_program("i32 Xor", build_IMM32_program('int', '%d', '^'))
