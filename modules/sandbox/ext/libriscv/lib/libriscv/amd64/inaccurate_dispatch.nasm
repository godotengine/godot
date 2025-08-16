[bits 64]
DEFAULT REL
section .text
global riscv64gb_inaccurate_dispatch
extern riscv64gb_cpu_relative_icounter
extern riscv64gb_cpu_relative_imaxcounter
extern riscv64gb_cpu_syscall_array
%define CPU_PC  rdi + 32*8   ; Program Counter
%define CPU_REG rdi          ; CPU register base address
%macro FETCH_REG 2
	;; Fetch a register value from the CPU state
	;; %1 - destination register
	;; %2 - source register index
	mov %1, [rdi + %2 * 8] ; Load the value from the CPU register
%endmacro
%macro STORE_REG 2
	;; Store a value into a CPU register
	;; %1 - destination register index
	;; %2 - source register
	mov [rdi + %1 * 8], %2 ; Store the value in the CPU register
%endmacro
%macro JUMP_BYTECODE 0
	;; Load the next bytecode from the decoder cache
	movzx eax, byte [rdx]  ; Load the bytecode index
	;; Jump to the handler for the next bytecode
	mov r15d, DWORD [dispatch_table + eax * 4] ; Relative address
	jmp r15 ; Jump to the bytecode handler
%endmacro
%macro NEXT_BYTECODE 0
	;; Load the next bytecode from the decoder cache
	movzx eax, byte [rdx + 8]  ; Load the bytecode index
	;; Increment decoder cache entry *ONLY*
	add rdx, 8
	;; Jump to the handler for the next bytecode
	mov r15d, DWORD [dispatch_table + eax * 4] ; Relative address
	jmp r15 ; Jump to the bytecode handler
%endmacro
%macro JUMP_PREPARED_BYTECODE 0
	;; Load the next bytecode from the decoder cache
	movzx r15d, byte [rdx + 8]  ; Load the bytecode index
	;; Increment decoder cache entry *ONLY*
	add rdx, 8
	;; Jump to the handler for the next bytecode
	mov r15d, DWORD [dispatch_table + r15d * 4] ; Relative address
	jmp r15 ; Jump to the bytecode handler
%endmacro
%macro NEXT_BLOCK 1
	;; Increment PC
	add rdx, %1 * 8
	add rcx, %1 * 4
	;; Load the 16-bit block size
	movzx rax, word [rdx + 2]       ; Load the block size
	;; Increment PC by the block size * 4
	shl rax, 2                ; Convert block size to bytes
	add rcx, rax
	JUMP_BYTECODE
%endmacro
%macro NEXT_BLOCK_REG_BYTES 1
	;; Increment PC *and* decoder by %1 bytes
	add rcx, %1
	shl %1, 1                ; Convert block size to decoder cache entry size
	add rdx, %1
	;; Load the 16-bit block size
	movzx rax, word [rdx + 2]       ; Load the block size
	;; Increment PC by the block size * 4
	shl rax, 2                ; Convert block size to bytes
	add rcx, rax
	JUMP_BYTECODE
%endmacro
%macro PUSH_SYSV_REGS 0
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	;push rax
%endmacro
%macro POP_SYSV_REGS 0
	;pop rax
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
%endmacro

;; Dispatch table contains 32-bit relative jump locations
section .text
align 64
dispatch_table:
	dd 0x00000000 ;; RV32I_BC_INVALID
	dd .bytecode_addi ;; RV32I_BC_ADDI
	dd .bytecode_li   ;; RV32I_BC_LI
	dd .bytecode_mv   ;; RV32I_BC_MV

	dd 0x00000000 ;; RV32I_BC_SLLI
	dd 0x00000000 ;; RV32I_BC_SLTI
	dd 0x00000000 ;; RV32I_BC_SLTIU
	dd 0x00000000 ;; RV32I_BC_XORI
	dd 0x00000000 ;; RV32I_BC_SRLI
	dd 0x00000000 ;; RV32I_BC_SRAI
	dd 0x00000000 ;; RV32I_BC_ORI
	dd 0x00000000 ;; RV32I_BC_ANDI

	dd .bytecode_lui ;; RV32I_BC_LUI
	dd 0x00000000 ;; RV32I_BC_AUIPC

	dd 0x00000000 ;; RV32I_BC_LDB
	dd 0x00000000 ;; RV32I_BC_LDBU
	dd 0x00000000 ;; RV32I_BC_LDH
	dd 0x00000000 ;; RV32I_BC_LDHU
	dd 0x00000000 ;; RV32I_BC_LDW

	dd 0x00000000 ;; RV32I_BC_STB
	dd 0x00000000 ;; RV32I_BC_STH
	dd 0x00000000 ;; RV32I_BC_STW

	dd 0x00000000 ;; RV32I_BC_LDWU
	dd 0x00000000 ;; RV32I_BC_LDD
	dd 0x00000000 ;; RV32I_BC_STD

	dd .bytecode_beq ;; RV32I_BC_BEQ
	dd .bytecode_bne ;; RV32I_BC_BNE
	dd 0x00000000 ;; RV32I_BC_BLT
	dd 0x00000000 ;; RV32I_BC_BGE
	dd 0x00000000 ;; RV32I_BC_BLTU
	dd 0x00000000 ;; RV32I_BC_BGEU
	dd .bytecode_beq ;; RV32I_BC_BEQ_FW
	dd .bytecode_bne ;; RV32I_BC_BNE_FW

	dd 0x00000000 ;; RV32I_BC_JAL
	dd 0x00000000 ;; RV32I_BC_JALR
	dd 0x00000000 ;; RV32I_BC_FAST_JAL
	dd 0x00000000 ;; RV32I_BC_FAST_CALL

	dd .bytecode_add ;; RV32I_BC_OP_ADD
	dd 0x00000000 ;; RV32I_BC_OP_SUB
	dd 0x00000000 ;; RV32I_BC_OP_SLL
	dd 0x00000000 ;; RV32I_BC_OP_SLT
	dd 0x00000000 ;; RV32I_BC_OP_SLTU
	dd 0x00000000 ;; RV32I_BC_OP_XOR
	dd 0x00000000 ;; RV32I_BC_OP_SRL
	dd 0x00000000 ;; RV32I_BC_OP_OR
	dd 0x00000000 ;; RV32I_BC_OP_AND
	dd 0x00000000 ;; RV32I_BC_OP_MUL
	dd 0x00000000 ;; RV32I_BC_OP_DIV
	dd 0x00000000 ;; RV32I_BC_OP_DIVU
	dd 0x00000000 ;; RV32I_BC_OP_REM
	dd 0x00000000 ;; RV32I_BC_OP_REMU
	dd 0x00000000 ;; RV32I_BC_OP_SRA
	dd 0x00000000 ;; RV32I_BC_OP_ZEXT_H
	dd 0x00000000 ;; RV32I_BC_OP_SH1ADD
	dd 0x00000000 ;; RV32I_BC_OP_SH2ADD
	dd 0x00000000 ;; RV32I_BC_OP_SH3ADD

	dd 0x00000000 ;; RV32I_BC_SEXT_B
	dd 0x00000000 ;; RV32I_BC_SEXT_H
	dd 0x00000000 ;; RV32I_BC_BSETI
	dd 0x00000000 ;; RV32I_BC_BEXTI

	dd 0x00000000 ;; RV64I_BC_ADDIW
	dd 0x00000000 ;; RV64I_BC_SLLIW
	dd 0x00000000 ;; RV64I_BC_SRLIW
	dd 0x00000000 ;; RV64I_BC_SRAIW
	dd 0x00000000 ;; RV64I_BC_OP_ADDW
	dd 0x00000000 ;; RV64I_BC_OP_SUBW
	dd 0x00000000 ;; RV64I_BC_OP_MULW
	dd 0x00000000 ;; RV64I_BC_OP_ADD_UW
	dd 0x00000000 ;; RV64I_BC_OP_SH1ADD_UW
	dd 0x00000000 ;; RV64I_BC_OP_SH2ADD_UW

	dd .bytecode_syscall ;; RV32I_BC_SYSCALL
	dd .bytecode_stop ;; RV32I_BC_STOP

	dd 0x00000000 ;; RV32F_BC_FLW
	dd 0x00000000 ;; RV32F_BC_FLD
	dd 0x00000000 ;; RV32F_BC_FSW
	dd 0x00000000 ;; RV32F_BC_FSD
	dd 0x00000000 ;; RV32F_BC_FADD
	dd 0x00000000 ;; RV32F_BC_FSUB
	dd 0x00000000 ;; RV32F_BC_FMUL
	dd 0x00000000 ;; RV32F_BC_FDIV
	dd 0x00000000 ;; RV32F_BC_FMADD
	dd 0x00000000 ;; RV32I_BC_FUNCTION
	dd 0x00000000 ;; RV32I_BC_FUNCBLOCK
	dd 0x00000000 ;; RV32I_BC_LIVEPATCH
	dd 0x00000000 ;; RV32I_BC_SYSTEM
	;; BYTECODES_MAX

;; -== Bytecode format ==-
;; uint8_t  bytecode:    index into the dispatch table
;; uint8_t  handler:     only for bytecodes that use external handler
;; uint16_t block_size:  1/2 size of the current block in bytes
;; uint32_t instruction: instruction bits needed by the bytecode/handler

;; --== Bytecode handlers ==-
align 64
.bytecode_addi:
	;; Add immediate value to a register
	;; IMM is [rdx + 0x4] ;: first 16-bits
	;; RS1 is [rdx + 0x6] ;: second 8-bits
	;; RD  is [rdx + 0x7] ;: last 8-bits
	movsx rax, word [rdx + 0x4]
	movzx ebx,  byte [rdx + 0x6] ; Load the source register index
	movzx r10d, byte [rdx + 0x7] ; Load the destination register index
	add rax, [CPU_REG + r10 * 8] ; Add the immediate value to the source register
	STORE_REG rbx, eax ; Store the result in the destination register
	JUMP_PREPARED_BYTECODE
align 16
.bytecode_li:
	;; Load immediate value into a register
	;; RD  is [rdx + 0x4] ;: first 8-bits
	;; IMM is [rdx + 0x6] ;: last 16-bits
	movzx eax, byte [rdx + 0x4]  ; Load the destination register index
	movsx rbx, word [rdx + 0x6]  ; Load the *signed* immediate value
	STORE_REG rax, rbx
	JUMP_PREPARED_BYTECODE
align 16
.bytecode_mv:
	;; Move value from one register to another
	;; RS1 is [rdx + 0x4] ;: first 8-bits
	;; RD  is [rdx + 0x5] ;: second 8-bits
	;; Optimization: Load the entire instruction bits,
	;; then extract the register indices
	movzx ebx, byte [rdx + 0x4]      ; Load 16-bit instruction bits
	movzx eax, byte [rdx + 0x5]      ; Load 16-bit instruction bits
	movzx r10d, bl                ; Copy the lower 8 bits to r10d
	FETCH_REG rbx, r10  ; Fetch the value from the source register
	STORE_REG rax, rbx  ; Store the value in the destination register
	JUMP_PREPARED_BYTECODE
align 16
.bytecode_add:
	;; Add two registers and store the result in a third register
	;; RD  is [rdx + 0x4] ;: first 16-bits
	;; RS1 is [rdx + 0x5] ;: second 8-bits
	;; RS2 is [rdx + 0x6] ;: third 8-bits
	movzx eax, word [rdx + 0x4]  ; Load the destination register index
	movzx ebx, byte [rdx + 0x6]  ; Load the first source register index
	movzx r10, byte [rdx + 0x7]  ; Load the second source register index
	;; Add the two values and store the result in the destination register
	FETCH_REG rbx, rbx
	;FETCH_REG r10, r10 ; Fetch the values from the source registers
	;add rbx, r10
	add rbx, [rdi + r10 * 8]
	STORE_REG rax, rbx ; Store the result in the destination register
	JUMP_PREPARED_BYTECODE
align 16
.bytecode_lui:
	;; Load upper immediate value into a register
	;; IMM is [rdx + 0x4] ;: first 24-bits
	;; RD is [rdx + 0x7]  ;: last 8-bits
	mov ebx, dword [rdx + 0x4]
	shl ebx, 8                   ; Push out upper 8 bits, make 32-bit value
	movsx rax, byte [rdx + 0x7]
	STORE_REG rax, rbx           ; Store the value in the destination register
	JUMP_PREPARED_BYTECODE
align 16
.bytecode_beq:
	;; Branch if equal
	;; IMM is [rdx + 0x4] ;: first 16-bits
	;; RS1 is [rdx + 0x6] ;: third 8-bits
	;; RS2 is [rdx + 0x7] ;: fourth 8-bits
	movzx rax, word [rdx + 0x6]  ; Load the two source register indices
	movzx ebx, ah
	movzx eax, al
	FETCH_REG rax, rax           ; Fetch the value from the first source register
	cmp rax, [CPU_REG + rbx * 8] ; Compare with the value from the second source register
	jne .next_block              ; If not equal, continue to next instruction
	;; We have to perform a jump to the target address
	movsx rax, word [rdx + 0x4]  ; Load the *signed* immediate value (offset)
	NEXT_BLOCK_REG_BYTES rax
align 16
.bytecode_bne:
	;; Branch if not equal
	;; IMM is [rdx + 0x4] ;: first 16-bits
	;; RS1 is [rdx + 0x6] ;: third 8-bits
	;; RS2 is [rdx + 0x7] ;: fourth 8-bits
	movzx rax, byte [rdx + 0x6]  ; Load the first source register index
	movzx rbx, byte [rdx + 0x7]  ; Load the second source register index
	FETCH_REG rax, rax           ; Fetch the value from the first source register
	cmp rax, [CPU_REG + rbx * 8] ; Compare with the value from the second source register
	je .next_block               ; If equal, continue to next instruction
	;; We have to perform a jump to the target address
	movsx rax, word [rdx + 0x4]  ; Load the *signed* immediate value (offset)
	NEXT_BLOCK_REG_BYTES rax
.next_block:
	NEXT_BLOCK 1
align 16
.bytecode_syscall:
	;; Handle a system call in the emulator
	;; All the system call arguments are already in the registers,
	;; but we still have to realize PC, push regs and pop after
	mov [CPU_PC], rcx     ; Store the current PC
	;; Push function call registers
	PUSH_SYSV_REGS
	;; Call the system call handler
	mov eax, DWORD [rdi + 17 * 8]          ; Load the syscall number
	shl eax, 3                             ; Multiply by 8 to get the function pointer offset
	add rax, [riscv64gb_cpu_syscall_array] ; Add the base address of the syscall array
	call [rax]                             ; Call the syscall handler
	;; Restore registers after the system call
	POP_SYSV_REGS
	;; Check if max instructions limit is reached
	mov rax, rdi
	add eax, DWORD [riscv64gb_cpu_relative_icounter]
	mov r10, [rax] ;; Load the current instruction counter
	mov r11, [rax + 8] ;; Load the maximum instruction counter
	cmp r10, r11
	jge .machine_stopped ;; If reached, stop execution
	;; Restore the program counter
	mov rcx, [CPU_PC]
	;; Re-create the decoder cache entry
	;; TODO: If PC is not in the current segment, we need to handle it
	NEXT_BLOCK 1
.machine_stopped:
	;; This is the end of the machine execution.
	ret
.bytecode_stop:
	;; Store current PC + 4 into CPU_PC
	add rcx, 4          ; Move to the next instruction
	mov [CPU_PC], rcx
	ret

;; Arguments:
;; rdi - cpu: CPU pointer (persistent emulator state)
;; rsi - exec: Current execute segment
;; rdx - decoder: Current decoder cache entry (bytecode struct)
;; rcx - pc: Program counter (virtual instruction address)
;; r8  - current_begin: Lowest address of the current execute segment
;; r9  - current_end: Highest address of the current execute segment
riscv64gb_inaccurate_dispatch:
	;; Geronimo!
	JUMP_BYTECODE
; End of riscv64gb_inaccurate_dispatch
