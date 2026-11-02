#!/usr/bin/env node
// SPIR-V preprocessing tests for the Tint-based WGSL conversion.
//
// Exercises every pass of spirv_preprocess::run_pipeline() end-to-end through tint_convert_cli, which runs
// the same pipeline as the WebGPU driver.
//
// Usage: node run_tests.mjs

import { readdirSync, unlinkSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { REPO_ROOT, convertSpirvFile, findTintCli } from "../tint_cli.mjs";

const FIXTURES_DIR = join(REPO_ROOT, "drivers", "webgpu", "tests", "shader_corpus", "fixtures");
const TINT_CLI = findTintCli();

// ─────────────────────────────────────────────────────────────────────────────
// Test infrastructure
// ─────────────────────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (cond) {
    passed++;
  } else {
    console.log(`  FAIL: ${msg}`);
    failed++;
  }
}

function assertEq(actual, expected, msg) {
  if (JSON.stringify(actual) === JSON.stringify(expected)) {
    passed++;
  } else {
    console.log(`  FAIL: ${msg}`);
    console.log(`    expected: ${JSON.stringify(expected)}`);
    console.log(`    actual:   ${JSON.stringify(actual)}`);
    failed++;
  }
}

// Convert SPIR-V bytes to WGSL. Returns { wgsl, error, stderr }; wgsl is null on failure.
function convertToWgsl(spvBytes) {
  const tmp = join(tmpdir(), `preprocess_test_${process.pid}_${Math.random().toString(36).slice(2)}.spv`);
  try {
    writeFileSync(tmp, spvBytes);
    return convertSpirvFile(TINT_CLI, tmp);
  } finally {
    try { unlinkSync(tmp); } catch (_) {}
  }
}

function convertFixture(name) {
  return convertSpirvFile(TINT_CLI, join(FIXTURES_DIR, name));
}

// Converts a fixture, asserting success; returns "" on failure so later checks fail too.
function fixtureWgsl(name) {
  const r = convertFixture(name);
  assert(r.wgsl !== null, `${name} converts: ${r.error}`);
  return r.wgsl ?? "";
}

// Converts constructed SPIR-V, asserting success.
function constructedWgsl(spv, label) {
  const r = convertToWgsl(spv);
  assert(r.wgsl !== null, `${label} converts: ${r.error}`);
  return r.wgsl ?? "";
}

// Sorted "group,binding" pairs declared by the WGSL.
function bindingsOf(wgsl) {
  return [...wgsl.matchAll(/@group\((\d+)u?\)\s*@binding\((\d+)u?\)/g)].map((m) => `${m[1]},${m[2]}`).sort();
}

// ─────────────────────────────────────────────────────────────────────────────
// SPIR-V construction helpers
// ─────────────────────────────────────────────────────────────────────────────

const SPIRV_MAGIC = 0x07230203;

// Encode a single SPIR-V instruction: (wordCount << 16) | opcode, followed by operands.
function encodeInst(opcode, ...operands) {
  const wc = 1 + operands.length;
  return [(wc << 16) | opcode, ...operands];
}

// Build a SPIR-V 1.3 binary, the version Godot emits for WebGPU, from instruction words.
function buildSpirv(bound, instructionWords) {
  const words = [SPIRV_MAGIC, 0x00010300, 0, bound, 0, ...instructionWords];
  const buf = Buffer.alloc(words.length * 4);
  for (let i = 0; i < words.length; i++) {
    buf.writeUInt32LE(words[i] >>> 0, i * 4);
  }
  return buf;
}

// SPIR-V opcodes.
const Op = {
  Extension: 10, EntryPoint: 15, ExecutionMode: 16, Capability: 17, MemoryModel: 14,
  TypeVoid: 19, TypeBool: 20, TypeInt: 21, TypeFloat: 22, TypeVector: 23,
  TypeStruct: 30, TypePointer: 32, TypeFunction: 33,
  Constant: 43, SpecConstantTrue: 48, SpecConstantFalse: 49, SpecConstant: 50,
  SpecConstantComposite: 51, SpecConstantOp: 52,
  Function: 54, FunctionEnd: 56, Variable: 59, Load: 61, Store: 62, AccessChain: 65,
  Decorate: 71, MemberDecorate: 72, CompositeConstruct: 80, CompositeExtract: 81,
  Select: 169, IEqual: 170, INotEqual: 171, SLessThan: 177,
  Label: 248, Return: 253, TerminateInvocation: 4416,
};

// SPIR-V opcodes evaluated by OpSpecConstantOp.
const SpecOp = {
  SNegate: 126, IAdd: 128, ISub: 130, IMul: 132, UDiv: 134, SDiv: 135, UMod: 137, SRem: 138, SMod: 139,
  ShiftRightLogical: 194, ShiftLeftLogical: 196, BitwiseOr: 197, BitwiseXor: 198, BitwiseAnd: 199, Not: 200,
};

// Decoration IDs.
const Deco = { SpecId: 1, Block: 2, BuiltIn: 11, Binding: 33, DescriptorSet: 34, Offset: 35 };

// Storage classes.
const SC = { Output: 3, PushConstant: 9, StorageBuffer: 12 };

// Execution models.
const ExecModel = { Vertex: 0, Fragment: 4, GLCompute: 5 };

// Encode a string as SPIR-V literal words (null-terminated, padded to 4 bytes).
function encodeString(s) {
  const bytes = Buffer.from(s + "\0", "utf-8");
  const padded = Buffer.alloc(Math.ceil(bytes.length / 4) * 4);
  bytes.copy(padded);
  const words = [];
  for (let i = 0; i < padded.length; i += 4) {
    words.push(padded.readUInt32LE(i));
  }
  return words;
}

// Encode an OpEntryPoint instruction with correct word count.
function encodeEntryPoint(execModel, funcId, name, ...interfaceVars) {
  const nameWords = encodeString(name);
  const wc = 3 + nameWords.length + interfaceVars.length;
  return [(wc << 16) | Op.EntryPoint, execModel, funcId, ...nameWords, ...interfaceVars];
}

const COMPUTE_PREAMBLE = (main) => [
  ...encodeInst(Op.Capability, 1),
  ...encodeInst(Op.MemoryModel, 0, 1),
  ...encodeEntryPoint(ExecModel.GLCompute, main, "main"),
  ...encodeInst(Op.ExecutionMode, main, 17, 1, 1, 1),
];

// Build a minimal vertex shader that outputs gl_Position.
function buildMinimalVertexShader() {
  // IDs: 1=void, 2=fn_type, 3=float, 4=vec4, 5=ptr_output_vec4,
  //      6=gl_Position, 7=main, 8=label, 9=const0, 10=const1, 11=constructed
  return buildSpirv(12, [
    ...encodeInst(Op.Capability, 1),
    ...encodeInst(Op.MemoryModel, 0, 1),
    ...encodeEntryPoint(ExecModel.Vertex, 7, "main", 6),
    ...encodeInst(Op.Decorate, 6, Deco.BuiltIn, 0), // Position
    ...encodeInst(Op.TypeVoid, 1),
    ...encodeInst(Op.TypeFunction, 2, 1),
    ...encodeInst(Op.TypeFloat, 3, 32),
    ...encodeInst(Op.TypeVector, 4, 3, 4),
    ...encodeInst(Op.TypePointer, 5, SC.Output, 4),
    ...encodeInst(Op.Variable, 5, 6, SC.Output),
    ...encodeInst(Op.Constant, 3, 9, 0x00000000), // 0.0
    ...encodeInst(Op.Constant, 3, 10, 0x3F800000), // 1.0
    ...encodeInst(Op.Function, 1, 7, 0, 2),
    ...encodeInst(Op.Label, 8),
    ...encodeInst(Op.CompositeConstruct, 4, 11, 9, 9, 9, 10),
    ...encodeInst(Op.Store, 6, 11),
    ...encodeInst(Op.Return),
    ...encodeInst(Op.FunctionEnd),
  ]);
}

// Build a minimal fragment shader with OpTerminateInvocation.
function buildFragmentWithTerminateInvocation() {
  return buildSpirv(10, [
    ...encodeInst(Op.Capability, 1),
    ...encodeInst(Op.Extension, ...encodeString("SPV_KHR_terminate_invocation")),
    ...encodeInst(Op.MemoryModel, 0, 1),
    ...encodeEntryPoint(ExecModel.Fragment, 5, "main"),
    ...encodeInst(Op.ExecutionMode, 5, 7), // OriginUpperLeft
    ...encodeInst(Op.TypeVoid, 1),
    ...encodeInst(Op.TypeFunction, 2, 1),
    ...encodeInst(Op.Function, 1, 5, 0, 2),
    ...encodeInst(Op.Label, 6),
    ...encodeInst(Op.TerminateInvocation),
    ...encodeInst(Op.FunctionEnd),
  ]);
}

// Build a minimal compute shader.
function buildMinimalComputeShader() {
  return buildSpirv(8, [
    ...COMPUTE_PREAMBLE(5),
    ...encodeInst(Op.TypeVoid, 1),
    ...encodeInst(Op.TypeFunction, 2, 1),
    ...encodeInst(Op.Function, 1, 5, 0, 2),
    ...encodeInst(Op.Label, 6),
    ...encodeInst(Op.Return),
    ...encodeInst(Op.FunctionEnd),
  ]);
}

// Build a compute shader with push constants.
function buildComputeWithPushConstants() {
  // IDs: 1=void, 2=fn_type, 3=uint, 4=struct{uint}, 5=ptr_pc,
  //      6=pc_var, 7=main, 8=label, 9=const0, 10=ptr_uint, 11=ac, 12=loaded
  return buildSpirv(13, [
    ...COMPUTE_PREAMBLE(7),
    ...encodeInst(Op.Decorate, 4, Deco.Block),
    ...encodeInst(Op.MemberDecorate, 4, 0, Deco.Offset, 0),
    ...encodeInst(Op.TypeVoid, 1),
    ...encodeInst(Op.TypeFunction, 2, 1),
    ...encodeInst(Op.TypeInt, 3, 32, 0), // uint
    ...encodeInst(Op.TypePointer, 10, SC.PushConstant, 3),
    ...encodeInst(Op.TypeStruct, 4, 3),
    ...encodeInst(Op.TypePointer, 5, SC.PushConstant, 4),
    ...encodeInst(Op.Variable, 5, 6, SC.PushConstant),
    ...encodeInst(Op.Constant, 3, 9, 0),
    ...encodeInst(Op.Function, 1, 7, 0, 2),
    ...encodeInst(Op.Label, 8),
    ...encodeInst(Op.AccessChain, 10, 11, 6, 9),
    ...encodeInst(Op.Load, 3, 12, 11),
    ...encodeInst(Op.Return),
    ...encodeInst(Op.FunctionEnd),
  ]);
}

// Compute shader with optional spec constants a (SpecId 0) and b (SpecId 1) that stores the last value built by
// `values` into a storage buffer. Ordinary operands test folding; specializable operands retain IDs.
// IDs: 3=int, 4=a, 5=b, 14=bool, 16=v2int; `values(id)` returns instructions defining ids 20, 21, ...
function buildSpecOpStore({ signed = true, a = 10, b = 20, values, specializable = false }) {
  const defs = values(20);
  const result = 20 + defs.length - 1;
  return buildSpirv(40, [
    ...COMPUTE_PREAMBLE(11),
    ...(specializable ? encodeInst(Op.Decorate, 4, Deco.SpecId, 0) : []),
    ...(specializable ? encodeInst(Op.Decorate, 5, Deco.SpecId, 1) : []),
    ...encodeInst(Op.Decorate, 6, Deco.Block),
    ...encodeInst(Op.MemberDecorate, 6, 0, Deco.Offset, 0),
    ...encodeInst(Op.Decorate, 9, Deco.DescriptorSet, 0),
    ...encodeInst(Op.Decorate, 9, Deco.Binding, 0),
    ...encodeInst(Op.TypeVoid, 1),
    ...encodeInst(Op.TypeFunction, 2, 1),
    ...encodeInst(Op.TypeInt, 3, 32, signed ? 1 : 0),
    ...encodeInst(Op.TypeBool, 14),
    ...encodeInst(Op.TypeVector, 16, 3, 2),
    ...encodeInst(specializable ? Op.SpecConstant : Op.Constant, 3, 4, a >>> 0),
    ...encodeInst(specializable ? Op.SpecConstant : Op.Constant, 3, 5, b >>> 0),
    ...encodeInst(Op.TypeStruct, 6, 3),
    ...encodeInst(Op.TypePointer, 7, SC.StorageBuffer, 6),
    ...encodeInst(Op.TypePointer, 8, SC.StorageBuffer, 3),
    ...encodeInst(Op.Variable, 7, 9, SC.StorageBuffer),
    ...encodeInst(Op.Constant, 3, 10, 0),
    ...defs.flat(),
    ...encodeInst(Op.Function, 1, 11, 0, 2),
    ...encodeInst(Op.Label, 12),
    ...encodeInst(Op.AccessChain, 8, 13, 9, 10),
    ...encodeInst(Op.Store, 13, result),
    ...encodeInst(Op.Return),
    ...encodeInst(Op.FunctionEnd),
  ]);
}

// A single OpSpecConstantOp on a and b with an int result.
const specOp = (opcode, ...args) => (id) => [encodeInst(Op.SpecConstantOp, 3, id, opcode, ...(args.length ? args : [4, 5]))];

// The constant stored by buildSpecOpStore, e.g. "30i", or null.
function storedValue(wgsl) {
  const m = wgsl.match(/\.\w+ = ([^;]+);/);
  return m ? m[1] : null;
}

// Build a compute shader with a storage buffer at the given binding, optionally written.
function buildComputeWithStorageBuffer(hasStore = false, binding = 0) {
  // IDs: 1=void, 2=fn_type, 3=uint, 4=struct, 5=ptr_sb, 6=sb_var,
  //      7=main, 8=label, 9=const0, 10=ptr_uint, 11=ac, 12=loaded
  return buildSpirv(13, [
    ...COMPUTE_PREAMBLE(7),
    ...encodeInst(Op.Decorate, 4, Deco.Block),
    ...encodeInst(Op.MemberDecorate, 4, 0, Deco.Offset, 0),
    ...encodeInst(Op.Decorate, 6, Deco.DescriptorSet, 0),
    ...encodeInst(Op.Decorate, 6, Deco.Binding, binding),
    ...encodeInst(Op.TypeVoid, 1),
    ...encodeInst(Op.TypeFunction, 2, 1),
    ...encodeInst(Op.TypeInt, 3, 32, 0),
    ...encodeInst(Op.TypeStruct, 4, 3),
    ...encodeInst(Op.TypePointer, 5, SC.StorageBuffer, 4),
    ...encodeInst(Op.TypePointer, 10, SC.StorageBuffer, 3),
    ...encodeInst(Op.Variable, 5, 6, SC.StorageBuffer),
    ...encodeInst(Op.Constant, 3, 9, 0),
    ...encodeInst(Op.Function, 1, 7, 0, 2),
    ...encodeInst(Op.Label, 8),
    ...encodeInst(Op.AccessChain, 10, 11, 6, 9),
    ...encodeInst(Op.Load, 3, 12, 11),
    ...(hasStore ? encodeInst(Op.Store, 11, 12) : []),
    ...encodeInst(Op.Return),
    ...encodeInst(Op.FunctionEnd),
  ]);
}

const FIXTURES = readdirSync(FIXTURES_DIR).filter((f) => f.endsWith(".spv")).sort();

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: freeze_and_inline — specialization constants
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 1: specialization constants ===");

{
  // 1a. Scalar IDs survive. Unsupported composite/select expressions use the frozen fallback.
  for (const name of ["per_stage_overrides_frag.spv"]) {
    assert(fixtureWgsl(name).includes("@id("), `${name}: scalar overrides retained`);
  }
  for (const name of ["spec_constants.spv", "many_overrides.spv", "per_stage_overrides_vert.spv"]) {
    assert(!fixtureWgsl(name).includes("@id("), `${name}: unsupported composite/select expressions frozen`);
  }
  assert(fixtureWgsl("per_stage_overrides_vert.spv").includes("override v : f32 = select(1.0f, 2.0f, false);"),
    "per_stage_overrides_vert: VERTEX_FLAGS & 1 folded to false");
  for (const name of ["many_overrides.spv"]) {
    assert(!fixtureWgsl(name).includes("override"), `${name}: spec constants folded`);
  }
}

{
  // 1b. Chained scalar expressions remain overridable without translating again.
  const wgsl = fixtureWgsl("chained_spec_ops.spv");
  assert(wgsl.includes("@id(0)") && wgsl.includes("@id(1)") && wgsl.includes("@id(2)"),
    "chained_spec_ops: all three scalar IDs retained");
  assert(wgsl.includes("(SPEC_A + SPEC_B)"), "chained_spec_ops: derived expression retained");
  const computed = constructedWgsl(buildSpecOpStore({ specializable: true, values: specOp(SpecOp.IAdd) }), "scalar override expression");
  assert(computed.includes("@id(0)") && computed.includes("@id(1)"), "both expression operands remain overridable");
  assert(/override \w+ : i32 = \(\w+ \+ \w+\);/.test(computed), "derived integer override remains an addition");
}

{
  // 1c. OpSpecConstantOp with ordinary constant operands folds with 32-bit two's-complement semantics.
  const cases = [
    ["IAdd", {}, specOp(SpecOp.IAdd), "30i"],
    ["ISub", {}, specOp(SpecOp.ISub), "-10i"],
    ["IMul", {}, specOp(SpecOp.IMul), "200i"],
    ["IAdd overflow wraps", { a: 0x7fffffff, b: 1 }, specOp(SpecOp.IAdd), "i32(-2147483648)"],
    ["SNegate", {}, specOp(SpecOp.SNegate, 4), "-10i"],
    ["SDiv", { a: -21, b: 4 }, specOp(SpecOp.SDiv), "-5i"],
    ["SDiv by zero", { a: 42, b: 0 }, specOp(SpecOp.SDiv), "0i"],
    ["SRem", { a: -7, b: 3 }, specOp(SpecOp.SRem), "-1i"],
    ["SMod", { a: -7, b: 3 }, specOp(SpecOp.SMod), "2i"],
    ["UDiv", { signed: false }, specOp(SpecOp.UDiv, 5, 4), "2u"],
    ["UMod", { signed: false, a: 20, b: 7 }, specOp(SpecOp.UMod), "6u"],
    ["BitwiseOr", {}, specOp(SpecOp.BitwiseOr), "30i"],
    ["BitwiseAnd", {}, specOp(SpecOp.BitwiseAnd), "0i"],
    ["BitwiseXor", {}, specOp(SpecOp.BitwiseXor), "30i"],
    ["Not", {}, specOp(SpecOp.Not, 4), "-11i"],
    ["ShiftLeftLogical", { b: 2 }, specOp(SpecOp.ShiftLeftLogical), "40i"],
    ["ShiftRightLogical", { signed: false, a: 20, b: 2 }, specOp(SpecOp.ShiftRightLogical), "5u"],
    ["SLessThan + Select", {}, (id) => [
      encodeInst(Op.SpecConstantOp, 14, id, Op.SLessThan, 4, 5),
      encodeInst(Op.SpecConstantOp, 3, id + 1, Op.Select, id, 5, 4),
    ], "20i"],
    ["IEqual + Select", { b: 10 }, (id) => [
      encodeInst(Op.SpecConstantOp, 14, id, Op.IEqual, 4, 5),
      encodeInst(Op.SpecConstantOp, 3, id + 1, Op.Select, id, 4, 10),
    ], "10i"],
    ["INotEqual + Select", { b: 10 }, (id) => [
      encodeInst(Op.SpecConstantOp, 14, id, Op.INotEqual, 4, 5),
      encodeInst(Op.SpecConstantOp, 3, id + 1, Op.Select, id, 4, 10),
    ], "0i"],
    ["Composite + CompositeExtract", {}, (id) => [
      encodeInst(Op.SpecConstantComposite, 16, id, 4, 5),
      encodeInst(Op.SpecConstantOp, 3, id + 1, Op.CompositeExtract, id, 1),
    ], "20i"],
  ];
  for (const [label, options, values, expected] of cases) {
    const wgsl = constructedWgsl(buildSpecOpStore({ ...options, values }), `OpSpecConstantOp ${label}`);
    assertEq(storedValue(wgsl), expected, `OpSpecConstantOp ${label} folds to ${expected}`);
  }
  const composite = constructedWgsl(buildSpecOpStore({ specializable: true, values: (id) => [
    encodeInst(Op.SpecConstantComposite, 16, id, 4, 5),
    encodeInst(Op.SpecConstantOp, 3, id + 1, Op.CompositeExtract, id, 1),
  ] }), "specialized composite fallback");
  assertEq(storedValue(composite), "20i", "unsupported composite overrides use their patched defaults");
  assert(!composite.includes("@id("), "composite fallback retains no external scalar IDs");
}

{
  // 1d. OpSpecConstantTrue / OpSpecConstantFalse become constants selecting the stored value.
  const wgsl = constructedWgsl(buildSpecOpStore({ values: (id) => [
    encodeInst(Op.SpecConstantTrue, 14, id),
    encodeInst(Op.SpecConstantOp, 3, id + 1, Op.Select, id, 4, 5),
  ] }), "OpSpecConstantTrue");
  assertEq(storedValue(wgsl), "10i", "OpSpecConstantTrue selects the first value");
  assert(!wgsl.includes("override"), "spec bool constants folded");
}

{
  // DXC corrupts row-major matrices in struct inout arguments emitted for ordinary GLSL helpers.
  const wgsl = fixtureWgsl("camera_struct.spv");
  assert(wgsl.includes("scene.current") && wgsl.includes("scene.previous"),
    "camera_struct: current and previous-frame cameras remain distinct");
  assertEq([...wgsl.matchAll(/\bfn\s+/g)].length, 1,
    "camera_struct: no camera helper argument boundary reaches DXC");
  assert(!wgsl.includes("ptr<function,"), "camera_struct: no struct inout arguments remain");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: rewrite_copy_logical and malformed input
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 2: rewrite_copy_logical ===");

{
  // 2a. Shader without OpCopyLogical converts fine (no-op pass).
  constructedWgsl(buildMinimalComputeShader(), "minimal compute (no OpCopyLogical)");
}

{
  // 2b. Input too small (< 20 bytes) returns error gracefully.
  const tiny = Buffer.from([0x03, 0x02, 0x23, 0x07, 0x00]);
  assert(convertToWgsl(tiny).wgsl === null, "input < 20 bytes returns error");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: rewrite_terminate_invocation
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 3: rewrite_terminate_invocation ===");

{
  // 3a. OpTerminateInvocation is rewritten to OpKill → WGSL "discard".
  const wgsl = constructedWgsl(buildFragmentWithTerminateInvocation(), "fragment with OpTerminateInvocation");
  assert(wgsl.includes("discard"), "OpTerminateInvocation → 'discard' in WGSL");
  assert(wgsl.includes("@fragment"), "fragment entry point preserved");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: convert_push_constants_to_storage_buffers
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 4: convert_push_constants_to_storage_buffers ===");

{
  // 4a. Push constants become a read-only storage buffer at group 3, binding 999.
  const wgsl = constructedWgsl(buildComputeWithPushConstants(), "compute with push constants");
  assert(/@group\(3u?\) @binding\(999u?\) var<storage, read> /.test(wgsl), "push constants → @group(3) @binding(999) var<storage, read>");
}

{
  // 4b. Shader without push constants is unaffected.
  const wgsl = constructedWgsl(buildMinimalComputeShader(), "compute without push constants");
  assert(!wgsl.includes("@binding(999"), "no push constants → no binding 999");
}

{
  // 4c. Fixture: basic_vertex.spv has push constants.
  const wgsl = fixtureWgsl("basic_vertex.spv");
  assert(/@group\(3u?\) @binding\(999u?\) var<storage, read> pc/.test(wgsl), "basic_vertex push constants at (3, 999), read-only");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: split_combined_samplers
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 5: split_combined_samplers ===");

{
  // 5a. Binding N becomes sampler 2N and image 2N+1; other bindings double; push constants stay at 999.
  assertEq(bindingsOf(fixtureWgsl("basic_fragment.spv")), ["0,0", "1,0", "1,1", "1,2", "1,3", "3,999"],
    "basic_fragment bindings after the split");
  assertEq(bindingsOf(fixtureWgsl("multi_texture.spv")),
    ["0,0", "0,1", "0,10", "0,11", "0,2", "0,3", "0,4", "0,5", "0,6", "0,7", "0,8", "0,9", "1,0", "3,999"],
    "multi_texture bindings after the split");
}

{
  // 5b. Storage buffers double too and keep their access mode.
  assertEq(bindingsOf(fixtureWgsl("readonly_ssbo.spv")), ["0,0", "0,2", "3,999"], "readonly_ssbo bindings doubled");
}

{
  // 5c. depth_sampling has combined image+sampler with depth texture.
  const wgsl = fixtureWgsl("depth_sampling.spv");
  assert(wgsl.includes("texture_depth_2d") && wgsl.includes("sampler_comparison"), "depth_sampling: depth texture + comparison sampler");
}

{
  // 5d. A helper called with two combined samplers keeps each call's texture (opaque calls are inlined first).
  const wgsl = fixtureWgsl("sampler_params.spv");
  assert(/textureSample\(first_tex, /.test(wgsl) && /textureSample\(second_tex, /.test(wgsl), "sampler_params: both textures sampled");
  assertEq(bindingsOf(wgsl), ["0,0", "0,1", "0,2", "0,3"], "sampler_params bindings after the split");
}

{
  // 5e. Bindings that would double past the push-constant binding are rejected.
  const r = convertToWgsl(buildComputeWithStorageBuffer(false, 499));
  assert(r.stderr.includes("Binding too high"), "binding 499 reports an error");
  const ok = convertToWgsl(buildComputeWithStorageBuffer(false, 498));
  assert(ok.wgsl !== null && ok.wgsl.includes("@binding(996") && !ok.stderr.includes("ERROR"), "binding 498 doubles to 996");
}

{
  // Inlining the coordinate helper puts sampling in a different block from the combined-sampler load.
  const wgsl = fixtureWgsl("sampler_control_flow.spv");
  assert(wgsl.includes("textureSampleLevel(first_tex,") && wgsl.includes("textureSampleLevel(second_tex,"),
    "sampler_control_flow: 2D and array sampling survive inlined control flow");
  assertEq(bindingsOf(wgsl), ["0,0", "0,1", "0,2", "0,3", "0,4"],
    "sampler_control_flow: both sampler/image pairs and the output buffer retain distinct bindings");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: fix_depth2_images
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 6: fix_depth2_images ===");

{
  // 6a. depth_sampling.spv has depth images sampled with textureSample.
  const wgsl = fixtureWgsl("depth_sampling.spv");
  assert(wgsl.includes("texture_depth_2d"), "depth image becomes texture_depth_2d");
  assert(wgsl.includes("textureSample"), "depth sampling uses textureSample");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: negate_position_y
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 7: negate_position_y ===");

const NEGATED_Y = /\.y = -\(/;

{
  // 7a. Vertex shader fixture negates gl_Position.y (struct member Position).
  const wgsl = fixtureWgsl("basic_vertex.spv");
  assert(wgsl.includes("@builtin(position)"), "position builtin present");
  assert(NEGATED_Y.test(wgsl), "Y-negation present in vertex output");
}

{
  // 7b. Constructed vertex shader with a decorated Position variable gets Y-negation.
  const wgsl = constructedWgsl(buildMinimalVertexShader(), "constructed vertex shader");
  assert(NEGATED_Y.test(wgsl), "constructed vertex shader is Y-negated");
}

{
  // 7c. Fragment and compute shaders are NOT Y-negated.
  assert(!NEGATED_Y.test(fixtureWgsl("basic_fragment.spv")), "fragment shader NOT Y-negated");
  assert(!NEGATED_Y.test(fixtureWgsl("compute_particles.spv")), "compute shader NOT Y-negated");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: strip_restrict_decoration
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 8: strip_restrict_decoration ===");

{
  // 8a. readonly_ssbo declares a restrict buffer.
  assert(fixtureWgsl("readonly_ssbo.spv").includes("var<storage, read> instances"), "restrict readonly buffer converts");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: strip_memory_barrier
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 9: strip_memory_barrier ===");

{
  // 9a. memoryBarrierShared() (OpMemoryBarrier) is removed; barrier() (OpControlBarrier) remains.
  const wgsl = fixtureWgsl("memory_barrier.spv");
  assert(wgsl.includes("workgroupBarrier()"), "control barrier becomes workgroupBarrier()");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: fix_nonfinite_literals
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 10: fix_nonfinite_literals ===");

{
  // 10a. Infinity and NaN constants are replaced with +/-FLT_MAX.
  // IDs: 3=float, 4..6 non-finite constants, 7=main, 9=vec3, 10=Output ptr, 11=Output var, 12=composite.
  const spv = buildSpirv(13, [
    ...encodeInst(Op.Capability, 1),
    ...encodeInst(Op.MemoryModel, 0, 1),
    ...encodeEntryPoint(ExecModel.Fragment, 7, "main", 11),
    ...encodeInst(Op.ExecutionMode, 7, 7), // OriginUpperLeft
    ...encodeInst(Op.Decorate, 11, 30, 0), // Location 0
    ...encodeInst(Op.TypeVoid, 1),
    ...encodeInst(Op.TypeFunction, 2, 1),
    ...encodeInst(Op.TypeFloat, 3, 32),
    ...encodeInst(Op.TypeVector, 9, 3, 3),
    ...encodeInst(Op.TypePointer, 10, SC.Output, 9),
    ...encodeInst(Op.Variable, 10, 11, SC.Output),
    ...encodeInst(Op.Constant, 3, 4, 0x7F800000), // +Infinity
    ...encodeInst(Op.Constant, 3, 5, 0xFF800000), // -Infinity
    ...encodeInst(Op.Constant, 3, 6, 0x7FC00000), // NaN
    ...encodeInst(Op.Function, 1, 7, 0, 2),
    ...encodeInst(Op.Label, 8),
    ...encodeInst(Op.CompositeConstruct, 9, 12, 4, 5, 6),
    ...encodeInst(Op.Store, 11, 12),
    ...encodeInst(Op.Return),
    ...encodeInst(Op.FunctionEnd),
  ]);
  const wgsl = constructedWgsl(spv, "shader with inf/nan constants");
  assert(wgsl.includes("340282346638528859811704183484516925440.0f"), "non-finite constants become FLT_MAX");
  assert(wgsl.includes("-340282346638528859811704183484516925440.0f"), "-Infinity becomes -FLT_MAX");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: flatten_binding_arrays
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 11: flatten_binding_arrays ===");

{
  // 11a. A dynamic texture index selects one of four separately bound textures.
  const r = convertFixture("texture_array.spv");
  assert(r.wgsl !== null, `texture_array converts: ${r.error}`);
  const wgsl = r.wgsl ?? "";
  assert(!wgsl.includes("binding_array"), "no unsupported binding array remains");
  assertEq(bindingsOf(wgsl), ["0,0", "0,2", "0,4", "0,6", "0,8", "3,999"], "all four textures and the relocated sampler have distinct bindings");
  assert(wgsl.includes("switch(pc.layer)") && wgsl.includes("case 3u:"), "dynamic index selects every array element");
  assert(!r.stderr.includes("WARNING"), "dynamic indexing does not alias or warn");
}

{
  // 11b. A constant-indexed combined-sampler array keeps the selected element and a separate following binding.
  const r = convertFixture("sampler_array.spv");
  assert(r.wgsl !== null, `sampler_array converts: ${r.error}`);
  assertEq(bindingsOf(r.wgsl ?? ""), ["0,0", "0,1", "0,4", "0,5"], "sampler_array reserves both elements without colliding with the following binding");
  assert(!r.stderr.includes("WARNING"), "constant index 0 does not warn");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: End-to-End pipeline (all fixtures)
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 12: End-to-End fixture conversion ===");

for (const fix of FIXTURES) {
  const wgsl = fixtureWgsl(fix);
  const hasEntry = wgsl.includes("@vertex") || wgsl.includes("@fragment") || wgsl.includes("@compute");
  assert(hasEntry, `${fix} has an entry point`);
  assert(!wgsl.includes("var<push_constant>"), `${fix}: push constants converted`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13: Edge cases
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 13: Edge cases ===");

{
  // 13a. Empty input returns error.
  assert(convertToWgsl(Buffer.alloc(0)).wgsl === null, "empty input returns error");
}

{
  // 13b. Invalid magic returns error.
  const bad = Buffer.alloc(20);
  bad.writeUInt32LE(0xDEADBEEF, 0);
  assert(convertToWgsl(bad).wgsl === null, "invalid SPIR-V magic returns error");
}

{
  // 13c. Header-only input (5 words, no instructions) returns error.
  assert(convertToWgsl(buildSpirv(1, [])).wgsl === null, "header-only SPIR-V returns error (no entry point)");
}

{
  // 13d. Not aligned to 4 bytes returns error.
  const misaligned = Buffer.alloc(23);
  misaligned.writeUInt32LE(SPIRV_MAGIC, 0);
  assert(convertToWgsl(misaligned).wgsl === null, "misaligned input returns error");
}

{
  // 13e. Truncated instruction stream is reported and left unchanged.
  const truncated = buildMinimalComputeShader().subarray(0, 44); // Cuts OpEntryPoint short.
  const r = convertToWgsl(truncated);
  assert(r.wgsl === null && r.stderr.includes("Malformed SPIR-V"), "truncated module reports malformed SPIR-V");
}

{
  // 13f. Conversion is deterministic.
  assertEq(fixtureWgsl("multi_texture.spv"), fixtureWgsl("multi_texture.spv"), "conversion is deterministic (same output twice)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14: infer_readonly_storage
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 14: infer_readonly_storage ===");

{
  // 14a. Constructed read-only storage buffer → var<storage, read>.
  const wgsl = constructedWgsl(buildComputeWithStorageBuffer(false), "constructed read-only storage buffer");
  assert(wgsl.includes("var<storage, read>") && !wgsl.includes("read_write"), "read-only inferred as var<storage, read>");
}

{
  // 14b. Constructed read-write storage buffer → stays var<storage, read_write>.
  const wgsl = constructedWgsl(buildComputeWithStorageBuffer(true), "constructed read-write storage buffer");
  assert(wgsl.includes("var<storage, read_write>"), "written buffer is var<storage, read_write>");
}

{
  // 14c. readonly_ssbo: the instances buffer is read-only, output_buf is written.
  const wgsl = fixtureWgsl("readonly_ssbo.spv");
  assert(wgsl.includes("var<storage, read> instances"), "readonly_ssbo: instances is read-only");
  assert(wgsl.includes("var<storage, read_write> output_buf"), "readonly_ssbo: output_buf is read_write");
}

{
  // 14d. compute_particles: particle_buf is written.
  assert(fixtureWgsl("compute_particles.spv").includes("var<storage, read_write> particle_buf"), "compute_particles has read_write buffer");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15: WGSL output structure
// ─────────────────────────────────────────────────────────────────────────────
console.log("\n=== Test 15: WGSL output structure ===");

{
  const vertex = fixtureWgsl("basic_vertex.spv");
  assert(vertex.includes("@vertex") && vertex.includes("vec4<f32>"), "vertex WGSL has @vertex and vec4<f32>");
  const fragment = fixtureWgsl("basic_fragment.spv");
  assert(fragment.includes("@fragment") && fragment.includes("@location(0"), "fragment WGSL has location(0) output");
  assert(fragment.includes("diagnostic(off, derivative_uniformity)"), "Tint adds diagnostic directive");
  const compute = fixtureWgsl("compute_particles.spv");
  assert(compute.includes("@compute") && compute.includes("@workgroup_size"), "compute WGSL has workgroup_size");
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────
console.log(`\n${"=".repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);

if (failed > 0) {
  process.exit(1);
}
