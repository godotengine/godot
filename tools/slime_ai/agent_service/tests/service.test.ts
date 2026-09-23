import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { FrameDecoder, MAX_FRAME_BYTES } from '../src/frame_decoder.ts';
import { parseRequest, ProtocolFault } from '../src/protocol.ts';

const fixture = (name: string): string => readFileSync(new URL(`../../protocol/fixtures/${name}.json`, import.meta.url), 'utf8').trim();
const entry = new URL('../src/main.ts', import.meta.url);
const args = ['--disable-warning=ExperimentalWarning', '--experimental-strip-types', fileURLToPath(entry)];
const request = (method: string, params: object, requestId = 'test-1'): string =>
  JSON.stringify({ protocol_version: '1.0', request_id: requestId, method, params });
const proposal = (scenario: string, root_class = 'Node2D'): string => request('fake_propose_scene_patch',
  { scene_ref: 'scene-1', base_revision: 'revision-1', parent_ref: 'node-root', root_class, scenario });

function run(input: string): { status: number | null; lines: string[]; stderr: string; elapsedMs: number } {
  const start = Date.now();
  const child = spawnSync(process.execPath, args, { input, encoding: 'utf8', timeout: 3000 });
  if (child.error) throw child.error;
  return { status: child.status, lines: child.stdout.trimEnd().split('\n').filter(Boolean),
    stderr: child.stderr, elapsedMs: Date.now() - start };
}

function fault(source: string, code: string): void {
  assert.throws(() => parseRequest(source), (error: unknown) => error instanceof ProtocolFault && error.code === code);
}

test('golden hello matches exact wire response and EOF exits', () => {
  const result = run(`${fixture('hello_request')}\n`);
  assert.equal(result.status, 0);
  assert.deepEqual(JSON.parse(result.lines[0]), JSON.parse(fixture('hello_response')));
  assert.equal(result.lines.length, 1);
  assert.equal(result.stderr, '');
});

test('golden proposal matches exact wire response', () => {
  const result = run(`${fixture('proposal_request')}\n`);
  assert.equal(result.status, 0);
  assert.deepEqual(JSON.parse(result.lines[0]), JSON.parse(fixture('proposal_response')));
});

test('3D proposal has typed Vector3 value and echoed request ID', () => {
  const result = run(`${proposal('normal', 'Node3D')}\n`);
  const response = JSON.parse(result.lines[0]);
  assert.equal(response.request_id, 'test-1');
  assert.deepEqual(response.result.operations[0].properties.position, { type: 'Vector3', value: [48, 24, 0] });
});

test('frame decoder buffers split Unicode and JSON bytes', () => {
  const frame = Buffer.from(`${request('hello', { engine_revision: 'rev', workspace_id: '雪', client_capabilities: [] })}\n`);
  const split = frame.indexOf(Buffer.from('雪')) + 1;
  const decoder = new FrameDecoder();
  assert.deepEqual(decoder.push(frame.subarray(0, 10)), []);
  assert.deepEqual(decoder.push(frame.subarray(10, split)), []);
  assert.deepEqual(decoder.push(frame.subarray(split, split + 1)), []);
  const events = decoder.push(frame.subarray(split + 1));
  assert.equal(events.length, 1);
  assert.equal(events[0].kind, 'frame');
  if (events[0].kind === 'frame') {
    const decoded = parseRequest(events[0].text);
    assert.equal(decoded.method, 'hello');
    if (decoded.method === 'hello') assert.equal(decoded.params.workspace_id, '雪');
  }
});

test('process accepts fragmented Unicode request and exits on EOF', async () => {
  const child = spawn(process.execPath, args, { stdio: ['pipe', 'pipe', 'pipe'] });
  const frame = Buffer.from(`${request('hello', { engine_revision: 'rev', workspace_id: '雪', client_capabilities: [] })}\n`);
  const output: Buffer[] = [];
  child.stdout.on('data', (chunk: Buffer) => output.push(chunk));
  const split = frame.indexOf(Buffer.from('雪')) + 1;
  child.stdin.write(frame.subarray(0, split));
  await new Promise(resolve => setTimeout(resolve, 5));
  child.stdin.end(frame.subarray(split));
  const code = await new Promise<number | null>(resolve => child.on('exit', resolve));
  assert.equal(code, 0);
  const response = JSON.parse(Buffer.concat(output).toString('utf8'));
  assert.equal(response.request_id, 'test-1');
  assert.equal(response.status, 'ok');
});

test('decoder rejects invalid UTF-8', () => {
  const events = new FrameDecoder().push(Buffer.from([0xff, 0x0a]));
  assert.deepEqual(events, [{ kind: 'error', reason: 'Invalid UTF-8 frame' }]);
});

test('decoder rejects oversized frame and resumes at next newline', () => {
  const decoder = new FrameDecoder();
  assert.deepEqual(decoder.push(Buffer.alloc(MAX_FRAME_BYTES, 0x61)), []);
  const events = decoder.push(Buffer.from('\n{}\n'));
  assert.deepEqual(events, [{ kind: 'error', reason: 'Frame exceeds 1 MiB' }, { kind: 'frame', text: '{}' }]);
});

test('process returns explicit protocol error for malformed and oversized frames', () => {
  const result = run('{oops}\n' + 'a'.repeat(MAX_FRAME_BYTES) + '\n');
  assert.equal(result.status, 0);
  assert.equal(result.lines.length, 2);
  for (const line of result.lines) assert.equal(JSON.parse(line).error.code, 'PROVIDER_PROTOCOL_ERROR');
});

test('duplicate keys including escaped aliases are rejected', () => {
  fault('{"protocol_version":"1.0","request_id":"a","request_id":"b","method":"hello","params":{}}', 'PROVIDER_PROTOCOL_ERROR');
  fault('{"protocol_version":"1.0","request_id":"a","method":"hello","params":{"x":1,"\\u0078":2}}', 'PROVIDER_PROTOCOL_ERROR');
});

test('unsupported envelope/version and authority field fail closed', () => {
  fault(request('hello', { engine_revision: 'x', workspace_id: 'x', client_capabilities: [] }).replace('"method":', '"approved":true,"method":'), 'UNSUPPORTED_SCHEMA');
  fault(request('hello', { engine_revision: 'x', workspace_id: 'x', client_capabilities: [] }).replace('"1.0"', '"2.0"'), 'UNSUPPORTED_SCHEMA');
  fault(fixture('invalid_authority'), 'UNSUPPORTED_SCHEMA');
});

test('invalid method and arguments produce explicit errors with request ID echo', () => {
  const lines = [request('later_method', {}), proposal('unknown'), proposal('normal', 'Control')]
    .map(source => `${source}\n`).join('');
  const result = run(lines);
  assert.equal(result.status, 0);
  assert.deepEqual(result.lines.map(line => JSON.parse(line).error.code),
    ['UNKNOWN_TOOL', 'INVALID_ARGUMENT', 'INVALID_ARGUMENT']);
  for (const line of result.lines) assert.equal(JSON.parse(line).request_id, 'test-1');
});

test('delayed scenario waits before returning valid proposal', () => {
  const result = run(`${proposal('delayed')}\n`);
  assert.equal(result.status, 0);
  assert.ok(result.elapsedMs >= 75);
  assert.equal(JSON.parse(result.lines[0]).status, 'ok');
});

test('malformed scenario emits an intentionally invalid response frame', () => {
  const result = run(`${proposal('malformed')}\n`);
  assert.equal(result.status, 0);
  assert.equal(result.lines.length, 1);
  assert.throws(() => JSON.parse(result.lines[0]));
});

test('disconnect scenario exits without response', () => {
  const result = run(`${proposal('disconnect')}\n`);
  assert.equal(result.status, 0);
  assert.deepEqual(result.lines, []);
});

test('partial frame at EOF never executes', () => {
  const result = run(fixture('hello_request'));
  assert.equal(result.status, 0);
  assert.deepEqual(result.lines, []);
  assert.match(result.stderr, /Incomplete frame at EOF/);
});

test('nonfinite numeric JSON and excessive nesting are rejected', () => {
  fault('{"protocol_version":"1.0","request_id":"a","method":"hello","params":{"engine_revision":1e999,"workspace_id":"x","client_capabilities":[]}}', 'PROVIDER_PROTOCOL_ERROR');
  fault('['.repeat(66) + '0' + ']'.repeat(66), 'PROVIDER_PROTOCOL_ERROR');
});
