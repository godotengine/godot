import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createInterface } from 'node:readline';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { parseRequest, ProtocolFault, type RunStartRequest } from '../src/protocol.ts';
import { parseSse, ProviderTurnError, TurnAccumulator, type ProviderEvent, type TurnResult } from '../src/provider_events.ts';
import { fakeResponse, openAIResponse, openAIToolDefinitions, type ProviderRequest } from '../src/providers.ts';
import { RunManager } from '../src/run_manager.ts';
import { validateToolCall } from '../src/tool_schema.ts';

const limits = { max_attempts: 4, max_tool_calls: 4, max_output_tokens: 256, request_timeout_ms: 2000, deadline_ms: 10000 };
const context = JSON.stringify({ scene_ref: 'scene-雪', base_revision: 'revision-1', parent_ref: 'node-root', root_class: 'Node2D' });
const start = (provider: 'fake' | 'openai_responses' = 'fake', intent: 'discuss' | 'propose' | 'execute' = 'propose'): RunStartRequest =>
  ({ protocol_version: '1.1', request_id: 'req-1', method: 'run_start', params: { run_id: 'run-1', provider, model: provider === 'fake' ? null : 'explicit-model', intent, prompt: 'Add a marker', context, limits, live_authorized: provider !== 'fake' } });
const toolArgs = JSON.stringify({ scene_ref: 'scene-雪', base_revision: 'revision-1', operations: [{ op: 'create_child', parent_ref: 'node-root', class_name: 'Node2D', name: 'AI_Marker', properties: { position: { type: 'Vector2', value: [48, 24] } } }] });
const item = (name = 'scene_patch_preview', args = toolArgs, id = 'item-1', callId = 'call-1') => ({ type: 'function_call', id, call_id: callId, name, arguments: args });
const success = (items: object[], usage: object | null = null) => ({ type: 'response.completed', response: { id: 'resp-1', status: 'completed', output: items, usage } });

type WireFrame = { request_id: string; event?: string; status?: string; data?: { call_id?: string; state?: string } };
async function captureServiceFollowup(followup: 'run_cancel' | 'run_continue'): Promise<WireFrame[]> {
  const entry = fileURLToPath(new URL('../src/main.ts', import.meta.url));
  const child = spawn(process.execPath, ['--disable-warning=ExperimentalWarning', '--experimental-strip-types', entry],
    { stdio: ['pipe', 'pipe', 'pipe'] });
  const frames: WireFrame[] = [];
  let stderr = '';
  child.stderr.setEncoding('utf8').on('data', (chunk: string) => { stderr += chunk; });
  const reader = createInterface({ input: child.stdout });
  const listeners: Array<() => void> = [];
  reader.on('line', line => {
    frames.push(JSON.parse(line) as WireFrame);
    for (const listener of [...listeners]) listener();
  });
  const until = (predicate: (frame: WireFrame) => boolean): Promise<WireFrame> => new Promise((resolve, reject) => {
    const existing = frames.find(predicate);
    if (existing) { resolve(existing); return; }
    const timeout = setTimeout(() => { listeners.splice(listeners.indexOf(check), 1); reject(new Error('Timed out waiting for service frame.')); }, 3000);
    const check = (): void => {
      const match = frames.find(predicate);
      if (!match) return;
      clearTimeout(timeout);
      listeners.splice(listeners.indexOf(check), 1);
      resolve(match);
    };
    listeners.push(check);
  });
  try {
    child.stdin.write(`${JSON.stringify(start())}\n`);
    const ready = await until(frame => frame.event === 'tool_call_ready');
    const params = followup === 'run_cancel' ? { run_id: 'run-1' } :
      { run_id: 'run-1', call_id: ready.data?.call_id, tool_result: { status: 'unresolved', result: {} } };
    child.stdin.write(`${JSON.stringify({ protocol_version: '1.1', request_id: 'followup-1', method: followup, params })}\n`);
    await until(frame => frame.request_id === 'followup-1' && frame.event === 'run_state' &&
      frame.data?.state === (followup === 'run_cancel' ? 'cancelled' : 'reconciliation_required'));
    child.stdin.end();
    const code = await new Promise<number | null>(resolve => child.on('exit', resolve));
    assert.equal(code, 0, stderr);
    return frames;
  } finally {
    child.kill();
    reader.close();
  }
}

test('v1.1 run envelope requires explicit live authorization, model, finite limits, and exact fields', () => {
  assert.equal(parseRequest(JSON.stringify(start())).method, 'run_start');
  assert.equal(parseRequest(JSON.stringify({ protocol_version: '1.1', request_id: 'status-1', method: 'provider_status', params: {} })).method, 'provider_status');
  const bad = { ...start('openai_responses'), params: { ...start('openai_responses').params, model: null } };
  assert.throws(() => parseRequest(JSON.stringify(bad)), (error: unknown) => error instanceof ProtocolFault && error.code === 'LIVE_AUTHORIZATION_REQUIRED');
  const untrusted = { ...start(), params: { ...start().params, approved: true } };
  assert.throws(() => parseRequest(JSON.stringify(untrusted)), (error: unknown) => error instanceof ProtocolFault && error.code === 'UNSUPPORTED_SCHEMA');
  const unlimited = { ...start(), params: { ...start().params, limits: { ...limits, max_attempts: 1000 } } };
  assert.throws(() => parseRequest(JSON.stringify(unlimited)), (error: unknown) => error instanceof ProtocolFault && error.code === 'INVALID_ARGUMENT');
});

test('SSE parser preserves split Unicode and accumulator releases exactly one call after completed barrier', async () => {
  const current = item();
  const raw = [
    { type: 'response.output_text.delta', delta: '雪' },
    { type: 'response.output_item.added', item: current },
    { type: 'response.function_call_arguments.delta', item_id: current.id, delta: toolArgs.slice(0, 18) },
    { type: 'response.function_call_arguments.delta', item_id: current.id, delta: toolArgs.slice(18) },
    { type: 'response.function_call_arguments.done', item_id: current.id, arguments: toolArgs },
    { type: 'response.output_item.done', item: current }, success([current]),
  ].map(value => `data: ${JSON.stringify(value)}\n\n`).join('');
  const bytes = Buffer.from(raw);
  const unicode = bytes.indexOf(Buffer.from('雪'));
  async function* split(): AsyncGenerator<Uint8Array> { yield bytes.subarray(0, unicode + 1); yield bytes.subarray(unicode + 1, unicode + 2); yield bytes.subarray(unicode + 2); }
  const events: ProviderEvent[] = [];
  const acc = new TurnAccumulator(event => events.push(event));
  for await (const event of parseSse(split())) acc.feed(event);
  acc.finish();
  assert.equal(events[0].kind, 'text_delta');
  assert.equal(events.filter(event => event.kind === 'turn_completed').length, 1);
  const done = events.find(event => event.kind === 'turn_completed');
  assert.equal(done?.kind === 'turn_completed' ? done.result.call?.call_id : null, 'call-1');
  assert.deepEqual(done?.kind === 'turn_completed' ? done.result.usage : null, { input_tokens: null, output_tokens: null });
});

test('arguments.done followed by failure, incomplete output, or transport close never releases a tool', () => {
  for (const terminal of [{ type: 'response.failed' }, { type: 'response.incomplete' }, null]) {
    const events: ProviderEvent[] = [];
    const acc = new TurnAccumulator(event => events.push(event));
    const current = item();
    acc.feed({ type: 'response.output_item.added', item: current });
    acc.feed({ type: 'response.function_call_arguments.done', item_id: current.id, arguments: toolArgs });
    if (terminal) assert.throws(() => acc.feed(terminal), ProviderTurnError);
    else assert.throws(() => acc.finish(), ProviderTurnError);
    assert.equal(events.some(event => event.kind === 'turn_completed'), false);
  }
});

test('malformed, authority-bearing, unknown, and multi-call output are rejected before dispatch', () => {
  for (const [name, args] of [['unknown_tool', '{}'], ['scene_patch_preview', '{bad'], ['scene_patch_preview', toolArgs.replace('"scene_ref":', '"approved":true,"scene_ref":')]]) {
    assert.throws(() => validateToolCall(name, args));
  }
  const events: ProviderEvent[] = [];
  const acc = new TurnAccumulator(event => events.push(event));
  const first = item();
  const second = item('project_inspect', '{}', 'item-2', 'call-2');
  acc.feed({ type: 'response.output_item.added', item: first });
  assert.throws(() => acc.feed({ type: 'response.output_item.added', item: second }), ProviderTurnError);
  assert.equal(events.length, 0);
});

test('fake provider enters the same completed-turn and tool validator path', async () => {
  const events: ProviderEvent[] = [];
  const params: ProviderRequest = { ...start().params, max_output_tokens: 256, request_timeout_ms: 1000, signal: new AbortController().signal };
  const result = await fakeResponse(params, event => events.push(event));
  assert.equal(result.call?.tool_name, 'scene_patch_preview');
  assert.equal(events.filter(event => event.kind === 'turn_completed').length, 1);
  assert.equal(events.filter(event => event.kind === 'usage_update').length, 1);
  const discuss = await fakeResponse({ ...params, intent: 'discuss' }, () => {});
  assert.equal(discuss.call, null);
});

test('JSONL fake run is offline, emits one ready call, and does not expose sentinel environment data', () => {
  const entry = new URL('../src/main.ts', import.meta.url);
  const sentinel = 'SENTINEL_CREDENTIAL_NOT_FOR_OUTPUT';
  const child = spawnSync(process.execPath, ['--disable-warning=ExperimentalWarning', '--experimental-strip-types', fileURLToPath(entry)], {
    input: `${JSON.stringify(start())}\n`, encoding: 'utf8', timeout: 3000,
    env: { ...process.env, SLIME_AI_OPENAI_API_KEY: sentinel },
  });
  assert.equal(child.status, 0, child.stderr);
  assert.equal(child.stdout.includes(sentinel) || child.stderr.includes(sentinel), false);
  const frames = child.stdout.trim().split('\n').map(line => JSON.parse(line) as { event?: string; data?: { tool_name?: string }; status?: string });
  assert.equal(frames[0].status, 'ok');
  assert.equal(frames.filter(frame => frame.event === 'tool_call_ready').length, 1);
  assert.equal(frames.find(frame => frame.event === 'tool_call_ready')?.data?.tool_name, 'scene_patch_preview');
});

test('JSONL run_cancel acknowledgement precedes its cancelled event', async () => {
  const frames = await captureServiceFollowup('run_cancel');
  const ack = frames.findIndex(frame => frame.request_id === 'followup-1' && frame.status === 'ok');
  const event = frames.findIndex(frame => frame.request_id === 'followup-1' && frame.event === 'run_state' && frame.data?.state === 'cancelled');
  assert.ok(ack >= 0 && event > ack, JSON.stringify(frames));
});

test('JSONL unresolved continuation acknowledgement precedes reconciliation event', async () => {
  const frames = await captureServiceFollowup('run_continue');
  const ack = frames.findIndex(frame => frame.request_id === 'followup-1' && frame.status === 'ok');
  const event = frames.findIndex(frame => frame.request_id === 'followup-1' && frame.event === 'run_state' && frame.data?.state === 'reconciliation_required');
  assert.ok(ack >= 0 && event > ack, JSON.stringify(frames));
});

test('OpenAI adapter uses fixed endpoint, strict serial tools, call_id continuation, and redacted auth failures', async () => {
  const params: ProviderRequest = { ...start('openai_responses').params, max_output_tokens: 256, request_timeout_ms: 1000, previous_response_id: 'resp-prev', call_id: 'call-prev', tool_result: '{"status":"ok"}', signal: new AbortController().signal };
  let url = '';
  let body: Record<string, unknown> = {};
  const fakeFetch: typeof fetch = async (input, init) => { url = String(input); body = JSON.parse(String(init?.body)) as Record<string, unknown>; return new Response('', { status: 401 }); };
  const sentinel = 'sk-sentinel-do-not-log';
  await assert.rejects(() => openAIResponse(params, () => {}, fakeFetch, async () => sentinel), (error: unknown) => error instanceof ProviderTurnError && error.code === 'PROVIDER_AUTH_FAILED' && !error.message.includes(sentinel));
  assert.equal(url, 'https://api.openai.com/v1/responses');
  assert.equal(body.previous_response_id, 'resp-prev');
  assert.equal(body.parallel_tool_calls, false);
  assert.equal(body.store, true);
  assert.deepEqual(body.input, [{ type: 'function_call_output', call_id: 'call-prev', output: '{"status":"ok"}' }]);
  assert.equal(JSON.stringify(body).includes(sentinel), false);
  assert.ok((body.tools as object[]).every(tool => (tool as { strict?: boolean }).strict === true));
  assert.equal(openAIToolDefinitions('discuss').some(tool => (tool as { name: string }).name === 'scene_patch_preview'), false);
});

test('sent strict tool schemas omit model-dependent array bounds while local validation enforces one operation', () => {
  const definitions = openAIToolDefinitions('execute');
  const serialized = JSON.stringify(definitions);
  assert.equal(serialized.includes('minItems'), false);
  assert.equal(serialized.includes('maxItems'), false);
  const preview = definitions.find(tool => (tool as { name?: string }).name === 'scene_patch_preview');
  assert.ok(preview);
  assert.equal((preview as { strict?: boolean }).strict, true);
  const original = JSON.parse(toolArgs) as { operations: object[] };
  for (const operations of [[], [original.operations[0], original.operations[0]]]) {
    assert.throws(() => validateToolCall('scene_patch_preview', JSON.stringify({ ...original, operations })));
  }
  assert.equal(validateToolCall('scene_patch_preview', toolArgs).arguments.operations instanceof Array, true);
});

test('401, 403, and missing credentials fail without retry; 429 is marked retryable', async () => {
  const params: ProviderRequest = { ...start('openai_responses').params, max_output_tokens: 256, request_timeout_ms: 1000, signal: new AbortController().signal };
  await assert.rejects(() => openAIResponse(params, () => {}, async () => { throw new Error('should not fetch'); }, async () => null), (error: unknown) => error instanceof ProviderTurnError && error.code === 'CREDENTIAL_MISSING');
  for (const [status, code, retryable] of [[401, 'PROVIDER_AUTH_FAILED', false], [403, 'PROVIDER_AUTH_FAILED', false], [429, 'PROVIDER_RATE_LIMITED', true]] as const) {
    await assert.rejects(() => openAIResponse(params, () => {}, async () => new Response('', { status }), async () => 'sentinel'), (error: unknown) => error instanceof ProviderTurnError && error.code === code && error.retryable === retryable);
  }
});

test('cancellation during credential lookup prevents a later network request', async () => {
  const controller = new AbortController();
  const params: ProviderRequest = { ...start('openai_responses').params, max_output_tokens: 256, request_timeout_ms: 1000, signal: controller.signal };
  let fetches = 0;
  const credential = async (): Promise<string> => { controller.abort(); return 'sentinel'; };
  await assert.rejects(() => openAIResponse(params, () => {}, async () => { fetches++; return new Response('', { status: 200 }); }, credential),
    (error: unknown) => error instanceof ProviderTurnError && error.code === 'CANCELLED');
  assert.equal(fetches, 0);
});

test('OpenAI SSE adapter waits for completed response before returning one validated call', async () => {
  const current = item();
  const events = [
    { type: 'response.output_item.added', item: current },
    { type: 'response.function_call_arguments.delta', item_id: current.id, delta: toolArgs },
    { type: 'response.function_call_arguments.done', item_id: current.id, arguments: toolArgs },
    { type: 'response.output_item.done', item: current },
    success([current], { input_tokens: 12, output_tokens: 7 }),
  ].map(value => `data: ${JSON.stringify(value)}\n\n`).join('');
  const params: ProviderRequest = { ...start('openai_responses').params, max_output_tokens: 256, request_timeout_ms: 1000, signal: new AbortController().signal };
  const observed: ProviderEvent[] = [];
  const result = await openAIResponse(params, event => observed.push(event), async () => new Response(events, { status: 200 }), async () => 'sentinel');
  assert.equal(result.call?.call_id, 'call-1');
  assert.deepEqual(result.usage, { input_tokens: 12, output_tokens: 7 });
  assert.equal(observed.filter(event => event.kind === 'turn_completed').length, 1);
  const failedStream = events.replace('"response.completed"', '"response.failed"');
  await assert.rejects(() => openAIResponse(params, () => {}, async () => new Response(failedStream, { status: 200 }), async () => 'sentinel'));
});

test('immediate cancellation and attempt limits prevent late or repeated dispatch', async () => {
  const output: object[] = [];
  let requests = 0;
  const manager = new RunManager(value => output.push(value), async () => { requests++; throw new ProviderTurnError('PROVIDER_RATE_LIMITED', 'rate limited', true); });
  const request = start();
  manager.handle(request);
  manager.handle({ protocol_version: '1.1', request_id: 'cancel-1', method: 'run_cancel', params: { run_id: request.params.run_id } });
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(requests, 0);
  assert.equal(output.filter(value => (value as { event?: string }).event === 'tool_call_ready').length, 0);

  const manager2 = new RunManager(value => output.push(value), async () => { requests++; throw new ProviderTurnError('PROVIDER_RATE_LIMITED', 'rate limited', true); });
  manager2.handle({ ...start(), params: { ...start().params, run_id: 'run-limited', limits: { ...limits, max_attempts: 2 } } });
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(requests, 2);
  assert.equal(output.some(value => (value as { event?: string; data?: { code?: string } }).event === 'turn_failed' && (value as { data?: { code?: string } }).data?.code === 'PROVIDER_RATE_LIMITED'), true);
});

test('timeout and disconnect retries stay finite and never dispatch from failed turns', async () => {
  for (const code of ['PROVIDER_TIMEOUT', 'PROVIDER_DISCONNECTED']) {
    const output: object[] = [];
    let requests = 0;
    const manager = new RunManager(value => output.push(value), async () => { requests++; throw new ProviderTurnError(code, 'safe failure', true); });
    manager.handle({ ...start(), params: { ...start().params, run_id: `run-${code}` } });
    await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal(requests, 3);
    assert.equal(output.filter(value => (value as { event?: string }).event === 'tool_call_ready').length, 0);
    assert.equal(output.some(value => (value as { event?: string; data?: { code?: string } }).event === 'turn_failed' && (value as { data?: { code?: string } }).data?.code === code), true);
  }
});

test('run manager counts 429 retries and dispatches one validated call only', async () => {
  const output: object[] = [];
  let attempts = 0;
  const manager = new RunManager(value => output.push(value), async (request, onEvent) => {
    attempts++;
    if (attempts < 3) throw new ProviderTurnError('PROVIDER_RATE_LIMITED', 'Rate limited.', true);
    const turn = await fakeResponse(request, onEvent);
    return turn;
  });
  manager.handle(start());
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(attempts, 3);
  assert.equal(output.filter(value => (value as { event?: string }).event === 'tool_call_ready').length, 1);
  assert.equal(output.filter(value => (value as { event?: string }).event === 'run_state' && (value as { data?: { retrying?: boolean } }).data?.retrying).length, 2);
});

test('continuation is linked to pending call and duplicate delivery cannot cause another provider request', async () => {
  const output: object[] = [];
  const seen: ProviderRequest[] = [];
  const manager = new RunManager(value => output.push(value), async (request, onEvent) => { seen.push(request); return fakeResponse(request, onEvent); });
  manager.handle(start());
  await new Promise(resolve => setTimeout(resolve, 20));
  const tool = output.find(value => (value as { event?: string }).event === 'tool_call_ready') as { data: { call_id: string; provider_response_id: string } };
  const continuation = { protocol_version: '1.1', request_id: 'req-2', method: 'run_continue', params: { run_id: 'run-1', call_id: tool.data.call_id, tool_result: { status: 'ok', result: { status: 'preview' } } } } as const;
  assert.equal((manager.handle(continuation) as { status: string }).status, 'ok');
  assert.equal((manager.handle(continuation) as { result: { duplicate: boolean } }).result.duplicate, true);
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(seen.length, 2);
  assert.equal(seen[1].previous_response_id, tool.data.provider_response_id);
  assert.equal(seen[1].call_id, tool.data.call_id);
  assert.equal(JSON.parse(seen[1].tool_result ?? '{}').status, 'ok');
  assert.throws(() => manager.handle({ ...continuation, params: { ...continuation.params, tool_result: { status: 'ok', result: { status: 'different' } } } }), (error: unknown) => error instanceof ProtocolFault && error.code === 'REQUEST_ALREADY_RECORDED');
});

test('proposal repair attempts stop after two changed proposals', async () => {
  const output: object[] = [];
  let turns = 0;
  const manager = new RunManager(value => output.push(value), async () => {
    turns++;
    return { response_id: `response-${turns}`, call: { call_id: `call-${turns}`, tool_name: 'scene_patch_preview', arguments: {}, provider_response_id: `response-${turns}` } };
  });
  manager.handle(start());
  for (let i = 1; i <= 3; i++) {
    await new Promise(resolve => setTimeout(resolve, 5));
    manager.handle({ protocol_version: '1.1', request_id: `continue-${i}`, method: 'run_continue', params: { run_id: 'run-1', call_id: `call-${i}`, tool_result: { status: 'error', result: { code: 'REVISION_CONFLICT' } } } });
  }
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(turns, 3);
  assert.equal(output.some(value => (value as { event?: string; data?: { message?: string } }).event === 'turn_failed' && (value as { data?: { message?: string } }).data?.message === 'Proposal repair limit reached.'), true);
});

test('unresolved result halts continuation; cancel ignores late provider completion', async () => {
  const output: object[] = [];
  const manager = new RunManager(value => output.push(value), async (request, onEvent) => fakeResponse(request, onEvent));
  manager.handle(start());
  await new Promise(resolve => setTimeout(resolve, 20));
  const ready = output.find(value => (value as { event?: string }).event === 'tool_call_ready') as { data: { call_id: string } };
  manager.handle({ protocol_version: '1.1', request_id: 'req-2', method: 'run_continue', params: { run_id: 'run-1', call_id: ready.data.call_id, tool_result: { status: 'unresolved', result: {} } } });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(output.some(value => (value as { event?: string; data?: { state?: string } }).event === 'run_state' && (value as { data?: { state?: string } }).data?.state === 'reconciliation_required'), true);
  const late: object[] = [];
  let finish!: (value: TurnResult) => void;
  const blocked = new Promise<TurnResult>(resolve => { finish = resolve; });
  const manager2 = new RunManager(value => late.push(value), async () => blocked);
  manager2.handle({ ...start(), params: { ...start().params, run_id: 'run-2' } });
  await new Promise(resolve => setTimeout(resolve, 0));
  manager2.handle({ protocol_version: '1.1', request_id: 'req-c', method: 'run_cancel', params: { run_id: 'run-2' } });
  finish({ response_id: 'late', text: '', call: { call_id: 'late', tool_name: 'scene_patch_preview', arguments: {}, provider_response_id: 'late' }, usage: { input_tokens: null, output_tokens: null } });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(late.some(value => (value as { event?: string }).event === 'tool_call_ready'), false);
});
