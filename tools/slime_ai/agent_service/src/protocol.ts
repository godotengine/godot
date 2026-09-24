import { parseStrictJson } from './strict_json.ts';

export type HelloRequest = {
  protocol_version: '1.0'; request_id: string; method: 'hello';
  params: { engine_revision: string; workspace_id: string; client_capabilities: string[] };
};
export type ProposalRequest = {
  protocol_version: '1.0'; request_id: string; method: 'fake_propose_scene_patch';
  params: { scene_ref: string; base_revision: string; parent_ref: string;
    root_class: 'Node2D' | 'Node3D'; scenario: 'normal' | 'malformed' | 'delayed' | 'disconnect' };
};
export type RunLimits = {
  max_attempts: number; max_tool_calls: number; max_output_tokens: number;
  request_timeout_ms: number; deadline_ms: number;
};
export type RunStartRequest = {
  protocol_version: '1.1'; request_id: string; method: 'run_start';
  params: { run_id: string; provider: 'fake' | 'openai_responses'; model: string | null;
    intent: 'discuss' | 'propose' | 'execute'; prompt: string; context: string;
    limits: RunLimits; live_authorized: boolean };
};
export type RunContinueRequest = {
  protocol_version: '1.1'; request_id: string; method: 'run_continue';
  params: { run_id: string; call_id: string; tool_result: { status: 'ok' | 'error' | 'unresolved'; result: Record<string, unknown> } };
};
export type RunCancelRequest = {
  protocol_version: '1.1'; request_id: string; method: 'run_cancel'; params: { run_id: string };
};
export type ProviderStatusRequest = {
  protocol_version: '1.1'; request_id: string; method: 'provider_status'; params: Record<string, never>;
};
export type ServiceRequest = HelloRequest | ProposalRequest | RunStartRequest | RunContinueRequest | RunCancelRequest | ProviderStatusRequest;

export class ProtocolFault extends Error {
  readonly code: string;
  readonly requestId: string;
  readonly recovery: string;
  constructor(code: string, message: string, requestId: string, recovery: string) {
    super(message);
    this.code = code;
    this.requestId = requestId;
    this.recovery = recovery;
  }
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], requestId: string): void {
  const actual = Object.keys(value);
  if (actual.length !== keys.length || actual.some(key => !keys.includes(key))) {
    throw new ProtocolFault('UNSUPPORTED_SCHEMA', 'Unexpected or missing protocol field.', requestId,
      'Update the client to the supported protocol schema.');
  }
}

function boundedString(value: unknown, requestId: string, field: string): string {
  if (typeof value !== 'string' || value.length === 0 || Buffer.byteLength(value, 'utf8') > 4096 || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(value)) {
    throw new ProtocolFault('INVALID_ARGUMENT', `${field} must be a nonempty UTF-8 string of at most 4 KiB.`,
      requestId, `Provide a valid ${field}.`);
  }
  return value;
}

function boundedText(value: unknown, requestId: string, field: string, maxBytes: number): string {
  if (typeof value !== 'string' || Buffer.byteLength(value, 'utf8') > maxBytes || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(value)) {
    throw new ProtocolFault('INVALID_ARGUMENT', `${field} must be UTF-8 of at most ${maxBytes} bytes.`, requestId,
      `Provide a bounded ${field}.`);
  }
  return value;
}

function boundedInteger(value: unknown, min: number, max: number, requestId: string, field: string): number {
  if (!Number.isInteger(value) || Number(value) < min || Number(value) > max) {
    throw new ProtocolFault('INVALID_ARGUMENT', `${field} must be an integer from ${min} to ${max}.`, requestId,
      `Choose a supported ${field}.`);
  }
  return Number(value);
}

export function parseRequest(text: string): ServiceRequest {
  let raw: unknown;
  try { raw = parseStrictJson(text); }
  catch {
    throw new ProtocolFault('PROVIDER_PROTOCOL_ERROR', 'Malformed JSON request.', '__protocol_error__',
      'Send one valid JSON request per UTF-8 line.');
  }
  if (!record(raw)) {
    throw new ProtocolFault('UNSUPPORTED_SCHEMA', 'Request must be an object.', '__protocol_error__',
      'Send a request envelope object.');
  }
  const requestId = typeof raw.request_id === 'string' && raw.request_id.length > 0 &&
    Buffer.byteLength(raw.request_id, 'utf8') <= 4096 ? raw.request_id : '__protocol_error__';
  exactKeys(raw, ['protocol_version', 'request_id', 'method', 'params'], requestId);
  boundedString(raw.request_id, requestId, 'request_id');
  if (raw.protocol_version !== '1.0' && raw.protocol_version !== '1.1') {
    throw new ProtocolFault('UNSUPPORTED_SCHEMA', 'Unsupported protocol version.', requestId,
      'Use protocol version 1.0.');
  }
  if (typeof raw.method !== 'string' || !record(raw.params)) {
    throw new ProtocolFault('INVALID_ARGUMENT', 'Method and params must have the required types.', requestId,
      'Supply a method string and params object.');
  }
  if (raw.protocol_version === '1.1') {
    if (raw.method === 'provider_status') {
      exactKeys(raw.params, [], requestId);
      return { protocol_version: '1.1', request_id: requestId, method: 'provider_status', params: {} };
    }
    if (raw.method === 'run_start') {
      exactKeys(raw.params, ['run_id', 'provider', 'model', 'intent', 'prompt', 'context', 'limits', 'live_authorized'], requestId);
      const run_id = boundedString(raw.params.run_id, requestId, 'run_id');
      if (Buffer.byteLength(run_id) > 128) throw new ProtocolFault('INVALID_ARGUMENT', 'run_id exceeds 128 bytes.', requestId, 'Choose a shorter run ID.');
      const provider = raw.params.provider;
      const model = raw.params.model;
      const intent = raw.params.intent;
      if (provider !== 'fake' && provider !== 'openai_responses') throw new ProtocolFault('INVALID_ARGUMENT', 'Unknown provider.', requestId, 'Select a configured provider.');
      if (intent !== 'discuss' && intent !== 'propose' && intent !== 'execute') throw new ProtocolFault('INVALID_ARGUMENT', 'Unknown intent.', requestId, 'Choose Discuss, Propose, or Execute.');
      if (typeof raw.params.live_authorized !== 'boolean') throw new ProtocolFault('INVALID_ARGUMENT', 'live_authorized must be boolean.', requestId, 'Use the trusted live-run control.');
      if (provider === 'fake' && (model !== null || raw.params.live_authorized)) throw new ProtocolFault('INVALID_ARGUMENT', 'Fake provider requires null model and no live authorization.', requestId, 'Use offline fake mode.');
      if (provider === 'openai_responses' && (typeof model !== 'string' || !model.trim() || Buffer.byteLength(model) > 128 || !raw.params.live_authorized)) throw new ProtocolFault('LIVE_AUTHORIZATION_REQUIRED', 'A live OpenAI run requires an explicit model and live authorization.', requestId, 'Choose the model and authorize this bounded run.');
      const prompt = boundedText(raw.params.prompt, requestId, 'prompt', 16384);
      const context = boundedText(raw.params.context, requestId, 'context', 32768);
      if (!prompt.trim()) throw new ProtocolFault('INVALID_ARGUMENT', 'Prompt is empty.', requestId, 'Enter a request.');
      try {
        if (!record(parseStrictJson(context))) throw new Error('Context must be an object.');
      } catch {
        throw new ProtocolFault('INVALID_ARGUMENT', 'Context must be one bounded JSON object.', requestId, 'Select a valid host context.');
      }
      if (!record(raw.params.limits)) throw new ProtocolFault('INVALID_ARGUMENT', 'Missing run limits.', requestId, 'Provide finite limits.');
      exactKeys(raw.params.limits, ['max_attempts', 'max_tool_calls', 'max_output_tokens', 'request_timeout_ms', 'deadline_ms'], requestId);
      const limits: RunLimits = {
        max_attempts: boundedInteger(raw.params.limits.max_attempts, 1, 8, requestId, 'max_attempts'),
        max_tool_calls: boundedInteger(raw.params.limits.max_tool_calls, 1, 24, requestId, 'max_tool_calls'),
        max_output_tokens: boundedInteger(raw.params.limits.max_output_tokens, 64, 4096, requestId, 'max_output_tokens'),
        request_timeout_ms: boundedInteger(raw.params.limits.request_timeout_ms, 1000, 120000, requestId, 'request_timeout_ms'),
        deadline_ms: boundedInteger(raw.params.limits.deadline_ms, 1000, 600000, requestId, 'deadline_ms'),
      };
      return { protocol_version: '1.1', request_id: requestId, method: 'run_start', params: { run_id, provider, model: model as string | null, intent, prompt, context, limits, live_authorized: raw.params.live_authorized } };
    }
    if (raw.method === 'run_continue') {
      exactKeys(raw.params, ['run_id', 'call_id', 'tool_result'], requestId);
      const run_id = boundedString(raw.params.run_id, requestId, 'run_id');
      const call_id = boundedString(raw.params.call_id, requestId, 'call_id');
      if (Buffer.byteLength(run_id) > 128 || Buffer.byteLength(call_id) > 128 || !record(raw.params.tool_result)) throw new ProtocolFault('INVALID_ARGUMENT', 'Invalid continuation identity or result.', requestId, 'Return the pending call result.');
      exactKeys(raw.params.tool_result, ['status', 'result'], requestId);
      const status = raw.params.tool_result.status;
      const result = raw.params.tool_result.result;
      if ((status !== 'ok' && status !== 'error' && status !== 'unresolved') || !record(result) || Buffer.byteLength(JSON.stringify(result)) > 65536) throw new ProtocolFault('INVALID_ARGUMENT', 'Invalid bounded tool result.', requestId, 'Return one structured result.');
      return { protocol_version: '1.1', request_id: requestId, method: 'run_continue', params: { run_id, call_id, tool_result: { status, result } } };
    }
    if (raw.method === 'run_cancel') {
      exactKeys(raw.params, ['run_id'], requestId);
      const run_id = boundedString(raw.params.run_id, requestId, 'run_id');
      if (Buffer.byteLength(run_id) > 128) throw new ProtocolFault('INVALID_ARGUMENT', 'Invalid run ID.', requestId, 'Use the active run ID.');
      return { protocol_version: '1.1', request_id: requestId, method: 'run_cancel', params: { run_id } };
    }
    throw new ProtocolFault('UNKNOWN_TOOL', 'Unknown service method.', requestId, 'Use provider_status, run_start, run_continue, or run_cancel.');
  }
  if (raw.method === 'hello') {
    exactKeys(raw.params, ['engine_revision', 'workspace_id', 'client_capabilities'], requestId);
    const engine_revision = boundedString(raw.params.engine_revision, requestId, 'engine_revision');
    const workspace_id = boundedString(raw.params.workspace_id, requestId, 'workspace_id');
    const capabilities = raw.params.client_capabilities;
    if (!Array.isArray(capabilities) || capabilities.length > 64) {
      throw new ProtocolFault('INVALID_ARGUMENT', 'client_capabilities must be an array of at most 64 strings.',
        requestId, 'Supply a bounded client_capabilities array.');
    }
    const client_capabilities = capabilities.map((value: unknown) => boundedString(value, requestId, 'client_capability'));
    return { protocol_version: '1.0', request_id: requestId, method: 'hello',
      params: { engine_revision, workspace_id, client_capabilities } };
  }
  if (raw.method === 'fake_propose_scene_patch') {
    exactKeys(raw.params, ['scene_ref', 'base_revision', 'parent_ref', 'root_class', 'scenario'], requestId);
    const scene_ref = boundedString(raw.params.scene_ref, requestId, 'scene_ref');
    const base_revision = boundedString(raw.params.base_revision, requestId, 'base_revision');
    const parent_ref = boundedString(raw.params.parent_ref, requestId, 'parent_ref');
    const root_class = raw.params.root_class;
    const scenario = raw.params.scenario;
    if (root_class !== 'Node2D' && root_class !== 'Node3D') {
      throw new ProtocolFault('INVALID_ARGUMENT', 'root_class must be Node2D or Node3D.', requestId,
        'Select a supported scene root class.');
    }
    if (scenario !== 'normal' && scenario !== 'malformed' && scenario !== 'delayed' && scenario !== 'disconnect') {
      throw new ProtocolFault('INVALID_ARGUMENT', 'Unknown fake scenario.', requestId,
        'Use normal, malformed, delayed, or disconnect.');
    }
    return { protocol_version: '1.0', request_id: requestId, method: 'fake_propose_scene_patch',
      params: { scene_ref, base_revision, parent_ref, root_class, scenario } };
  }
  throw new ProtocolFault('UNKNOWN_TOOL', 'Unknown service method.', requestId,
    'Use hello or fake_propose_scene_patch.');
}

export function errorResponse(fault: ProtocolFault, version: '1.0' | '1.1' = '1.0'): object {
  return { protocol_version: version, request_id: fault.requestId, status: 'error',
    error: { code: fault.code, message: fault.message, recovery: fault.recovery } };
}
