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
export type ServiceRequest = HelloRequest | ProposalRequest;

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
  if (raw.protocol_version !== '1.0') {
    throw new ProtocolFault('UNSUPPORTED_SCHEMA', 'Unsupported protocol version.', requestId,
      'Use protocol version 1.0.');
  }
  if (typeof raw.method !== 'string' || !record(raw.params)) {
    throw new ProtocolFault('INVALID_ARGUMENT', 'Method and params must have the required types.', requestId,
      'Supply a method string and params object.');
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

export function errorResponse(fault: ProtocolFault): object {
  return { protocol_version: '1.0', request_id: fault.requestId, status: 'error',
    error: { code: fault.code, message: fault.message, recovery: fault.recovery } };
}
