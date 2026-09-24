import { parseStrictJson } from './strict_json.ts';
import { validateToolCall, type ToolName } from './tool_schema.ts';

type Shape = Record<string, unknown>;
const object = (value: unknown): value is Shape => value !== null && typeof value === 'object' && !Array.isArray(value);

export type Usage = { input_tokens: number | null; output_tokens: number | null };
export type ReadyCall = { call_id: string; tool_name: ToolName; arguments: Shape; provider_response_id: string };
export type TurnResult = { response_id: string; text: string; call: ReadyCall | null; usage: Usage };
export type ProviderEvent =
  | { kind: 'text_delta'; text: string }
  | { kind: 'usage_update'; usage: Usage }
  | { kind: 'turn_completed'; result: TurnResult }
  | { kind: 'turn_failed'; code: string; message: string; retryable: boolean };

export class ProviderTurnError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  constructor(code: string, message: string, retryable = false) { super(message); this.code = code; this.retryable = retryable; }
}

function usageFrom(value: unknown): Usage {
  if (!object(value)) return { input_tokens: null, output_tokens: null };
  const token = (v: unknown): number | null => Number.isSafeInteger(v) && Number(v) >= 0 ? Number(v) : null;
  return { input_tokens: token(value.input_tokens), output_tokens: token(value.output_tokens) };
}

// Streaming deltas are display-only. A tool is released only from a successful
// terminal response whose complete output item agrees with the buffered item.
export class TurnAccumulator {
  private terminal = false;
  private failed = false;
  private text = '';
  private readonly argumentDeltas = new Map<string, string>();
  private readonly argumentDone = new Map<string, string>();
  private readonly itemDone = new Map<string, Shape>();
  private readonly items = new Set<string>();
  private completedResult: TurnResult | null = null;
  private finished = false;
  private readonly onEvent: (event: ProviderEvent) => void;

  constructor(onEvent: (event: ProviderEvent) => void) { this.onEvent = onEvent; }

  feed(event: unknown): void {
    if (!object(event) || typeof event.type !== 'string' || this.terminal) {
      throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Invalid or late provider event.');
    }
    switch (event.type) {
      case 'response.output_text.delta': {
        if (typeof event.delta !== 'string' || Buffer.byteLength(event.delta) > 32768 || Buffer.byteLength(this.text + event.delta) > 262144) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Text output exceeded the limit.');
        this.text += event.delta;
        this.onEvent({ kind: 'text_delta', text: event.delta });
        break;
      }
      case 'response.output_item.added': {
        if (object(event.item) && event.item.type === 'function_call') {
          if (typeof event.item.id !== 'string' || !event.item.id || this.items.has(event.item.id)) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Invalid provider call item.');
          this.items.add(event.item.id);
          if (this.items.size > 1) throw new ProviderTurnError('UNSUPPORTED_OPERATION', 'Multiple tool calls in one provider turn are unsupported.');
        }
        break;
      }
      case 'response.function_call_arguments.delta': {
        if (typeof event.item_id !== 'string' || typeof event.delta !== 'string') throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Invalid argument delta.');
        const value = (this.argumentDeltas.get(event.item_id) ?? '') + event.delta;
        if (Buffer.byteLength(value) > 32768) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Tool arguments exceeded the limit.');
        this.argumentDeltas.set(event.item_id, value);
        break;
      }
      case 'response.function_call_arguments.done': {
        if (typeof event.item_id !== 'string' || typeof event.arguments !== 'string' || this.argumentDone.has(event.item_id) || Buffer.byteLength(event.arguments) > 32768) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Invalid completed arguments.');
        const delta = this.argumentDeltas.get(event.item_id);
        if (delta !== undefined && delta !== event.arguments) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Argument delta and completed arguments differ.');
        this.argumentDone.set(event.item_id, event.arguments);
        break;
      }
      case 'response.output_item.done': {
        if (object(event.item) && event.item.type === 'function_call') {
          if (typeof event.item.id !== 'string' || this.itemDone.has(event.item.id)) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Duplicate completed item.');
          this.itemDone.set(event.item.id, event.item);
        }
        break;
      }
      case 'response.completed': {
        this.terminal = true;
        if (!object(event.response) || event.response.status !== 'completed' || typeof event.response.id !== 'string' || !Array.isArray(event.response.output)) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Provider completion is missing a successful response.');
        const response = event.response;
        const calls = (response.output as unknown[]).filter((item: unknown) => object(item) && item.type === 'function_call') as Shape[];
        if (calls.length > 1 || this.items.size > 1) throw new ProviderTurnError('UNSUPPORTED_OPERATION', 'Multiple tool calls in one provider turn are unsupported.');
        let call: ReadyCall | null = null;
        if (calls.length === 1) {
          const item = calls[0];
          const done = typeof item.id === 'string' ? this.itemDone.get(item.id) : undefined;
          const complete = typeof item.id === 'string' && typeof item.call_id === 'string' &&
            Boolean(item.call_id) && typeof item.arguments === 'string' &&
            this.items.size === 1 && this.items.has(item.id) && done !== undefined &&
            this.argumentDone.get(item.id) === item.arguments &&
            done.arguments === item.arguments && done.call_id === item.call_id && done.name === item.name;
          if (!complete) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Incomplete or inconsistent function call.');
          const validated = validateToolCall(item.name, item.arguments);
          call = { call_id: item.call_id as string, tool_name: validated.name, arguments: validated.arguments, provider_response_id: response.id as string };
        } else if (this.items.size || this.argumentDone.size || this.argumentDeltas.size) {
          throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Unfinished function call.');
        }
        const usage = usageFrom(response.usage);
        this.onEvent({ kind: 'usage_update', usage });
        this.completedResult = { response_id: response.id as string, text: this.text, call, usage };
        break;
      }
      case 'response.failed': case 'response.incomplete': {
        this.terminal = true;
        this.failed = true;
        const code = event.type === 'response.incomplete' ? 'PROVIDER_INCOMPLETE' : 'PROVIDER_FAILED';
        throw new ProviderTurnError(code, 'Provider turn did not complete successfully.');
      }
      default:
        // Ignore other documented semantic events, including response.created,
        // output_text.done, and reasoning deltas. They never grant tool authority.
        break;
    }
  }

  finish(): void {
    if (this.finished) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Provider turn was already finished.');
    if (!this.terminal) throw new ProviderTurnError('PROVIDER_DISCONNECTED', 'Provider stream ended before a terminal response.', true);
    if (this.failed) throw new ProviderTurnError('PROVIDER_FAILED', 'Provider turn failed.');
    if (!this.completedResult) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Successful terminal response is missing.');
    this.finished = true;
    this.onEvent({ kind: 'turn_completed', result: this.completedResult });
  }
}

// SSE framing is byte-oriented; TextDecoder preserves split Unicode sequences.
export async function* parseSse(stream: AsyncIterable<Uint8Array>, signal?: AbortSignal): AsyncGenerator<unknown> {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let pending = '';
  let totalBytes = 0;
  let data: string[] = [];
  const acceptLine = (line: string): unknown | undefined => {
    if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    if (line !== '') return undefined;
    const raw = data.join('\n');
    data = [];
    if (!raw || raw === '[DONE]') return undefined;
    try { return parseStrictJson(raw); }
    catch { throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Invalid provider SSE JSON.'); }
  };
  try {
    for await (const bytes of stream) {
      if (signal?.aborted) throw new ProviderTurnError('CANCELLED', 'Provider request cancelled.');
      totalBytes += bytes.byteLength;
      if (totalBytes > 2 * 1024 * 1024) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Provider stream exceeded the limit.');
      pending += decoder.decode(bytes, { stream: true });
      if (Buffer.byteLength(pending) > 262144) throw new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Provider event exceeded the limit.');
      let end: number;
      while ((end = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, end).replace(/\r$/, '');
        pending = pending.slice(end + 1);
        const value = acceptLine(line);
        if (value !== undefined) yield value;
      }
    }
    pending += decoder.decode();
  } catch (error) {
    if (error instanceof ProviderTurnError) throw error;
    throw new ProviderTurnError('PROVIDER_DISCONNECTED', 'Provider stream decoding failed.', true);
  }
  if (pending || data.length) throw new ProviderTurnError('PROVIDER_DISCONNECTED', 'Provider SSE frame was incomplete.', true);
}
