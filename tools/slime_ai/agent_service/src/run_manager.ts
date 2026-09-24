import { createHash } from 'node:crypto';
import { fakeResponse, openAIResponse, type ProviderRequest } from './providers.ts';
import { ProviderTurnError, type ProviderEvent, type ReadyCall } from './provider_events.ts';
import { ProtocolFault, type RunCancelRequest, type RunContinueRequest, type RunStartRequest } from './protocol.ts';

type Request = RunStartRequest | RunContinueRequest | RunCancelRequest;
type RunState = 'waiting_for_provider' | 'awaiting_tool' | 'completed' | 'failed' | 'cancelled' | 'reconciliation_required';
type ActiveRun = {
  start: RunStartRequest; eventRequestId: string; state: RunState; startedAt: number;
  attempts: number; toolCalls: number; previousResponseId?: string; pending?: ReadyCall;
  proposalRepairs: number; acceptedCallId?: string; acceptedHash?: string; abort: AbortController;
};

const hash = (value: unknown): string => createHash('sha256').update(JSON.stringify(value)).digest('hex');

export class RunManager {
  private run: ActiveRun | null = null;
  private readonly write: (value: object) => void;
  private readonly provider: (request: ProviderRequest, onEvent: (event: ProviderEvent) => void) => Promise<{ response_id: string; call: ReadyCall | null }>;
  constructor(write: (value: object) => void,
    provider: (request: ProviderRequest, onEvent: (event: ProviderEvent) => void) => Promise<{ response_id: string; call: ReadyCall | null }> =
      (request, onEvent) => request.provider === 'fake' ? fakeResponse(request, onEvent) : openAIResponse(request, onEvent)) {
    this.write = write;
    this.provider = provider;
  }

  handle(request: Request): object {
    if (request.method === 'run_start') return this.start(request);
    if (request.method === 'run_continue') return this.continue(request);
    return this.cancel(request);
  }

  private fault(code: string, message: string, request: Request): never {
    throw new ProtocolFault(code, message, request.request_id, 'Inspect the run state and retry only after a safe boundary.');
  }

  private event(run: ActiveRun, event: string, data: object, afterAck = false): void {
    const frame = { protocol_version: '1.1', request_id: run.eventRequestId, run_id: run.start.params.run_id, event, data };
    if (afterAck) queueMicrotask(() => this.write(frame));
    else this.write(frame);
  }

  private start(request: RunStartRequest): object {
    if (this.run && this.run.state !== 'completed' && this.run.state !== 'failed' && this.run.state !== 'cancelled') this.fault('RUN_BUSY', 'A bounded run is already active.', request);
    if (this.run?.start.params.run_id === request.params.run_id) this.fault('REQUEST_ALREADY_RECORDED', 'Run ID has already been used in this service session.', request);
    const run: ActiveRun = { start: request, eventRequestId: request.request_id, state: 'waiting_for_provider', startedAt: Date.now(), attempts: 0, toolCalls: 0, proposalRepairs: 0, abort: new AbortController() };
    this.run = run;
    setImmediate(() => void this.executeTurn(run));
    return { protocol_version: '1.1', request_id: request.request_id, status: 'ok', result: { run_id: request.params.run_id, state: run.state, provider: request.params.provider, model: request.params.model } };
  }

  private continue(request: RunContinueRequest): object {
    const run = this.run;
    if (!run || run.start.params.run_id !== request.params.run_id) this.fault('STALE_REFERENCE', 'Run ID is not active.', request);
    const digest = hash(request.params.tool_result);
    if (run.acceptedCallId === request.params.call_id) {
      if (run.acceptedHash !== digest) this.fault('REQUEST_ALREADY_RECORDED', 'Call ID was reused with a different result.', request);
      return { protocol_version: '1.1', request_id: request.request_id, status: 'ok', result: { run_id: request.params.run_id, state: run.state, duplicate: true } };
    }
    if (run.state !== 'awaiting_tool' || !run.pending || run.pending.call_id !== request.params.call_id) this.fault('STALE_REFERENCE', 'No matching tool call is pending.', request);
    const pendingTool = run.pending.tool_name;
    run.eventRequestId = request.request_id;
    run.acceptedCallId = request.params.call_id;
    run.acceptedHash = digest;
    run.pending = undefined;
    if (request.params.tool_result.status === 'unresolved') {
      run.state = 'reconciliation_required';
      this.event(run, 'run_state', { state: run.state, message: 'Native result unresolved; no new provider request or replay.' }, true);
    } else if (request.params.tool_result.status === 'error' && pendingTool === 'scene_patch_preview' && ++run.proposalRepairs > 2) {
      run.state = 'failed';
      this.event(run, 'turn_failed', { code: 'RUN_LIMIT', message: 'Proposal repair limit reached.', attempts: run.attempts }, true);
      this.event(run, 'run_state', { state: run.state, attempts: run.attempts, tool_calls: run.toolCalls }, true);
    } else {
      run.state = 'waiting_for_provider';
      setImmediate(() => void this.executeTurn(run, request.params.call_id, JSON.stringify(request.params.tool_result)));
    }
    return { protocol_version: '1.1', request_id: request.request_id, status: 'ok', result: { run_id: request.params.run_id, state: run.state } };
  }

  private cancel(request: RunCancelRequest): object {
    const run = this.run;
    if (!run || run.start.params.run_id !== request.params.run_id) this.fault('STALE_REFERENCE', 'Run ID is not active.', request);
    if (run.state !== 'completed' && run.state !== 'failed' && run.state !== 'cancelled') {
      run.state = 'cancelled';
      run.abort.abort();
      run.eventRequestId = request.request_id;
      this.event(run, 'run_state', { state: run.state, message: 'No further model work; already dispatched native work needs status reconciliation.' }, true);
    }
    return { protocol_version: '1.1', request_id: request.request_id, status: 'ok', result: { run_id: request.params.run_id, state: run.state } };
  }

  private async executeTurn(run: ActiveRun, callId?: string, toolResult?: string): Promise<void> {
    const { limits } = run.start.params;
    let retries = 0;
    while (run.state === 'waiting_for_provider') {
      if (Date.now() >= run.startedAt + limits.deadline_ms || run.attempts >= limits.max_attempts) {
        run.state = 'failed';
        this.event(run, 'turn_failed', { code: 'RUN_LIMIT', message: 'Run deadline or model-attempt limit reached.', attempts: run.attempts });
        this.event(run, 'run_state', { state: run.state, attempts: run.attempts, tool_calls: run.toolCalls });
        return;
      }
      run.attempts++;
      this.event(run, 'run_state', { state: run.state, attempts: run.attempts, tool_calls: run.toolCalls });
      let textEmitted = false;
      try {
        const params = run.start.params;
        const request: ProviderRequest = {
          provider: params.provider, model: params.model, intent: params.intent, prompt: params.prompt,
          context: params.context, max_output_tokens: limits.max_output_tokens,
          request_timeout_ms: Math.min(limits.request_timeout_ms, Math.max(1, run.startedAt + limits.deadline_ms - Date.now())),
          previous_response_id: run.previousResponseId, call_id: callId, tool_result: toolResult, signal: run.abort.signal,
        };
        const turn = await this.provider(request, event => {
          if (run.state !== 'waiting_for_provider') return;
          if (event.kind === 'text_delta') { textEmitted = true; this.event(run, 'text_delta', { text: event.text, attempt: run.attempts }); }
          else if (event.kind === 'usage_update') this.event(run, 'usage_update', { ...event.usage, attempt: run.attempts });
          else if (event.kind === 'turn_completed') this.event(run, 'turn_completed', { response_id: event.result.response_id, provider_turn_complete: true, task_complete: false, has_tool_call: event.result.call !== null });
        });
        if (run.state !== 'waiting_for_provider') return;
        run.previousResponseId = turn.response_id;
        if (turn.call) {
          if (run.toolCalls >= limits.max_tool_calls) throw new ProviderTurnError('RUN_LIMIT', 'Tool-call limit reached.');
          if (params.intent === 'discuss' && turn.call.tool_name === 'scene_patch_preview') throw new ProviderTurnError('PERMISSION_DENIED', 'Discuss cannot request scene previews.');
          run.toolCalls++;
          run.pending = turn.call;
          run.state = 'awaiting_tool';
          this.event(run, 'tool_call_ready', turn.call);
          this.event(run, 'run_state', { state: run.state, attempts: run.attempts, tool_calls: run.toolCalls });
        } else {
          run.state = 'completed';
          this.event(run, 'run_state', { state: run.state, attempts: run.attempts, tool_calls: run.toolCalls });
        }
        return;
      } catch (error) {
        if ((run as ActiveRun).state === 'cancelled') return;
        const fault = error instanceof ProviderTurnError ? error : new ProviderTurnError('PROVIDER_PROTOCOL_ERROR', 'Provider turn was rejected.');
        if (fault.retryable && !textEmitted && retries < 2 && run.attempts < limits.max_attempts && Date.now() < run.startedAt + limits.deadline_ms) {
          retries++;
          this.event(run, 'run_state', { state: run.state, retrying: true, code: fault.code, attempts: run.attempts });
          continue;
        }
        run.state = 'failed';
        this.event(run, 'turn_failed', { code: fault.code, message: fault.message, attempts: run.attempts });
        this.event(run, 'run_state', { state: run.state, attempts: run.attempts, tool_calls: run.toolCalls });
        return;
      }
    }
  }
}
