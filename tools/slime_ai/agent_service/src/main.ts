import { FrameDecoder } from './frame_decoder.ts';
import { DELAY_MS, responseFor } from './fake_provider.ts';
import { errorResponse, parseRequest, ProtocolFault } from './protocol.ts';
import { RunManager } from './run_manager.ts';
import { parseStrictJson } from './strict_json.ts';
import { openAICredentialStatus } from './credentials.ts';

const decoder = new FrameDecoder();
const write = (value: object): void => { process.stdout.write(`${JSON.stringify(value)}\n`); };
const diagnostic = (message: string): void => { process.stderr.write(`slime-ai-fake: ${message}\n`); };
const runs = new RunManager(write);

function responseVersion(text: string): '1.0' | '1.1' {
  try {
    const value = parseStrictJson(text);
    if (value !== null && typeof value === 'object' && !Array.isArray(value) &&
        'protocol_version' in value && value.protocol_version === '1.1') return '1.1';
  } catch { /* Bad frames use the legacy error envelope. */ }
  return '1.0';
}

async function handle(text: string): Promise<boolean> {
  try {
    const request = parseRequest(text);
    if (request.method === 'fake_propose_scene_patch') {
      if (request.params.scenario === 'disconnect') return false;
      if (request.params.scenario === 'delayed') await new Promise(resolve => setTimeout(resolve, DELAY_MS));
      if (request.params.scenario === 'malformed') {
        process.stdout.write('{"protocol_version":"1.0","request_id":\n');
        return true;
      }
    }
    if (request.protocol_version === '1.1' && request.method === 'provider_status') {
      write({ protocol_version: '1.1', request_id: request.request_id, status: 'ok', result: {
        provider: 'openai_responses', credential: await openAICredentialStatus(), endpoint: 'https://api.openai.com/v1/responses',
      } });
    } else if (request.protocol_version === '1.1') write(runs.handle(request));
    else write(responseFor(request));
  } catch (error) {
    const fault = error instanceof ProtocolFault ? error :
      new ProtocolFault('PROVIDER_PROTOCOL_ERROR', 'Unexpected service error.', '__protocol_error__',
        'Restart the local fake service.');
    diagnostic(fault.message);
    write(errorResponse(fault, responseVersion(text)));
  }
  return true;
}

for await (const chunk of process.stdin) {
  if (!Buffer.isBuffer(chunk)) continue;
  let connected = true;
  for (const event of decoder.push(chunk)) {
    if (event.kind === 'error') {
      diagnostic(event.reason);
      write(errorResponse(new ProtocolFault('PROVIDER_PROTOCOL_ERROR', event.reason,
        '__protocol_error__', 'Send a bounded valid UTF-8 frame.')));
    } else {
      connected = await handle(event.text);
      if (!connected) break;
    }
  }
  if (!connected) break;
}
for (const event of decoder.end()) diagnostic(event.reason);
