export const MAX_FRAME_BYTES = 1024 * 1024;

export type FrameEvent = { kind: 'frame'; text: string } | { kind: 'error'; reason: string };

export class FrameDecoder {
  private pending = Buffer.alloc(0);
  private discarding = false;
  private readonly utf8 = new TextDecoder('utf-8', { fatal: true });

  push(bytes: Buffer): FrameEvent[] {
    const events: FrameEvent[] = [];
    let start = 0;
    while (start < bytes.length) {
      const newline = bytes.indexOf(0x0a, start);
      const end = newline < 0 ? bytes.length : newline + 1;
      const fragment = bytes.subarray(start, end);
      start = end;
      if (this.discarding) {
        if (newline >= 0) this.discarding = false;
        continue;
      }
      if (this.pending.length + fragment.length > MAX_FRAME_BYTES) {
        events.push({ kind: 'error', reason: 'Frame exceeds 1 MiB' });
        this.pending = Buffer.alloc(0);
        this.discarding = newline < 0;
        continue;
      }
      this.pending = Buffer.concat([this.pending, fragment]);
      if (newline < 0) continue;
      const frame = this.pending.subarray(0, this.pending.length - 1);
      this.pending = Buffer.alloc(0);
      try { events.push({ kind: 'frame', text: this.utf8.decode(frame) }); }
      catch { events.push({ kind: 'error', reason: 'Invalid UTF-8 frame' }); }
    }
    return events;
  }

  end(): Extract<FrameEvent, { kind: 'error' }>[] {
    if (!this.discarding && this.pending.length > 0) return [{ kind: 'error', reason: 'Incomplete frame at EOF' }];
    return [];
  }
}
