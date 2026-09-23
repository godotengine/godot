// JSON.parse accepts duplicate object keys. The wire format does not.
export class JsonParseError extends Error {}

export function parseStrictJson(source: string): unknown {
  let offset = 0;
  const fail = (message: string): never => {
    throw new JsonParseError(`${message} at character ${offset}`);
  };
  const whitespace = (): void => {
    while (offset < source.length && /[\t\n\r ]/.test(source[offset])) offset++;
  };
  const string = (): string => {
    const start = offset;
    if (source[offset++] !== '"') fail('Expected string');
    while (offset < source.length) {
      const char = source[offset++];
      if (char === '"') {
        try { return JSON.parse(source.slice(start, offset)) as string; }
        catch { return fail('Invalid string escape'); }
      }
      if (char === '\\') offset++;
      else if (char.charCodeAt(0) < 0x20) fail('Control character in string');
    }
    return fail('Unterminated string');
  };
  const value = (depth: number): void => {
    if (depth > 64) fail('JSON nesting limit exceeded');
    whitespace();
    const current = source[offset];
    if (current === '"') { string(); return; }
    if (current === '{') {
      offset++;
      whitespace();
      const keys = new Set<string>();
      if (source[offset] === '}') { offset++; return; }
      for (;;) {
        whitespace();
        if (source[offset] !== '"') fail('Expected object key');
        const key = string();
        if (keys.has(key)) fail('Duplicate object key');
        keys.add(key);
        whitespace();
        if (source[offset++] !== ':') fail('Expected colon');
        value(depth + 1);
        whitespace();
        const separator = source[offset++];
        if (separator === '}') return;
        if (separator !== ',') fail('Expected comma or closing brace');
      }
    }
    if (current === '[') {
      offset++;
      whitespace();
      if (source[offset] === ']') { offset++; return; }
      for (;;) {
        value(depth + 1);
        whitespace();
        const separator = source[offset++];
        if (separator === ']') return;
        if (separator !== ',') fail('Expected comma or closing bracket');
      }
    }
    const rest = source.slice(offset);
    const match = /^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/.exec(rest);
    if (match === null) return fail('Invalid JSON value');
    if (/^-?\d/.test(match[0]) && !Number.isFinite(Number(match[0]))) fail('Nonfinite number');
    offset += match[0].length;
  };
  value(0);
  whitespace();
  if (offset !== source.length) fail('Trailing JSON content');
  try { return JSON.parse(source) as unknown; }
  catch { return fail('Invalid JSON'); }
}
