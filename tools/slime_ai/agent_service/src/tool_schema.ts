import { parseStrictJson } from './strict_json.ts';

export const TOOL_NAMES = [
  'project_inspect', 'scene_inspect', 'object_inspect', 'api_describe',
  'project_search', 'code_read', 'scene_patch_preview', 'changeset_status',
] as const;
export type ToolName = typeof TOOL_NAMES[number];

type Shape = Record<string, unknown>;
const object = (value: unknown): value is Shape => value !== null && typeof value === 'object' && !Array.isArray(value);
const keys = (value: Shape, expected: string[]): boolean =>
  Object.keys(value).length === expected.length && expected.every(key => Object.hasOwn(value, key));
const bounded = (value: unknown, max = 4096): value is string =>
  typeof value === 'string' && value.length > 0 && Buffer.byteLength(value) <= max;
const integer = (value: unknown, min: number, max: number): boolean =>
  Number.isInteger(value) && Number(value) >= min && Number(value) <= max;

export class ToolArgumentError extends Error {}

// These are model-facing schemas. Native validation remains authoritative.
const POSITION_SCHEMA = {
  type: 'object',
  properties: {
    type: { type: 'string', enum: ['Vector2', 'Vector3'] },
    value: { type: 'array', items: { type: 'number' } },
  },
  required: ['type', 'value'], additionalProperties: false,
};
const CREATE_CHILD_SCHEMA = {
  type: 'object',
  properties: {
    op: { type: 'string', enum: ['create_child'] },
    parent_ref: { type: 'string' },
    class_name: { type: 'string', enum: ['Node2D', 'Node3D'] },
    name: { type: 'string' },
    properties: {
      type: 'object', properties: { position: POSITION_SCHEMA },
      required: ['position'], additionalProperties: false,
    },
  },
  required: ['op', 'parent_ref', 'class_name', 'name', 'properties'],
  additionalProperties: false,
};
const PREVIEW_SCHEMA = {
  type: 'object',
  properties: {
    scene_ref: { type: 'string' },
    base_revision: { type: 'string' },
    operations: { type: 'array', items: CREATE_CHILD_SCHEMA },
  },
  required: ['scene_ref', 'base_revision', 'operations'],
  additionalProperties: false,
};
export const OPENAI_TOOLS = [
  { name: 'project_inspect', description: 'Inspect the selected project summary.', parameters: { type: 'object', properties: {}, required: [], additionalProperties: false } },
  { name: 'scene_inspect', description: 'Inspect the selected loaded scene.', parameters: { type: 'object', properties: {}, required: [], additionalProperties: false } },
  { name: 'object_inspect', description: 'Inspect a selected object by its native reference.', parameters: { type: 'object', properties: { node_ref: { type: 'string' } }, required: ['node_ref'], additionalProperties: false } },
  { name: 'api_describe', description: 'Describe a native class property.', parameters: { type: 'object', properties: { class_name: { type: 'string' }, property: { type: 'string' } }, required: ['class_name', 'property'], additionalProperties: false } },
  { name: 'project_search', description: 'Search allowed project files for exact text.', parameters: { type: 'object', properties: { query: { type: 'string' }, page: { type: 'integer' } }, required: ['query', 'page'], additionalProperties: false } },
  { name: 'code_read', description: 'Read a bounded project-relative text range.', parameters: { type: 'object', properties: { path: { type: 'string' }, start_line: { type: 'integer' }, line_count: { type: 'integer' } }, required: ['path', 'start_line', 'line_count'], additionalProperties: false } },
  { name: 'scene_patch_preview', description: 'Preview one create_child operation in the selected scene. This never applies it.', parameters: PREVIEW_SCHEMA },
  { name: 'changeset_status', description: 'Read status of a host-created operation.', parameters: { type: 'object', properties: { operation_id: { type: 'string' } }, required: ['operation_id'], additionalProperties: false } },
] as const;

export function validateToolCall(name: unknown, rawArguments: unknown): { name: ToolName; arguments: Shape } {
  if (typeof name !== 'string' || !TOOL_NAMES.includes(name as ToolName)) throw new ToolArgumentError('Unknown tool.');
  if (typeof rawArguments !== 'string' || Buffer.byteLength(rawArguments) > 32768) throw new ToolArgumentError('Tool arguments exceed the limit.');
  let args: unknown;
  try { args = parseStrictJson(rawArguments); }
  catch { throw new ToolArgumentError('Malformed tool arguments.'); }
  if (!object(args)) throw new ToolArgumentError('Tool arguments must be an object.');
  switch (name) {
    case 'project_inspect': case 'scene_inspect':
      if (!keys(args, [])) throw new ToolArgumentError('Unexpected tool argument.');
      break;
    case 'object_inspect':
      if (!keys(args, ['node_ref']) || !bounded(args.node_ref)) throw new ToolArgumentError('Invalid node reference.');
      break;
    case 'api_describe':
      if (!keys(args, ['class_name', 'property']) || !bounded(args.class_name, 128) || !bounded(args.property, 128)) throw new ToolArgumentError('Invalid API query.');
      break;
    case 'project_search':
      if (!keys(args, ['query', 'page']) || !bounded(args.query, 512) || !integer(args.page, 0, 100)) throw new ToolArgumentError('Invalid search query.');
      break;
    case 'code_read':
      if (!keys(args, ['path', 'start_line', 'line_count']) || !bounded(args.path, 1024) || !integer(args.start_line, 1, 1000000) || !integer(args.line_count, 1, 200)) throw new ToolArgumentError('Invalid code range.');
      break;
    case 'changeset_status':
      if (!keys(args, ['operation_id']) || !bounded(args.operation_id, 128)) throw new ToolArgumentError('Invalid operation ID.');
      break;
    case 'scene_patch_preview': {
      if (!keys(args, ['scene_ref', 'base_revision', 'operations']) || !bounded(args.scene_ref) || !bounded(args.base_revision) || !Array.isArray(args.operations) || args.operations.length !== 1 || !object(args.operations[0])) throw new ToolArgumentError('Invalid scene proposal.');
      const op = args.operations[0];
      if (!keys(op, ['op', 'parent_ref', 'class_name', 'name', 'properties']) || op.op !== 'create_child' || !bounded(op.parent_ref) || (op.class_name !== 'Node2D' && op.class_name !== 'Node3D') || !bounded(op.name, 64) || !object(op.properties) || !keys(op.properties, ['position']) || !object(op.properties.position)) throw new ToolArgumentError('Unsupported scene operation.');
      const position = op.properties.position;
      const dimension = op.class_name === 'Node2D' ? 2 : 3;
      if (!keys(position, ['type', 'value']) || position.type !== (dimension === 2 ? 'Vector2' : 'Vector3') || !Array.isArray(position.value) || position.value.length !== dimension || !position.value.every((value: unknown) => typeof value === 'number' && Number.isFinite(value))) throw new ToolArgumentError('Invalid typed position.');
      break;
    }
  }
  return { name: name as ToolName, arguments: args };
}
