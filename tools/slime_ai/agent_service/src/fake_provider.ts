import type { ServiceRequest } from './protocol.ts';

export const DELAY_MS = 75;

export function responseFor(request: ServiceRequest): object {
  if (request.method === 'hello') {
    return { protocol_version: '1.0', request_id: request.request_id, status: 'ok',
      result: { service_build: 'fake-1.0', protocol_version: '1.0',
        capabilities: ['fake_propose_scene_patch'] } };
  }
  const { scene_ref, base_revision, parent_ref, root_class } = request.params;
  const is3d = root_class === 'Node3D';
  return { protocol_version: '1.0', request_id: request.request_id, status: 'ok',
    result: { scene_ref, base_revision, operations: [{ op: 'create_child', parent_ref,
      class_name: root_class, name: 'AI_Marker', properties: {
        position: { type: is3d ? 'Vector3' : 'Vector2', value: is3d ? [48, 24, 0] : [48, 24] }
      } }] } };
}
