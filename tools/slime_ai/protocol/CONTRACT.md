# Slime AI local protocol, revision 1.1

The editor launches one fixed local service process. Its standard input and output carry UTF-8 JSON objects, one line per frame. Standard error is diagnostic only. A frame is at most 1 MiB including its newline. Receivers buffer partial byte sequences until a newline, decode strictly as UTF-8, reject malformed JSON and duplicate keys, and never execute a partial frame. Unknown protocol major versions and authority-bearing unknown fields fail closed. A response repeats `request_id` exactly. No network listener is part of this revision.

## Envelope

Request: `{ "protocol_version": "1.0", "request_id": string, "method": string, "params": object }`.

Success: `{ "protocol_version": "1.0", "request_id": string, "status": "ok", "result": object }`.

Failure: `{ "protocol_version": "1.0", "request_id": string, "status": "error", "error": { "code": string, "message": string, "recovery": string } }`.

All listed fields are required. Strings are bounded to 4 KiB unless a method states otherwise. `request_id` is nonempty and is only a transport correlation ID. It is not an edit operation ID or an approval grant. The native editor assigns edit operation IDs and owns all grants, revisions, and journal records.

## Service methods

`hello` takes `engine_revision`, `workspace_id`, and `client_capabilities` strings/arrays. It returns `service_build`, `protocol_version`, and `capabilities`. The only v1 service capability is `fake_propose_scene_patch`. The editor must never treat a capability declaration as permission to edit.

`fake_propose_scene_patch` takes `scene_ref`, `base_revision`, `parent_ref`, `root_class` (`Node2D` or `Node3D`), and `scenario` (`normal`, `malformed`, `delayed`, or `disconnect`). It returns `scene_ref`, `base_revision`, and `operations`. The normal deterministic operation is one `create_child` with `parent_ref` echoed from the request, class `Node2D` or `Node3D`, name `AI_Marker`, and a typed `position` value (`Vector2` `[48, 24]` or `Vector3` `[48, 24, 0]`). `scenario` is a local test control, not an authorization. The service has no write access to the game project.

Native-only tools for this assignment are `engine_capabilities`, `session_status`, `project_inspect`, `scene_inspect`, `object_inspect`, `api_describe`, `scene_patch_preview`, `changeset_apply`, `changeset_status`, and `changeset_revert`. Only implemented methods are advertised. Unimplemented or invalid requests receive a structured error. No method accepts an `approved` field from the service or a model.

## Proposal value subset

An operation is an object with `op`, `parent_ref`, `class_name`, `name`, and `properties` for `create_child`. `properties.position` is `{ "type": "Vector2" | "Vector3", "value": [finite numbers] }`. The native command registry determines which classes and properties are writable; service output is untrusted even when produced by the fake provider. Future operation shapes require a contract revision and matching native validation.

Errors in scope: `INVALID_ARGUMENT`, `UNKNOWN_TOOL`, `UNSUPPORTED_OPERATION`, `UNSUPPORTED_SCHEMA`, `CAPABILITY_UNAVAILABLE`, `PERMISSION_REQUIRED`, `PERMISSION_DENIED`, `STALE_REFERENCE`, `REVISION_CONFLICT`, `READ_ONLY_RESOURCE`, `REQUEST_ALREADY_RECORDED`, `CANCELLED`, `APPLY_FAILED_RECOVERY_REQUIRED`, and `PROVIDER_PROTOCOL_ERROR`. `message` is safe to display; `recovery` gives a concrete next action.

## Native P03 command revision 1.1

The service wire envelope remains version `1.0`; revision 1.1 extends only the native allowlisted scene command subset. A P03 preview contains one operation in `operations` and exactly one loaded scene. A request with additional operations must return `UNSUPPORTED_OPERATION` until native multi-operation rollback and undo are tested. Do not advertise a 256-operation patch as implemented yet.

Supported operation objects have exact keys:

- `create_child`: `op`, `parent_ref`, `class_name`, `name`, `properties`. Class is built-in `Node2D` or `Node3D`, matching the selected native parent type. `properties` contains exactly one `position` typed value. The name is a nonempty, unique sibling name of at most 64 UTF-8 bytes, with no slash, colon, or Godot internal `@` prefix. The fake service uses `AI_Marker`, but native validation does not require that particular name.
- `set_property`: `op`, `node_ref`, `property`, `value`. The target is a built-in, script-free `Node2D` or `Node3D` owned by this scene (the root itself may be targeted). The property is exactly `position` with a matching finite `Vector2`/`Vector3`, or `visible` with `{ "type": "bool", "value": true | false }`. No other property or custom setter is allowed.
- `rename_node`: `op`, `node_ref`, `name`. The target is an owned built-in descendant, not the scene root or an instance/inherited node. Apply the same name validation and sibling uniqueness as create.
- `remove_node`: `op`, `node_ref`. The target is an owned built-in descendant, not the scene root. Reject an instanced/inherited/scripted subtree or a subtree containing nodes not owned by this scene. Preview names the complete removed subtree. Undo restores its ownership, sibling order, native storage properties, and structure.

Each preview reports the exact before/after typed values or complete affected subtree, base live and disk revisions, scene scope, risk, required grant, and a hash of immutable preview contents. The trusted native UI alone may issue a grant tied to that hash. `Manual` requires a grant for each operation, `Protected` requires one for `remove_node`, and `Freedom` suppresses per-operation prompts inside the explicitly selected scene scope. A mode change invalidates outstanding previews. Model/service data cannot change mode, scope, or grant. A duplicate native operation ID with an identical payload returns recorded status; same ID with another payload returns `REQUEST_ALREADY_RECORDED`.

`fixtures/native_operations_v1_1.json` is a catalog of four individual operation shapes for validator tests; its four-element array is intentionally not one valid P03 preview.

This contract deliberately does not promise a general JSON Schema implementation. Its validator covers the explicit envelope and method fields above, array lengths, and finite typed numbers. Golden fixtures in this directory establish the wire examples; native tests and service tests must both use the relevant ones.
