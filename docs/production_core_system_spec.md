# Production Core System — Functional Specification

## 1. System Architecture

### Sources and integrations
- **OmaCRM** -> **Webhook** -> **Backend (Production Core System)**
- **Telegram Bot API** (direct user notifications)
- **Telegram Channel API** (broadcast production updates)

### Central module
- **Production Core System** (single source of truth for order card and production workflow)

---

## 2. Core Data Model

### 2.1 `users`
- `id`
- `name`
- `phone`
- `is_active`

### 2.2 `roles`
- `id`
- `code` (`CUTTER`, `SEWER`, `EMBROIDERY`, `FASTENING`, `PACKING`, `MANAGER`, `DISTRIBUTOR`)
- `name`

### 2.3 `user_roles` (many-to-many)
- `id`
- `user_id`
- `role_id`

> One employee may have multiple roles.

### 2.4 `orders`
- `id`
- `order_number`
- `crm_id`
- `client_name`
- `client_phone`
- `car_model`
- `complexity` (`LOW`, `MEDIUM`, `HIGH`)
- `workshop` (`NULL | 1 | 2`)
- `is_special` (`boolean`)
- `status` (`NEW`, `DISTRIBUTION`, `IN_PRODUCTION`, `CHANGED`, `READY`, `SHIPPED`)
- `delivery_city`
- `delivery_type`
- `created_at`
- `updated_at`

### 2.5 `order_stages`
- `id`
- `order_id`
- `stage_type` (`CUT`, `SEWING`, `EMBROIDERY`, `FASTENING`, `PACKING`)
- `status` (`NOT_STARTED`, `IN_PROGRESS`, `DONE`, `BLOCKED`)
- `confirmed_by`
- `confirmed_at`

> All stages are created automatically for each new order.

### 2.6 `order_changes`
- `id`
- `order_id`
- `field_name`
- `old_value`
- `new_value`
- `changed_by`
- `changed_at`
- `is_confirmed`
- `confirmed_by`
- `confirmed_at`

---

## 3. New Order Intake (OmaCRM Webhook)

When a new order arrives from OmaCRM:
1. Create record in `orders`.
2. Create full stage set in `order_stages` with `NOT_STARTED`.
3. Set order `status = NEW`.

---

## 4. "Send to production" Action

### Endpoint
`POST /orders/{id}/send-to-production`

### Behavior
1. Set `orders.status = DISTRIBUTION`.
2. Generate message by template.
3. Send message:
   - to client via Telegram Bot API,
   - to internal Telegram channel.
4. Log action event.

---

## 5. Manual Workshop Assignment

### Endpoint
`PATCH /orders/{id}/assign-workshop`

### Request body
```json
{
  "workshop": 1
}
```
or
```json
{
  "workshop": 2
}
```

### Behavior
1. Save selected workshop.
2. Set `orders.status = IN_PRODUCTION`.
3. Make order visible:
   - to cutter (shared across both workshops),
   - to sewers of selected workshop.

---

## 6. Visibility Rules by Role

### 6.1 Cutter (`CUTTER`)
Visible if:
- `orders.workshop IS NOT NULL`
- `CUT.status != DONE`

### 6.2 Sewer (`SEWER`)
Visible if:
- order belongs to sewer's workshop,
- `CUT.status = DONE`

### 6.3 Embroidery (`EMBROIDERY`)
Visible if:
- `EMBROIDERY.status != DONE`

### 6.4 Fastening (`FASTENING`)
Visible if:
- `FASTENING.status != DONE`
- `SEWING.status = DONE`

### 6.5 Packing (`PACKING`)
Visible if:
- all prior stages are `DONE`,
- `PACKING.status != DONE`

---

## 7. Order Change Handling

### Endpoint
`PATCH /orders/{id}`

When any critical field changes:
1. Create `order_changes` records for every changed critical field.
2. Set `orders.status = CHANGED`.
3. Set all `order_stages.status = BLOCKED`.

### Change confirmation endpoint
`POST /orders/{id}/confirm-change`

After confirmation:
1. Mark corresponding `order_changes.is_confirmed = true`.
2. Set `orders.status = IN_PRODUCTION`.
3. Unblock production stages.

---

## 8. Special Orders (`is_special = true`)

For changed special orders:
- requires **double confirmation**:
  1. `MANAGER`
  2. responsible role for affected stage

Until both confirmations are received:
- stage statuses remain `BLOCKED`.

---

## 9. Telegram Automation

The backend must call:
- `sendTelegramMessage(user_id, template)`
- `sendTelegramChannelPost(order_summary)`

Trigger points:
- send to production,
- order changes,
- shipment dispatch.

---

## 10. Automatic READY Status

When all required production stages are `DONE`:
- `CUT`
- `SEWING`
- `EMBROIDERY`
- `FASTENING`

Then automatically:
1. set `orders.status = READY`,
2. open packing work (stage `PACKING`).

---

## 11. Frontend (Kanban)

Columns:
- `NEW`
- `DISTRIBUTION`
- `IN_PRODUCTION`
- `CHANGED`
- `READY`
- `SHIPPED`

Permission:
- drag-and-drop between columns is allowed only for role `DISTRIBUTOR`.

---

## 12. Data Safety Rules

1. Order cannot be changed without logging into `order_changes`.
2. Production cannot continue while `orders.status = CHANGED`.
3. All changes require explicit confirmation workflow.
4. Change history is immutable (no edit/delete).

---

## 13. Core Principles

1. Group chat is **not** source of truth.
2. Order card in Production Core System is the single source of truth.
3. Roles control visibility.
4. Stages control execution flow.
5. Changes block production until validated.
