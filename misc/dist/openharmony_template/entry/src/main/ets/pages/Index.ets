/**************************************************************************/
/*  Index.ets                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

import plugin, { SimplifiedKeyEvent, SimplifiedMouseEvent } from 'libentry.so';
import { resourceManager } from '@kit.LocalizationKit';
import { display } from '@kit.ArkUI';
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { KeyCode } from '@kit.InputKit';

class PluginXComponentController extends XComponentController {
  ctx: common.UIAbilityContext
  resMgr: resourceManager.ResourceManager
  windowId: number

  constructor(context: common.UIAbilityContext, resourceManager: resourceManager.ResourceManager) {
    super()
    this.ctx = context
    this.resMgr = resourceManager
    this.windowId = this.ctx.windowStage.getMainWindowSync()
      .getWindowProperties()
      .id
  }

  onSurfaceCreated(surfaceId: string): void {
    plugin.setResourceManager(this.resMgr)
    plugin.setWindowId(this.windowId)
    plugin.setSurfaceId(BigInt(surfaceId))
  }

  onSurfaceChanged(surfaceId: string, rect: SurfaceRect): void {
    plugin.changeSurface(BigInt(surfaceId), rect.surfaceWidth, rect.surfaceHeight)
  }

  onSurfaceDestroyed(surfaceId: string): void {
    plugin.destroySurface(BigInt(surfaceId))
  }
}

@Entry
@Component
struct Index {
  resMgr = this.getUIContext().getHostContext()!.resourceManager
  context = getContext(this) as common.UIAbilityContext
  controller: XComponentController = new PluginXComponentController(this.context, this.resMgr)

  convertMouseButton(button: MouseButton) {
    switch (button) {
      case MouseButton.Left:
        return 1;
      case MouseButton.Right:
        return 2;
      case MouseButton.Middle:
        return 3;
      case MouseButton.Forward:
        return 8;
      case MouseButton.Back:
        return 9;
    }
    return 0;
  }

  convertMouseAction(action: MouseAction) {
    switch (action) {
      case MouseAction.Press:
        return 0;
      case MouseAction.Release:
        return 1;
    }
    return 2;
  }

  build() {
    Row() {
      XComponent({
        id: "window",
        type: XComponentType.SURFACE,
        controller: this.controller,
      })
        .onLoad(() => {
          display.getAllDisplays((err, data: Array<display.Display>) => {
            let screenWidth: number = data[0].width
            let screenHeight: number = data[0].height
            let surfaceRect: SurfaceRect = {
              offsetX: 0,
              offsetY: 0,
              surfaceWidth: screenWidth,
              surfaceHeight: screenHeight
            }
            this.controller.setXComponentSurfaceRect(surfaceRect);

            let user_permissions = this.context.resourceManager.getStringSync($r("app.string.user_permissions"));
            let user_permissions_list: Array<Permissions> = [];
            user_permissions.split(",").forEach((value) => {
              user_permissions_list.push(value.trim() as Permissions)
            })
            if (user_permissions_list.length > 0) {
              let atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
              atManager.requestPermissionsFromUser(this.context, user_permissions_list).then((data) => {
                let grantStatus: Array<number> = data.authResults;
                let length: number = grantStatus.length;
                let allowed_permissions: Array<string> = []
                for (let i = 0; i < length; i++) {
                  if (grantStatus[i] === 0) {
                    allowed_permissions.push(user_permissions_list[i])
                  }
                }
                plugin.setup(allowed_permissions.join(","))
              }).catch(() => {
                plugin.setup("")
              })
            } else {
              plugin.setup("")
            }
          })
        })
        .onTouch((event) => {
          let events: plugin.SimplifiedTouchEvent[] = [];
          let ctx = this.getUIContext()
          for (let index = 0; index < event.touches.length; index++) {
            const touch = event.touches[index];
            events.push({
              type: touch.type,
              id: touch.id,
              x: ctx.vp2px(touch.x),
              y: ctx.vp2px(touch.y),
            })
          }
          plugin.inputTouch(events);
        })
        .onMouse((event) => {
          let button = this.convertMouseButton(event.button)
          let type = this.convertMouseAction(event.action)
          if (button == 0 && type != 2) {
            return;
          }
          let ctx = this.getUIContext()
          let mouseEvent: SimplifiedMouseEvent = {
            type: type,
            button: button,
            mask: 0,
            x: ctx.vp2px(event.x),
            y: ctx.vp2px(event.y),
          };
          if (event.pressedButtons) {
            for (let index = 0; index < event.pressedButtons.length; index++) {
              mouseEvent.mask |= 1 << this.convertMouseButton(event.pressedButtons[index])
            }
          }
          plugin.inputMouse(mouseEvent);
        })
        .onAxisEvent((event) => {
          if (!event.scrollStep) {
            return;
          }
          let ctx = this.getUIContext()
          let value = event.getVerticalAxisValue()
          if (value != 0) {
            let mouseEvent: SimplifiedMouseEvent = {
              type: this.convertMouseAction(MouseAction.Press),
              button: value > 0 ? 5 : 4, // 5: WHEEL_DOWN, 4: WHEEL_UP
              mask: 0,
              x: ctx.vp2px(event.x),
              y: ctx.vp2px(event.y),
            };
            plugin.inputMouse(mouseEvent);
            mouseEvent.type = this.convertMouseAction(MouseAction.Release);
            plugin.inputMouse(mouseEvent);
          }
          value = event.getHorizontalAxisValue()
          if (value != 0) {
            let mouseEvent: SimplifiedMouseEvent = {
              type: this.convertMouseAction(MouseAction.Press),
              button: value > 0 ? 7 : 6, // 7: WHEEL_RIGHT, 4: WHEEL_LEFT
              mask: 0,
              x: ctx.vp2px(event.x),
              y: ctx.vp2px(event.y),
            };
            plugin.inputMouse(mouseEvent);
            mouseEvent.type = this.convertMouseAction(MouseAction.Release);
            plugin.inputMouse(mouseEvent);
          }
        })
        .width('100%')
        .height('100%')
        .expandSafeArea([SafeAreaType.SYSTEM])
    }
    .width('100%')
    .height('100%')
    .expandSafeArea([SafeAreaType.SYSTEM])
    .onKeyEvent((event) => {
      let code = mapKeyCode(event.keyCode);
      if (code == 0) {
        return;
      }
      let keyEvent: SimplifiedKeyEvent = {
        code: code,
        unicode: event.unicode ? event.unicode! : 0,
        pressed: event.type == KeyType.Down,
        alt: event.getModifierKeyState ? event.getModifierKeyState(['alt']) : false,
        ctrl: event.getModifierKeyState ? event.getModifierKeyState(['ctrl']) : false,
        shift: event.getModifierKeyState ? event.getModifierKeyState(['shift']) : false,
        meta: event.metaKey != 0,
      };
      plugin.inputKey(keyEvent);
    })
  }
}

function mapKeyCode(keyCode: KeyCode): number {
  const SPECIAL = (1 << 22);

  switch (keyCode) {
    // Special keys
    case KeyCode.KEYCODE_ESCAPE:
      return SPECIAL | 0x01; // Key::ESCAPE
    case KeyCode.KEYCODE_TAB:
      return SPECIAL | 0x02; // Key::TAB
    case KeyCode.KEYCODE_DEL:
      return SPECIAL | 0x04; // Key::BACKSPACE
    case KeyCode.KEYCODE_ENTER:
      return SPECIAL | 0x05; // Key::ENTER
    case KeyCode.KEYCODE_NUMPAD_ENTER:
      return SPECIAL | 0x06; // Key::KP_ENTER
    case KeyCode.KEYCODE_INSERT:
      return SPECIAL | 0x07; // Key::INSERT
    case KeyCode.KEYCODE_FORWARD_DEL:
      return SPECIAL | 0x08; // Key::KEY_DELETE
    case KeyCode.KEYCODE_BREAK:
      return SPECIAL | 0x09; // Key::PAUSE
    case KeyCode.KEYCODE_PRINT:
      return SPECIAL | 0x0A; // Key::PRINT
    case KeyCode.KEYCODE_SYSRQ:
      return SPECIAL | 0x0B; // Key::SYSREQ
    case KeyCode.KEYCODE_MOVE_HOME:
      return SPECIAL | 0x0D; // Key::HOME
    case KeyCode.KEYCODE_MOVE_END:
      return SPECIAL | 0x0E; // Key::END
    case KeyCode.KEYCODE_DPAD_LEFT:
      return SPECIAL | 0x0F; // Key::LEFT
    case KeyCode.KEYCODE_DPAD_UP:
      return SPECIAL | 0x10; // Key::UP
    case KeyCode.KEYCODE_DPAD_RIGHT:
      return SPECIAL | 0x11; // Key::RIGHT
    case KeyCode.KEYCODE_DPAD_DOWN:
      return SPECIAL | 0x12; // Key::DOWN
    case KeyCode.KEYCODE_PAGE_UP:
      return SPECIAL | 0x13; // Key::PAGEUP
    case KeyCode.KEYCODE_PAGE_DOWN:
      return SPECIAL | 0x14; // Key::PAGEDOWN
    case KeyCode.KEYCODE_SHIFT_LEFT:
    case KeyCode.KEYCODE_SHIFT_RIGHT:
      return SPECIAL | 0x15; // Key::SHIFT
    case KeyCode.KEYCODE_CTRL_LEFT:
    case KeyCode.KEYCODE_CTRL_RIGHT:
      return SPECIAL | 0x16; // Key::CTRL
    case KeyCode.KEYCODE_META_LEFT:
    case KeyCode.KEYCODE_META_RIGHT:
      return SPECIAL | 0x17; // Key::META
    case KeyCode.KEYCODE_ALT_LEFT:
    case KeyCode.KEYCODE_ALT_RIGHT:
      return SPECIAL | 0x18; // Key::ALT
    case KeyCode.KEYCODE_CAPS_LOCK:
      return SPECIAL | 0x19; // Key::CAPSLOCK
    case KeyCode.KEYCODE_NUM_LOCK:
      return SPECIAL | 0x1A; // Key::NUMLOCK
    case KeyCode.KEYCODE_SCROLL_LOCK:
      return SPECIAL | 0x1B; // Key::SCROLLLOCK

  // Function keys
    case KeyCode.KEYCODE_F1:
      return SPECIAL | 0x1C; // Key::F1
    case KeyCode.KEYCODE_F2:
      return SPECIAL | 0x1D; // Key::F2
    case KeyCode.KEYCODE_F3:
      return SPECIAL | 0x1E; // Key::F3
    case KeyCode.KEYCODE_F4:
      return SPECIAL | 0x1F; // Key::F4
    case KeyCode.KEYCODE_F5:
      return SPECIAL | 0x20; // Key::F5
    case KeyCode.KEYCODE_F6:
      return SPECIAL | 0x21; // Key::F6
    case KeyCode.KEYCODE_F7:
      return SPECIAL | 0x22; // Key::F7
    case KeyCode.KEYCODE_F8:
      return SPECIAL | 0x23; // Key::F8
    case KeyCode.KEYCODE_F9:
      return SPECIAL | 0x24; // Key::F9
    case KeyCode.KEYCODE_F10:
      return SPECIAL | 0x25; // Key::F10
    case KeyCode.KEYCODE_F11:
      return SPECIAL | 0x26; // Key::F11
    case KeyCode.KEYCODE_F12:
      return SPECIAL | 0x27; // Key::F12
    case KeyCode.KEYCODE_F13:
      return SPECIAL | 0x28; // Key::F13
    case KeyCode.KEYCODE_F14:
      return SPECIAL | 0x29; // Key::F14
    case KeyCode.KEYCODE_F15:
      return SPECIAL | 0x2A; // Key::F15
    case KeyCode.KEYCODE_F16:
      return SPECIAL | 0x2B; // Key::F16
    case KeyCode.KEYCODE_F17:
      return SPECIAL | 0x2C; // Key::F17
    case KeyCode.KEYCODE_F18:
      return SPECIAL | 0x2D; // Key::F18
    case KeyCode.KEYCODE_F19:
      return SPECIAL | 0x2E; // Key::F19
    case KeyCode.KEYCODE_F20:
      return SPECIAL | 0x2F; // Key::F20
    case KeyCode.KEYCODE_F21:
      return SPECIAL | 0x30; // Key::F21
    case KeyCode.KEYCODE_F22:
      return SPECIAL | 0x31; // Key::F22
    case KeyCode.KEYCODE_F23:
      return SPECIAL | 0x32; // Key::F23
    case KeyCode.KEYCODE_F24:
      return SPECIAL | 0x33; // Key::F24

  // Keypad keys
    case KeyCode.KEYCODE_NUMPAD_MULTIPLY:
      return SPECIAL | 0x81; // Key::KP_MULTIPLY
    case KeyCode.KEYCODE_NUMPAD_DIVIDE:
      return SPECIAL | 0x82; // Key::KP_DIVIDE
    case KeyCode.KEYCODE_NUMPAD_SUBTRACT:
      return SPECIAL | 0x83; // Key::KP_SUBTRACT
    case KeyCode.KEYCODE_NUMPAD_DOT:
      return SPECIAL | 0x84; // Key::KP_PERIOD
    case KeyCode.KEYCODE_NUMPAD_ADD:
      return SPECIAL | 0x85; // Key::KP_ADD
    case KeyCode.KEYCODE_NUMPAD_0:
      return SPECIAL | 0x86; // Key::KP_0
    case KeyCode.KEYCODE_NUMPAD_1:
      return SPECIAL | 0x87; // Key::KP_1
    case KeyCode.KEYCODE_NUMPAD_2:
      return SPECIAL | 0x88; // Key::KP_2
    case KeyCode.KEYCODE_NUMPAD_3:
      return SPECIAL | 0x89; // Key::KP_3
    case KeyCode.KEYCODE_NUMPAD_4:
      return SPECIAL | 0x8A; // Key::KP_4
    case KeyCode.KEYCODE_NUMPAD_5:
      return SPECIAL | 0x8B; // Key::KP_5
    case KeyCode.KEYCODE_NUMPAD_6:
      return SPECIAL | 0x8C; // Key::KP_6
    case KeyCode.KEYCODE_NUMPAD_7:
      return SPECIAL | 0x8D; // Key::KP_7
    case KeyCode.KEYCODE_NUMPAD_8:
      return SPECIAL | 0x8E; // Key::KP_8
    case KeyCode.KEYCODE_NUMPAD_9:
      return SPECIAL | 0x8F; // Key::KP_9

  // Menu and navigation
    case KeyCode.KEYCODE_MENU:
      return SPECIAL | 0x42; // Key::MENU
    case KeyCode.KEYCODE_HELP:
      return SPECIAL | 0x45; // Key::HELP
    case KeyCode.KEYCODE_BACK:
      return SPECIAL | 0x48; // Key::BACK
    case KeyCode.KEYCODE_FORWARD:
      return SPECIAL | 0x49; // Key::FORWARD
    case KeyCode.KEYCODE_STOP:
      return SPECIAL | 0x4A; // Key::STOP
    case KeyCode.KEYCODE_REFRESH:
      return SPECIAL | 0x4B; // Key::REFRESH

  // Volume and media keys
    case KeyCode.KEYCODE_VOLUME_DOWN:
      return SPECIAL | 0x4C; // Key::VOLUMEDOWN
    case KeyCode.KEYCODE_VOLUME_MUTE:
    case KeyCode.KEYCODE_MUTE:
      return SPECIAL | 0x4D; // Key::VOLUMEMUTE
    case KeyCode.KEYCODE_VOLUME_UP:
      return SPECIAL | 0x4E; // Key::VOLUMEUP
    case KeyCode.KEYCODE_MEDIA_PLAY:
    case KeyCode.KEYCODE_MEDIA_PLAY_PAUSE:
      return SPECIAL | 0x54; // Key::MEDIAPLAY
    case KeyCode.KEYCODE_MEDIA_STOP:
      return SPECIAL | 0x55; // Key::MEDIASTOP
    case KeyCode.KEYCODE_MEDIA_PREVIOUS:
      return SPECIAL | 0x56; // Key::MEDIAPREVIOUS
    case KeyCode.KEYCODE_MEDIA_NEXT:
      return SPECIAL | 0x57; // Key::MEDIANEXT
    case KeyCode.KEYCODE_MEDIA_RECORD:
      return SPECIAL | 0x58; // Key::MEDIARECORD
    case KeyCode.KEYCODE_SEARCH:
      return SPECIAL | 0x5B; // Key::SEARCH

  // Printable characters - Numbers
    case KeyCode.KEYCODE_0:
      return 0x0030; // Key::KEY_0
    case KeyCode.KEYCODE_1:
      return 0x0031; // Key::KEY_1
    case KeyCode.KEYCODE_2:
      return 0x0032; // Key::KEY_2
    case KeyCode.KEYCODE_3:
      return 0x0033; // Key::KEY_3
    case KeyCode.KEYCODE_4:
      return 0x0034; // Key::KEY_4
    case KeyCode.KEYCODE_5:
      return 0x0035; // Key::KEY_5
    case KeyCode.KEYCODE_6:
      return 0x0036; // Key::KEY_6
    case KeyCode.KEYCODE_7:
      return 0x0037; // Key::KEY_7
    case KeyCode.KEYCODE_8:
      return 0x0038; // Key::KEY_8
    case KeyCode.KEYCODE_9:
      return 0x0039; // Key::KEY_9

  // Letters A-Z
    case KeyCode.KEYCODE_A:
      return 0x0041; // Key::A
    case KeyCode.KEYCODE_B:
      return 0x0042; // Key::B
    case KeyCode.KEYCODE_C:
      return 0x0043; // Key::C
    case KeyCode.KEYCODE_D:
      return 0x0044; // Key::D
    case KeyCode.KEYCODE_E:
      return 0x0045; // Key::E
    case KeyCode.KEYCODE_F:
      return 0x0046; // Key::F
    case KeyCode.KEYCODE_G:
      return 0x0047; // Key::G
    case KeyCode.KEYCODE_H:
      return 0x0048; // Key::H
    case KeyCode.KEYCODE_I:
      return 0x0049; // Key::I
    case KeyCode.KEYCODE_J:
      return 0x004A; // Key::J
    case KeyCode.KEYCODE_K:
      return 0x004B; // Key::K
    case KeyCode.KEYCODE_L:
      return 0x004C; // Key::L
    case KeyCode.KEYCODE_M:
      return 0x004D; // Key::M
    case KeyCode.KEYCODE_N:
      return 0x004E; // Key::N
    case KeyCode.KEYCODE_O:
      return 0x004F; // Key::O
    case KeyCode.KEYCODE_P:
      return 0x0050; // Key::P
    case KeyCode.KEYCODE_Q:
      return 0x0051; // Key::Q
    case KeyCode.KEYCODE_R:
      return 0x0052; // Key::R
    case KeyCode.KEYCODE_S:
      return 0x0053; // Key::S
    case KeyCode.KEYCODE_T:
      return 0x0054; // Key::T
    case KeyCode.KEYCODE_U:
      return 0x0055; // Key::U
    case KeyCode.KEYCODE_V:
      return 0x0056; // Key::V
    case KeyCode.KEYCODE_W:
      return 0x0057; // Key::W
    case KeyCode.KEYCODE_X:
      return 0x0058; // Key::X
    case KeyCode.KEYCODE_Y:
      return 0x0059; // Key::Y
    case KeyCode.KEYCODE_Z:
      return 0x005A; // Key::Z

  // Punctuation and symbols
    case KeyCode.KEYCODE_SPACE:
      return 0x0020; // Key::SPACE
    case KeyCode.KEYCODE_COMMA:
      return 0x002C; // Key::COMMA
    case KeyCode.KEYCODE_PERIOD:
      return 0x002E; // Key::PERIOD
    case KeyCode.KEYCODE_MINUS:
      return 0x002D; // Key::MINUS
    case KeyCode.KEYCODE_EQUALS:
      return 0x003D; // Key::EQUAL
    case KeyCode.KEYCODE_LEFT_BRACKET:
      return 0x005B; // Key::BRACKETLEFT
    case KeyCode.KEYCODE_RIGHT_BRACKET:
      return 0x005D; // Key::BRACKETRIGHT
    case KeyCode.KEYCODE_BACKSLASH:
      return 0x005C; // Key::BACKSLASH
    case KeyCode.KEYCODE_SEMICOLON:
      return 0x003B; // Key::SEMICOLON
    case KeyCode.KEYCODE_APOSTROPHE:
      return 0x0027; // Key::APOSTROPHE
    case KeyCode.KEYCODE_GRAVE:
      return 0x0060; // Key::QUOTELEFT
    case KeyCode.KEYCODE_SLASH:
      return 0x002F; // Key::SLASH
    case KeyCode.KEYCODE_AT:
      return 0x0040; // Key::AT
    case KeyCode.KEYCODE_PLUS:
      return 0x002B; // Key::PLUS
    case KeyCode.KEYCODE_STAR:
      return 0x002A; // Key::ASTERISK
    case KeyCode.KEYCODE_POUND:
      return 0x0023; // Key::NUMBERSIGN
    default:
      return 0; // Return 0 for unmapped keys
  }
}
