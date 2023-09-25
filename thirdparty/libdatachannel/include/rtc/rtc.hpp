/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// C API
#include "rtc.h"

// C++ API
#include "common.hpp"
#include "global.hpp"
//
#include "datachannel.hpp"
#include "peerconnection.hpp"
#include "track.hpp"

#if RTC_ENABLE_WEBSOCKET

// WebSocket
#include "websocket.hpp"
#include "websocketserver.hpp"

#endif // RTC_ENABLE_WEBSOCKET

#if RTC_ENABLE_MEDIA

// Media handling
#include "mediachainablehandler.hpp"
#include "rtcpnackresponder.hpp"
#include "rtcpreceivingsession.hpp"
#include "rtcpsrreporter.hpp"

// Opus/AAC/h264/h265/AV1 streaming
#include "aacrtppacketizer.hpp"
#include "av1packetizationhandler.hpp"
#include "h264packetizationhandler.hpp"
#include "h265packetizationhandler.hpp"
#include "opuspacketizationhandler.hpp"

#endif // RTC_ENABLE_MEDIA
