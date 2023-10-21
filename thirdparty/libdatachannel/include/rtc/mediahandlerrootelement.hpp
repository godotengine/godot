/**
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_MEDIA_HANDLER_ROOT_ELEMENT_H
#define RTC_MEDIA_HANDLER_ROOT_ELEMENT_H

#if RTC_ENABLE_MEDIA

#include "mediahandlerelement.hpp"

namespace rtc {

/// Chainable message handler
class RTC_CPP_EXPORT MediaHandlerRootElement : public MediaHandlerElement {
public:
	MediaHandlerRootElement() {}

	/// Reduce multiple messages into one message
	/// @param messages Messages to reduce
	virtual message_ptr reduce(ChainedMessagesProduct messages);

	/// Splits message into multiple messages
	/// @param message Message to split
	virtual ChainedMessagesProduct split(message_ptr message);
};

} // namespace rtc

#endif // RTC_ENABLE_MEDIA

#endif // RTC_MEDIA_HANDLER_ROOT_ELEMENT_H
