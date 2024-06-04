// Copyright (c) 2007, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  MachIPC.mm
//  Wrapper for mach IPC calls

#import <stdio.h>
#import "MachIPC.h"
#include "common/mac/bootstrap_compat.h"

namespace google_breakpad {
//==============================================================================
MachSendMessage::MachSendMessage(int32_t message_id) : MachMessage() {
  head.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0);

  // head.msgh_remote_port = ...; // filled out in MachPortSender::SendMessage()
  head.msgh_local_port = MACH_PORT_NULL;
  head.msgh_reserved = 0;
  head.msgh_id = 0;

  SetDescriptorCount(0);  // start out with no descriptors

  SetMessageID(message_id);
  SetData(NULL, 0);       // client may add data later
}

//==============================================================================
// returns true if successful
bool MachMessage::SetData(void* data,
                          int32_t data_length) {
  // first check to make sure we have enough space
  size_t size = CalculateSize();
  size_t new_size = size + data_length;
  
  if (new_size > sizeof(MachMessage)) {
    return false;  // not enough space
  }

  GetDataPacket()->data_length = EndianU32_NtoL(data_length);
  if (data) memcpy(GetDataPacket()->data, data, data_length);

  CalculateSize();

  return true;
}

//==============================================================================
// calculates and returns the total size of the message
// Currently, the entire message MUST fit inside of the MachMessage
//    messsage size <= sizeof(MachMessage)
mach_msg_size_t MachMessage::CalculateSize() {
  size_t size = sizeof(mach_msg_header_t) + sizeof(mach_msg_body_t);
  
  // add space for MessageDataPacket
  int32_t alignedDataLength = (GetDataLength() + 3) & ~0x3;
  size += 2*sizeof(int32_t) + alignedDataLength;
  
  // add space for descriptors
  size += GetDescriptorCount() * sizeof(MachMsgPortDescriptor);
  
  head.msgh_size = static_cast<mach_msg_size_t>(size);
  
  return head.msgh_size;
}

//==============================================================================
MachMessage::MessageDataPacket* MachMessage::GetDataPacket() {
  size_t desc_size = sizeof(MachMsgPortDescriptor)*GetDescriptorCount();
  MessageDataPacket* packet =
    reinterpret_cast<MessageDataPacket*>(padding + desc_size);

  return packet;
}

//==============================================================================
void MachMessage::SetDescriptor(int n,
                                const MachMsgPortDescriptor& desc) {
  MachMsgPortDescriptor* desc_array =
    reinterpret_cast<MachMsgPortDescriptor*>(padding);
  desc_array[n] = desc;
}

//==============================================================================
// returns true if successful otherwise there was not enough space
bool MachMessage::AddDescriptor(const MachMsgPortDescriptor& desc) {
  // first check to make sure we have enough space
  int size = CalculateSize();
  size_t new_size = size + sizeof(MachMsgPortDescriptor);
  
  if (new_size > sizeof(MachMessage)) {
    return false;  // not enough space
  }

  // unfortunately, we need to move the data to allow space for the
  // new descriptor
  u_int8_t* p = reinterpret_cast<u_int8_t*>(GetDataPacket());
  bcopy(p, p+sizeof(MachMsgPortDescriptor), GetDataLength()+2*sizeof(int32_t));
  
  SetDescriptor(GetDescriptorCount(), desc);
  SetDescriptorCount(GetDescriptorCount() + 1);

  CalculateSize();
  
  return true;
}

//==============================================================================
void MachMessage::SetDescriptorCount(int n) {
  body.msgh_descriptor_count = n;

  if (n > 0) {
    head.msgh_bits |= MACH_MSGH_BITS_COMPLEX;
  } else {
    head.msgh_bits &= ~MACH_MSGH_BITS_COMPLEX;
  }
}

//==============================================================================
MachMsgPortDescriptor* MachMessage::GetDescriptor(int n) {
  if (n < GetDescriptorCount()) {
    MachMsgPortDescriptor* desc =
      reinterpret_cast<MachMsgPortDescriptor*>(padding);
    return desc + n;
  }
  
  return nil;
}

//==============================================================================
mach_port_t MachMessage::GetTranslatedPort(int n) {
  if (n < GetDescriptorCount()) {
    return GetDescriptor(n)->GetMachPort();
  }
  return MACH_PORT_NULL;
}

#pragma mark -

//==============================================================================
// create a new mach port for receiving messages and register a name for it
ReceivePort::ReceivePort(const char* receive_port_name) {
  mach_port_t current_task = mach_task_self();

  init_result_ = mach_port_allocate(current_task,
                                    MACH_PORT_RIGHT_RECEIVE,
                                    &port_);

  if (init_result_ != KERN_SUCCESS)
    return;
    
  init_result_ = mach_port_insert_right(current_task,
                                        port_,
                                        port_,
                                        MACH_MSG_TYPE_MAKE_SEND);

  if (init_result_ != KERN_SUCCESS)
    return;

  mach_port_t task_bootstrap_port = 0;
  init_result_ = task_get_bootstrap_port(current_task, &task_bootstrap_port);

  if (init_result_ != KERN_SUCCESS)
    return;

  init_result_ = breakpad::BootstrapRegister(
      bootstrap_port,
      const_cast<char*>(receive_port_name),
      port_);
}

//==============================================================================
// create a new mach port for receiving messages
ReceivePort::ReceivePort() {
  mach_port_t current_task = mach_task_self();

  init_result_ = mach_port_allocate(current_task,
                                    MACH_PORT_RIGHT_RECEIVE,
                                    &port_);

  if (init_result_ != KERN_SUCCESS)
    return;

  init_result_ =   mach_port_insert_right(current_task,
                                          port_,
                                          port_,
                                          MACH_MSG_TYPE_MAKE_SEND);
}

//==============================================================================
// Given an already existing mach port, use it.  We take ownership of the
// port and deallocate it in our destructor.
ReceivePort::ReceivePort(mach_port_t receive_port)
  : port_(receive_port),
    init_result_(KERN_SUCCESS) {
}

//==============================================================================
ReceivePort::~ReceivePort() {
  if (init_result_ == KERN_SUCCESS)
    mach_port_deallocate(mach_task_self(), port_);
}

//==============================================================================
kern_return_t ReceivePort::WaitForMessage(MachReceiveMessage* out_message,
                                          mach_msg_timeout_t timeout) {
  if (!out_message) {
    return KERN_INVALID_ARGUMENT;
  }

  // return any error condition encountered in constructor
  if (init_result_ != KERN_SUCCESS)
    return init_result_;
  
  out_message->head.msgh_bits = 0;
  out_message->head.msgh_local_port = port_;
  out_message->head.msgh_remote_port = MACH_PORT_NULL;
  out_message->head.msgh_reserved = 0;
  out_message->head.msgh_id = 0;

  mach_msg_option_t options = MACH_RCV_MSG;
  if (timeout != MACH_MSG_TIMEOUT_NONE)
    options |= MACH_RCV_TIMEOUT;
  kern_return_t result = mach_msg(&out_message->head,
                                  options,
                                  0,
                                  sizeof(MachMessage),
                                  port_,
                                  timeout,              // timeout in ms
                                  MACH_PORT_NULL);

  return result;
}

#pragma mark -

//==============================================================================
// get a port with send rights corresponding to a named registered service
MachPortSender::MachPortSender(const char* receive_port_name) {
  mach_port_t task_bootstrap_port = 0;
  init_result_ = task_get_bootstrap_port(mach_task_self(), 
                                         &task_bootstrap_port);
  
  if (init_result_ != KERN_SUCCESS)
    return;

  init_result_ = bootstrap_look_up(task_bootstrap_port,
                    const_cast<char*>(receive_port_name),
                    &send_port_);
}

//==============================================================================
MachPortSender::MachPortSender(mach_port_t send_port) 
  : send_port_(send_port),
    init_result_(KERN_SUCCESS) {
}

//==============================================================================
kern_return_t MachPortSender::SendMessage(MachSendMessage& message,
                                          mach_msg_timeout_t timeout) {
  if (message.head.msgh_size == 0) {
    return KERN_INVALID_VALUE;    // just for safety -- never should occur
  };
  
  if (init_result_ != KERN_SUCCESS)
    return init_result_;
  
  message.head.msgh_remote_port = send_port_;

  kern_return_t result = mach_msg(&message.head,
                                  MACH_SEND_MSG | MACH_SEND_TIMEOUT,
                                  message.head.msgh_size,
                                  0,
                                  MACH_PORT_NULL,
                                  timeout,              // timeout in ms
                                  MACH_PORT_NULL);

  return result;
}

}  // namespace google_breakpad
