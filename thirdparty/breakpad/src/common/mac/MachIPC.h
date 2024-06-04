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
//  MachIPC.h
//
//  Some helpful wrappers for using Mach IPC calls

#ifndef MACH_IPC_H__
#define MACH_IPC_H__

#import <mach/mach.h>
#import <mach/message.h>
#import <servers/bootstrap.h>
#import <sys/types.h>

#import <CoreServices/CoreServices.h>

//==============================================================================
// DISCUSSION:
//
// The three main classes of interest are
//
//  MachMessage:    a wrapper for a mach message of the following form
//   mach_msg_header_t
//   mach_msg_body_t
//   optional descriptors
//   optional extra message data
//
//  MachReceiveMessage and MachSendMessage subclass MachMessage
//    and are used instead of MachMessage which is an abstract base class
//
//  ReceivePort:
//    Represents a mach port for which we have receive rights
//
//  MachPortSender:
//    Represents a mach port for which we have send rights
//
// Here's an example to receive a message on a server port:
//
//        // This creates our named server port
//        ReceivePort receivePort("com.Google.MyService");
//
//        MachReceiveMessage message;
//        kern_return_t result = receivePort.WaitForMessage(&message, 0);
//
//        if (result == KERN_SUCCESS && message.GetMessageID() == 57) {
//          mach_port_t task = message.GetTranslatedPort(0);
//          mach_port_t thread = message.GetTranslatedPort(1);
//
//          char* messageString = message.GetData();
//
//          printf("message string = %s\n", messageString);
//        }
//
// Here is an example of using these classes to send a message to this port:
//
//    // send to already named port
//    MachPortSender sender("com.Google.MyService");
//    MachSendMessage message(57);      // our message ID is 57
//
//    // add some ports to be translated for us
//    message.AddDescriptor(mach_task_self());     // our task
//    message.AddDescriptor(mach_thread_self());   // this thread
//
//    char messageString[] = "Hello server!\n";
//    message.SetData(messageString, strlen(messageString)+1);
//
//    kern_return_t result = sender.SendMessage(message, 1000); // timeout 1000ms
//

namespace google_breakpad {
#define PRINT_MACH_RESULT(result_, message_) \
  printf(message_" %s (%d)\n", mach_error_string(result_), result_ );

//==============================================================================
// A wrapper class for mach_msg_port_descriptor_t (with same memory layout)
// with convenient constructors and accessors
class MachMsgPortDescriptor : public mach_msg_port_descriptor_t {
 public:
  // General-purpose constructor
  MachMsgPortDescriptor(mach_port_t in_name,
                        mach_msg_type_name_t in_disposition) {
    name = in_name;
    pad1 = 0;
    pad2 = 0;
    disposition = in_disposition;
    type = MACH_MSG_PORT_DESCRIPTOR;
  }

  // For passing send rights to a port
  MachMsgPortDescriptor(mach_port_t in_name) {
    name = in_name;
    pad1 = 0;
    pad2 = 0;
    disposition = MACH_MSG_TYPE_COPY_SEND;
    type = MACH_MSG_PORT_DESCRIPTOR;
  }

  // Copy constructor
  MachMsgPortDescriptor(const MachMsgPortDescriptor& desc) {
    name = desc.name;
    pad1 = desc.pad1;
    pad2 = desc.pad2;
    disposition = desc.disposition;
    type = desc.type;
  }

  mach_port_t GetMachPort() const {
    return name;
  }

  mach_msg_type_name_t GetDisposition() const {
    return disposition;
  }

  // For convenience
  operator mach_port_t() const {
    return GetMachPort();
  }
};

//==============================================================================
// MachMessage: a wrapper for a mach message
//  (mach_msg_header_t, mach_msg_body_t, extra data)
//
//  This considerably simplifies the construction of a message for sending
//  and the getting at relevant data and descriptors for the receiver.
//
//  Currently the combined size of the descriptors plus data must be
//  less than 1024.  But as a benefit no memory allocation is necessary.
//
// TODO: could consider adding malloc() support for very large messages
//
//  A MachMessage object is used by ReceivePort::WaitForMessage
//  and MachPortSender::SendMessage
//
class MachMessage {
 public:

  // The receiver of the message can retrieve the raw data this way
  uint8_t* GetData() {
    return GetDataLength() > 0 ? GetDataPacket()->data : NULL;
  }

  uint32_t GetDataLength() {
    return EndianU32_LtoN(GetDataPacket()->data_length);
  }

  // The message ID may be used as a code identifying the type of message
  void SetMessageID(int32_t message_id) {
    GetDataPacket()->id = EndianU32_NtoL(message_id);
  }

  int32_t GetMessageID() { return EndianU32_LtoN(GetDataPacket()->id); }

  // Adds a descriptor (typically a mach port) to be translated
  // returns true if successful, otherwise not enough space
  bool AddDescriptor(const MachMsgPortDescriptor& desc);

  int GetDescriptorCount() const { return body.msgh_descriptor_count; }
  MachMsgPortDescriptor* GetDescriptor(int n);

  // Convenience method which gets the mach port described by the descriptor
  mach_port_t GetTranslatedPort(int n);

  // A simple message is one with no descriptors
  bool IsSimpleMessage() const { return GetDescriptorCount() == 0; }

  // Sets raw data for the message (returns false if not enough space)
  bool SetData(void* data, int32_t data_length);

 protected:
  // Consider this an abstract base class - must create an actual instance
  // of MachReceiveMessage or MachSendMessage

  MachMessage() {
    memset(this, 0, sizeof(MachMessage));
  }

  friend class ReceivePort;
  friend class MachPortSender;

  // Represents raw data in our message
  struct MessageDataPacket {
    int32_t      id;          // little-endian
    int32_t      data_length; // little-endian
    uint8_t      data[1];     // actual size limited by sizeof(MachMessage)
  };

  MessageDataPacket* GetDataPacket();

  void SetDescriptorCount(int n);
  void SetDescriptor(int n, const MachMsgPortDescriptor& desc);

  // Returns total message size setting msgh_size in the header to this value
  mach_msg_size_t CalculateSize();

  mach_msg_header_t  head;
  mach_msg_body_t    body;
  uint8_t            padding[1024]; // descriptors and data may be embedded here
};

//==============================================================================
// MachReceiveMessage and MachSendMessage are useful to separate the idea
// of a mach message being sent and being received, and adds increased type
// safety:
//  ReceivePort::WaitForMessage() only accepts a MachReceiveMessage
//  MachPortSender::SendMessage() only accepts a MachSendMessage

//==============================================================================
class MachReceiveMessage : public MachMessage {
 public:
  MachReceiveMessage() : MachMessage() {}
};

//==============================================================================
class MachSendMessage : public MachMessage {
 public:
  MachSendMessage(int32_t message_id);
};

//==============================================================================
// Represents a mach port for which we have receive rights
class ReceivePort {
 public:
  // Creates a new mach port for receiving messages and registers a name for it
  explicit ReceivePort(const char* receive_port_name);

  // Given an already existing mach port, use it.  We take ownership of the
  // port and deallocate it in our destructor.
  explicit ReceivePort(mach_port_t receive_port);

  // Create a new mach port for receiving messages
  ReceivePort();

  ~ReceivePort();

  // Waits on the mach port until message received or timeout
  kern_return_t WaitForMessage(MachReceiveMessage* out_message,
                               mach_msg_timeout_t timeout);

  // The underlying mach port that we wrap
  mach_port_t  GetPort() const { return port_; }

 private:
  ReceivePort(const ReceivePort&);  // disable copy c-tor

  mach_port_t   port_;
  kern_return_t init_result_;
};

//==============================================================================
// Represents a mach port for which we have send rights
class MachPortSender {
 public:
  // get a port with send rights corresponding to a named registered service
  explicit MachPortSender(const char* receive_port_name);


  // Given an already existing mach port, use it.
  explicit MachPortSender(mach_port_t send_port);

  kern_return_t SendMessage(MachSendMessage& message,
                            mach_msg_timeout_t timeout);

 private:
  MachPortSender(const MachPortSender&);  // disable copy c-tor

  mach_port_t   send_port_;
  kern_return_t init_result_;
};

}  // namespace google_breakpad

#endif // MACH_IPC_H__
