using System;
using System.Text;

namespace GodotTools.IdeMessaging
{
    public class MessageDecoder
    {
        private class DecodedMessage
        {
            public MessageKind? Kind;
            public string? Id;
            public MessageStatus? Status;
            public readonly StringBuilder Body = new StringBuilder();
            public uint? PendingBodyLines;

            public void Clear()
            {
                Kind = null;
                Id = null;
                Status = null;
                Body.Clear();
                PendingBodyLines = null;
            }

            public Message ToMessage()
            {
                if (!Kind.HasValue || Id == null || !Status.HasValue ||
                    !PendingBodyLines.HasValue || PendingBodyLines.Value > 0)
                    throw new InvalidOperationException();

                return new Message(Kind.Value, Id, new MessageContent(Status.Value, Body.ToString()));
            }
        }

        public enum State
        {
            Decoding,
            Decoded,
            Errored
        }

        private readonly DecodedMessage decodingMessage = new DecodedMessage();

        public State Decode(string messageLine, out Message? decodedMessage)
        {
            decodedMessage = null;

            if (!decodingMessage.Kind.HasValue)
            {
                if (!Enum.TryParse(messageLine, ignoreCase: true, out MessageKind kind))
                {
                    decodingMessage.Clear();
                    return State.Errored;
                }

                decodingMessage.Kind = kind;
            }
            else if (decodingMessage.Id == null)
            {
                decodingMessage.Id = messageLine;
            }
            else if (decodingMessage.Status == null)
            {
                if (!Enum.TryParse(messageLine, ignoreCase: true, out MessageStatus status))
                {
                    decodingMessage.Clear();
                    return State.Errored;
                }

                decodingMessage.Status = status;
            }
            else if (decodingMessage.PendingBodyLines == null)
            {
                if (!uint.TryParse(messageLine, out uint pendingBodyLines))
                {
                    decodingMessage.Clear();
                    return State.Errored;
                }

                decodingMessage.PendingBodyLines = pendingBodyLines;
            }
            else
            {
                if (decodingMessage.PendingBodyLines > 0)
                {
                    decodingMessage.Body.AppendLine(messageLine);
                    decodingMessage.PendingBodyLines -= 1;
                }
                else
                {
                    decodedMessage = decodingMessage.ToMessage();
                    decodingMessage.Clear();
                    return State.Decoded;
                }
            }

            return State.Decoding;
        }
    }
}
