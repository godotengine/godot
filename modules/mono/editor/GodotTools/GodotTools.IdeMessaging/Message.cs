namespace GodotTools.IdeMessaging
{
    public class Message
    {
        public MessageKind Kind { get; }
        public string Id { get; }
        public MessageContent Content { get; }

        public Message(MessageKind kind, string id, MessageContent content)
        {
            Kind = kind;
            Id = id;
            Content = content;
        }

        public override string ToString()
        {
            return $"{Kind} | {Id}";
        }
    }

    public enum MessageKind
    {
        Request,
        Response
    }

    public enum MessageStatus
    {
        Ok,
        RequestNotSupported,
        InvalidRequestBody
    }

    public readonly struct MessageContent
    {
        public MessageStatus Status { get; }
        public string Body { get; }

        public MessageContent(string body)
        {
            Status = MessageStatus.Ok;
            Body = body;
        }

        public MessageContent(MessageStatus status, string body)
        {
            Status = status;
            Body = body;
        }
    }
}
