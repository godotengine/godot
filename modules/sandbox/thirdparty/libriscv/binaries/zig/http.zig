const std = @import("std");
var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 12 }){};
const alloc = gpa.allocator();

pub fn main() !void {
    defer _ = gpa.deinit();

    // our http client, this can make multiple requests (and is even threadsafe, although individual requests are not).
    var client = std.http.Client{
        .allocator = alloc,
    };

    // we can `catch unreachable` here because we can guarantee that this is a valid url.
    const uri = std.Uri.parse("https://example.com/") catch unreachable;

	var read_buffer: [8000]u8 = undefined;

    // make the connection and set up the request
    var req = try client.open(.GET, uri, .{
        .server_header_buffer = &read_buffer,
        .extra_headers = &.{
            .{ .name = "cookie", .value = "RISCV=awesome" },
        },
	});
    defer req.deinit();

    // send the request and headers to the server.
    try req.send();
    try req.wait();

    // read the entire response body
    const body = req.reader().readAllAlloc(alloc, 1024 * 1024) catch unreachable;
    defer alloc.free(body);

    std.debug.print("{s}", .{body});
}
