=========================================================================
 State machine transitions for the Crash Generation Server
=========================================================================

=========================================================================
               |
 STATE         | ACTIONS
               |
=========================================================================
 ERROR         | Clean up resources used to serve clients.
               | Always remain in ERROR state.
-------------------------------------------------------------------------
 INITIAL       | Connect to the pipe asynchronously.
               | If connection is successfully queued up asynchronously,
               | go into CONNECTING state.
               | If connection is done synchronously, go into CONNECTED
               | state.
               | For any unexpected problems, go into ERROR state.
-------------------------------------------------------------------------
 CONNECTING    | Get the result of async connection request.
               | If I/O is still incomplete, remain in the CONNECTING
               | state.
               | If connection is complete, go into CONNECTED state.
               | For any unexpected problems, go into DISCONNECTING state.
-------------------------------------------------------------------------
 CONNECTED     | Read from the pipe asynchronously.
               | If read request is successfully queued up asynchronously,
               | go into READING state.
               | For any unexpected problems, go into DISCONNECTING state.
-------------------------------------------------------------------------
 READING       | Get the result of async read request.
               | If read is done, go into READ_DONE state.
               | For any unexpected problems, go into DISCONNECTING state.
-------------------------------------------------------------------------
 READ_DONE     | Register the client, prepare the reply and write the
               | reply to the pipe asynchronously.
               | If write request is successfully queued up asynchronously,
               | go into WRITING state.
               | For any unexpected problems, go into DISCONNECTING state.
-------------------------------------------------------------------------
 WRITING       | Get the result of the async write request.
               | If write is done, go into WRITE_DONE state.
               | For any unexpected problems, go into DISCONNECTING state.
-------------------------------------------------------------------------
 WRITE_DONE    | Read from the pipe asynchronously (for an ACK).
               | If read request is successfully queued up asynchonously,
               | go into READING_ACK state.
               | For any unexpected problems, go into DISCONNECTING state.
-------------------------------------------------------------------------
 READING_ACK   | Get the result of the async read request.
               | If read is done, perform action for successful client
               | connection.
               | Go into DISCONNECTING state.
-------------------------------------------------------------------------
 DISCONNECTING | Disconnect from the pipe, reset the event and go into
               | INITIAL state and signal the event again. If anything
               | fails, go into ERROR state.
=========================================================================
