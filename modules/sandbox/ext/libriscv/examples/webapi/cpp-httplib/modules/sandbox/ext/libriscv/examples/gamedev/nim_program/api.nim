### 
### API for the Nim program
###
proc fast_write(fd: int, text: cstring, tlen: int) {.importc}
proc fast_exit(status: int) {.importc}
proc dyncall1*(i: int): int {.importc}
proc dyncall3*() {.importc}

type
    MyData* = object
        buffer*: array[32, cchar]
proc dyncall4*(d1: ptr MyData, s1: csize_t, d2: ptr MyData) {.importc}

proc print*(content: string) =
    fast_write(1, content, len(content))

proc exit*(status: int) =
    fast_exit(status)
