#!/usr/bin/env bash
GOARCH=riscv64 go build channel_ops.go
GOARCH=riscv64 go build fib.go
GOARCH=riscv64 go build example.go
GOARCH=riscv64 go build goroutines.go
GOARCH=riscv64 go build http_client.go
GOARCH=riscv64 go build timers.go
GOARCH=riscv64 go build worker_pools.go
