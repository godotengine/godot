#!/bin/bash
set -e

mkdir -p .build
pushd .build
make -j 4
popd

DIR=$HOME/github/sail-riscv/test/riscv-tests

# Skipping fence.i instruction
for i in $DIR/rv32ui-v-*.elf; do
	if [[ ${i} != *"fence_i"* ]];then
		.build/elfverify $i
	fi
done
for i in $DIR/rv64ui-v-*.elf; do
	if [[ ${i} != *"fence_i"* ]];then
		.build/elfverify $i
	fi
done

for i in $DIR/rv32um-v-*.elf; do
    .build/elfverify $i
done
for i in $DIR/rv64um-v-*.elf; do
    .build/elfverify $i
done

for i in $DIR/rv32ua-v-*.elf; do
    .build/elfverify $i
done
for i in $DIR/rv64ua-v-*.elf; do
    .build/elfverify $i
done

for i in $DIR/rv32uf-v-*.elf; do
	if [[ ${i} != *"fclass"* ]];then
	if [[ ${i} != *"fcvt"* ]];then
	if [[ ${i} != *"fmin"* ]];then
		.build/elfverify $i
	fi
	fi
	fi
done
for i in $DIR/rv64uf-v-*.elf; do
	if [[ ${i} != *"fclass"* ]];then
	if [[ ${i} != *"fcvt"* ]];then
	if [[ ${i} != *"fmin"* ]];then
		echo "Running $i"
		.build/elfverify $i
	fi
	fi
	fi
done

for i in $DIR/rv32ud-v-*.elf; do
	if [[ ${i} != *"fclass"* ]];then
	if [[ ${i} != *"fcvt"* ]];then
	if [[ ${i} != *"fmin"* ]];then
		.build/elfverify $i
	fi
	fi
	fi
done
for i in $DIR/rv64ud-v-*.elf; do
	if [[ ${i} != *"fclass"* ]];then
	if [[ ${i} != *"fcvt"* ]];then
	if [[ ${i} != *"fmin"* ]];then
		.build/elfverify $i
	fi
	fi
	fi
done
