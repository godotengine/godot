#version 460 core

layout(binding = 0) uniform atomic_uint aui;
uint ui;

void main()
{
    atomicCounterAdd(aui, ui);
    atomicCounterSubtract(aui, ui);
    atomicCounterMin(aui, ui);
    atomicCounterMax(aui, ui);
    atomicCounterAnd(aui, ui);
    atomicCounterOr(aui, ui);
    atomicCounterXor(aui, ui);
    atomicCounterExchange(aui, ui);
    atomicCounterCompSwap(aui, ui, ui);
}
