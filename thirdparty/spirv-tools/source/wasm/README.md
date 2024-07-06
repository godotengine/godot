# SPIRV-Tools

Wasm (WebAssembly) build of https://github.com/KhronosGroup/SPIRV-Tools

## Usage

```js
const spirvTools = require("spirv-tools");

const test = async () => {
  // Load the library
  const spv = await spirvTools();

  // assemble
  const source = `
             OpCapability Linkage 
             OpCapability Shader 
             OpMemoryModel Logical GLSL450 
             OpSource GLSL 450 
             OpDecorate %spec SpecId 1 
      %int = OpTypeInt 32 1 
     %spec = OpSpecConstant %int 0 
    %const = OpConstant %int 42`;
  const asResult = spv.as(
    source,
    spv.SPV_ENV_UNIVERSAL_1_3,
    spv.SPV_TEXT_TO_BINARY_OPTION_NONE
  );
  console.log(`as returned ${asResult.byteLength} bytes`);

  // re-disassemble
  const disResult = spv.dis(
    asResult,
    spv.SPV_ENV_UNIVERSAL_1_3,
    spv.SPV_BINARY_TO_TEXT_OPTION_INDENT |
      spv.SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
      spv.SPV_BINARY_TO_TEXT_OPTION_COLOR
  );
  console.log("dis:\n", disResult);
};

test();
```
