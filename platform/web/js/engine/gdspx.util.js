// TODO @jiepengtan cache function pointer

// Bool-related functions
function ToGdBool(value) {
    return Module._gdspx_new_bool(value);
}

function ToJsBool(ptr) {
    const HEAPU8 = Module.HEAPU8;
    const boolValue = HEAPU8[ptr];
    return boolValue !== 0;
}

function AllocGdBool() {
    return Module._gdspx_alloc_bool();
}

function PrintGdBool(ptr) {
    console.log(ToJsBool(ptr));
}

function FreeGdBool(ptr) {
    Module._gdspx_free_bool(ptr);
}



// Object-related functions
function ToGdObject(object) {
    return ToGdObj(object);
}
function ToJsObject(ptr) {
    return ToJsObj(ptr);
}
function FreeGdObject(ptr) {
    FreeGdObj(ptr);
}
function AllocGdObject() {
    return AllocGdObj();
}
function PrintGdObject(ptr) {
    PrintGdObj(ptr);
}

function ToGdObj(value) {
    return Module._gdspx_new_obj(value.high, value.low);
}

function ToJsObj(ptr) {
    const memoryBuffer = Module.HEAPU8.buffer;
    const dataView = new DataView(memoryBuffer);
    const low = dataView.getUint32(ptr, true);  // 低32位
    const high = dataView.getUint32(ptr + 4, true);  // 高32位
    //const int64Value = BigInt(high) << 32n | BigInt(low);
    return {
        low: low,
        high: high
    };
}

function AllocGdObj() {
    return Module._gdspx_alloc_obj();
}

function PrintGdObj(ptr) {
    console.log(ToJsObj(ptr));
}

function FreeGdObj(ptr) {
    Module._gdspx_free_obj(ptr);
}

// Int-related functions
function ToGdInt(value) {
    return Module._gdspx_new_int(value.high, value.low);
}

function ToJsInt(ptr) {
    const memoryBuffer = Module.HEAPU8.buffer;
    const dataView = new DataView(memoryBuffer);
    const low = dataView.getUint32(ptr, true);  // 低32位
    const high = dataView.getUint32(ptr + 4, true);  // 高32位
    return {
        low: low,
        high: high
    };
}
function AllocGdInt() {
    return Module._gdspx_alloc_int();
}

function PrintGdInt(ptr) {
    console.log(ToJsInt(ptr));
}

function FreeGdInt(ptr) {
    Module._gdspx_free_int(ptr);
}


// Float-related functions
function ToGdFloat(value) {
    return Module._gdspx_new_float(value);
}

function ToJsFloat(ptr) {
    const HEAPF32 = Module.HEAPF32;
    const floatIndex = ptr / 4;
    const floatValue = HEAPF32[floatIndex];
    return floatValue;
}

function AllocGdFloat() {
    return Module._gdspx_alloc_float();
}

function PrintGdFloat(ptr) {
    console.log(ToJsFloat(ptr));
}

function FreeGdFloat(ptr) {
    Module._gdspx_free_float(ptr);
}

// String-related functions
function ToGdString(str) {
    const encoder = new TextEncoder();
    const stringBytes = encoder.encode(str);
    const ptr = Module._cmalloc(stringBytes.length + 1);
    Module.HEAPU8.set(stringBytes, ptr);
    Module.HEAPU8[ptr + stringBytes.length] = 0;
    const gdstrPtr = Module._gdspx_new_string(ptr, stringBytes.length);
    Module._cfree(ptr);
    return gdstrPtr;
}

function ToJsString(gdstrPtr) {
    return _toJsString(gdstrPtr, true);
}

function _toJsString(gdstrPtr, isFree) {
    const length = Module._gdspx_get_string_len(gdstrPtr)
    const ptr = Module._gdspx_get_string(gdstrPtr)
    const stringBytes = Module.HEAPU8.subarray(ptr, ptr + length);
    const nonSharedBytes = stringBytes.slice();
    const decoder = new TextDecoder("utf-8")
    const result = decoder.decode(nonSharedBytes)
    if (isFree) {
        Module._gdspx_free_cstr(ptr);
    }
    return result;
}


function AllocGdString() {
    return Module._gdspx_alloc_string();
}
function PrintGdString(ptr) {
    console.log(_toJsString(gdstrPtr, false));
}

function FreeGdString(ptr) {
    Module._gdspx_free_string(ptr);
}



// Vec2-related functions
function ToGdVec2(vec) {
    return Module._gdspx_new_vec2(vec.x, vec.y);
}

function ToJsVec2(ptr) {
    const HEAPF32 = Module.HEAPF32;
    const floatIndex = ptr / 4;
    return {
        x: HEAPF32[floatIndex],
        y: HEAPF32[floatIndex + 1]
    };
}

function AllocGdVec2() {
    return Module._gdspx_alloc_vec2();
}

function PrintGdVec2(ptr) {
    console.log(ToJsVec2(ptr));
}

function FreeGdVec2(ptr) {
    Module._gdspx_free_vec2(ptr);
}


// Vec3-related functions
function ToGdVec3(vec) {
    return Module._gdspx_new_vec3(vec.x, vec.y, vec.z);
}

function ToJsVec3(ptr) {
    const HEAPF32 = Module.HEAPF32;
    const floatIndex = ptr / 4;
    return {
        x: HEAPF32[floatIndex],
        y: HEAPF32[floatIndex + 1],
        z: HEAPF32[floatIndex + 2]
    };
}

function AllocGdVec3() {
    return Module._gdspx_alloc_vec3();
}

function PrintGdVec3(ptr) {
    const vec3 = ToJsVec3(ptr);
    console.log(`Vec3(${vec3.x}, ${vec3.y}, ${vec3.z})`);
}

function FreeGdVec3(ptr) {
    Module._gdspx_free_vec3(ptr);
}


// Vec4-related functions
function ToGdVec4(vec) {
    return Module._gdspx_new_vec4(vec.x, vec.y, vec.z, vec.w);
}

function ToJsVec4(ptr) {
    const HEAPF32 = Module.HEAPF32;
    const floatIndex = ptr / 4;
    return {
        x: HEAPF32[floatIndex],
        y: HEAPF32[floatIndex + 1],
        z: HEAPF32[floatIndex + 2],
        w: HEAPF32[floatIndex + 3]
    };
}

function AllocGdVec4() {
    return Module._gdspx_alloc_vec4();
}

function PrintGdVec4(ptr) {
    const vec4 = ToJsVec4(ptr);
    console.log(`Vec4(${vec4.x}, ${vec4.y}, ${vec4.z}, ${vec4.w})`);
}

function FreeGdVec4(ptr) {
    Module._gdspx_free_vec4(ptr);
}


// Color-related functions
function ToGdColor(color) {
    return Module._gdspx_new_color(color.r, color.g, color.b, color.a);
}

function ToJsColor(ptr) {
    const HEAPF32 = Module.HEAPF32;
    const floatIndex = ptr / 4;
    return {
        r: HEAPF32[floatIndex],
        g: HEAPF32[floatIndex + 1],
        b: HEAPF32[floatIndex + 2],
        a: HEAPF32[floatIndex + 3]
    };
}

function AllocGdColor() {
    return Module._gdspx_alloc_color();
}

function PrintGdColor(ptr) {
    const color = ToJsColor(ptr);
    console.log(`Color(${color.r}, ${color.g}, ${color.b}, ${color.a})`);
}

function FreeGdColor(ptr) {
    Module._gdspx_free_color(ptr);
}


// Rect2-related functions
function ToGdRect2(rect) {
    return Module._gdspx_new_rect2(rect.position.x, rect.position.y, rect.size.x, rect.size.y);
}

function ToJsRect2(ptr) {
    const HEAPF32 = Module.HEAPF32;
    const floatIndex = ptr / 4;
    return {
        position: {
            x: HEAPF32[floatIndex],
            y: HEAPF32[floatIndex + 1]
        },
        size: {
            x: HEAPF32[floatIndex + 2],
            y: HEAPF32[floatIndex + 3]
        }
    };
}

function AllocGdRect2() {
    return Module._gdspx_alloc_rect2();
}

function PrintGdRect2(ptr) {
    const rect = ToJsRect2(ptr);
    console.log(`Rect2(position: (${rect.position.x}, ${rect.position.y}), size: (${rect.size.width}, ${rect.size.height}))`);
}

function FreeGdRect2(ptr) {
    Module._gdspx_free_rect2(ptr);
}

function ToGdArray(array) {
    if (!array) {
        throw new Error('Invalid array structure. Expected {type, count, data}');
    }
    const dataSize = array.length;
    const dataPtr = Module._cmalloc(dataSize);
    try {
        Module.HEAPU8.set(array, dataPtr);
        const gdArrayPtr = Module._gdspx_to_gd_array(dataPtr, dataSize);
        return gdArrayPtr;
    } finally {
        Module._cfree(dataPtr);
    }
}

function ToJsArray(gdArrayPtr) {
    if (!gdArrayPtr) {
        return null;
    }
    const outputSizePtr = Module._cmalloc(4);
    try {
        const serializedPtr = Module._gdspx_to_js_array(gdArrayPtr, outputSizePtr);
        if (!serializedPtr) {
            console.log("ToJsArray serializedPtr == null");
            return null;
        }
        const outputSize = Module.HEAP32[outputSizePtr >> 2];
        const data = new Uint8Array(outputSize);
        data.set(Module.HEAPU8.subarray(serializedPtr, serializedPtr + outputSize));
        Module._cfree(serializedPtr);
        return data;
    } finally {
        Module._cfree(outputSizePtr);
    }
}


function AllocGdArray() {
    return Module._gdspx_alloc_array();
}

function PrintGdArray(ptr) {
    const val = ToJsArray(ptr);
    console.log(`Array: ${val}`);
}

function FreeGdArray(ptr) {
    Module._gdspx_free_array(ptr);
}
