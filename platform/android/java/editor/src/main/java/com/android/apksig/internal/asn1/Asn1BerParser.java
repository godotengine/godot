/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.apksig.internal.asn1;

import com.android.apksig.internal.asn1.ber.BerDataValue;
import com.android.apksig.internal.asn1.ber.BerDataValueFormatException;
import com.android.apksig.internal.asn1.ber.BerDataValueReader;
import com.android.apksig.internal.asn1.ber.BerEncoding;
import com.android.apksig.internal.asn1.ber.ByteBufferBerDataValueReader;
import com.android.apksig.internal.util.ByteBufferUtils;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Parser of ASN.1 BER-encoded structures.
 *
 * <p>Structure is described to the parser by providing a class annotated with {@link Asn1Class},
 * containing fields annotated with {@link Asn1Field}.
 */
public final class Asn1BerParser {
    private Asn1BerParser() {}

    /**
     * Returns the ASN.1 structure contained in the BER encoded input.
     *
     * @param encoded encoded input. If the decoding operation succeeds, the position of this buffer
     *        is advanced to the first position following the end of the consumed structure.
     * @param containerClass class describing the structure of the input. The class must meet the
     *        following requirements:
     *        <ul>
     *        <li>The class must be annotated with {@link Asn1Class}.</li>
     *        <li>The class must expose a public no-arg constructor.</li>
     *        <li>Member fields of the class which are populated with parsed input must be
     *            annotated with {@link Asn1Field} and be public and non-final.</li>
     *        </ul>
     *
     * @throws Asn1DecodingException if the input could not be decoded into the specified Java
     *         object
     */
    public static <T> T parse(ByteBuffer encoded, Class<T> containerClass)
            throws Asn1DecodingException {
        BerDataValue containerDataValue;
        try {
            containerDataValue = new ByteBufferBerDataValueReader(encoded).readDataValue();
        } catch (BerDataValueFormatException e) {
            throw new Asn1DecodingException("Failed to decode top-level data value", e);
        }
        if (containerDataValue == null) {
            throw new Asn1DecodingException("Empty input");
        }
        return parse(containerDataValue, containerClass);
    }

    /**
     * Returns the implicit {@code SET OF} contained in the provided ASN.1 BER input. Implicit means
     * that this method does not care whether the tag number of this data structure is
     * {@code SET OF} and whether the tag class is {@code UNIVERSAL}.
     *
     * <p>Note: The returned type is {@link List} rather than {@link java.util.Set} because ASN.1
     * SET may contain duplicate elements.
     *
     * @param encoded encoded input. If the decoding operation succeeds, the position of this buffer
     *        is advanced to the first position following the end of the consumed structure.
     * @param elementClass class describing the structure of the values/elements contained in this
     *        container. The class must meet the following requirements:
     *        <ul>
     *        <li>The class must be annotated with {@link Asn1Class}.</li>
     *        <li>The class must expose a public no-arg constructor.</li>
     *        <li>Member fields of the class which are populated with parsed input must be
     *            annotated with {@link Asn1Field} and be public and non-final.</li>
     *        </ul>
     *
     * @throws Asn1DecodingException if the input could not be decoded into the specified Java
     *         object
     */
    public static <T> List<T> parseImplicitSetOf(ByteBuffer encoded, Class<T> elementClass)
            throws Asn1DecodingException {
        BerDataValue containerDataValue;
        try {
            containerDataValue = new ByteBufferBerDataValueReader(encoded).readDataValue();
        } catch (BerDataValueFormatException e) {
            throw new Asn1DecodingException("Failed to decode top-level data value", e);
        }
        if (containerDataValue == null) {
            throw new Asn1DecodingException("Empty input");
        }
        return parseSetOf(containerDataValue, elementClass);
    }

    private static <T> T parse(BerDataValue container, Class<T> containerClass)
            throws Asn1DecodingException {
        if (container == null) {
            throw new NullPointerException("container == null");
        }
        if (containerClass == null) {
            throw new NullPointerException("containerClass == null");
        }

        Asn1Type dataType = getContainerAsn1Type(containerClass);
        switch (dataType) {
            case CHOICE:
                return parseChoice(container, containerClass);

            case SEQUENCE:
            {
                int expectedTagClass = BerEncoding.TAG_CLASS_UNIVERSAL;
                int expectedTagNumber = BerEncoding.getTagNumber(dataType);
                if ((container.getTagClass() != expectedTagClass)
                        || (container.getTagNumber() != expectedTagNumber)) {
                    throw new Asn1UnexpectedTagException(
                            "Unexpected data value read as " + containerClass.getName()
                                    + ". Expected " + BerEncoding.tagClassAndNumberToString(
                                    expectedTagClass, expectedTagNumber)
                                    + ", but read: " + BerEncoding.tagClassAndNumberToString(
                                    container.getTagClass(), container.getTagNumber()));
                }
                return parseSequence(container, containerClass);
            }
            case UNENCODED_CONTAINER:
                return parseSequence(container, containerClass, true);
            default:
                throw new Asn1DecodingException("Parsing container " + dataType + " not supported");
        }
    }

    private static <T> T parseChoice(BerDataValue dataValue, Class<T> containerClass)
            throws Asn1DecodingException {
        List<AnnotatedField> fields = getAnnotatedFields(containerClass);
        if (fields.isEmpty()) {
            throw new Asn1DecodingException(
                    "No fields annotated with " + Asn1Field.class.getName()
                            + " in CHOICE class " + containerClass.getName());
        }

        // Check that class + tagNumber don't clash between the choices
        for (int i = 0; i < fields.size() - 1; i++) {
            AnnotatedField f1 = fields.get(i);
            int tagNumber1 = f1.getBerTagNumber();
            int tagClass1 = f1.getBerTagClass();
            for (int j = i + 1; j < fields.size(); j++) {
                AnnotatedField f2 = fields.get(j);
                int tagNumber2 = f2.getBerTagNumber();
                int tagClass2 = f2.getBerTagClass();
                if ((tagNumber1 == tagNumber2) && (tagClass1 == tagClass2)) {
                    throw new Asn1DecodingException(
                            "CHOICE fields are indistinguishable because they have the same tag"
                                    + " class and number: " + containerClass.getName()
                                    + "." + f1.getField().getName()
                                    + " and ." + f2.getField().getName());
                }
            }
        }

        // Instantiate the container object / result
        T obj;
        try {
            obj = containerClass.getConstructor().newInstance();
        } catch (IllegalArgumentException | ReflectiveOperationException e) {
            throw new Asn1DecodingException("Failed to instantiate " + containerClass.getName(), e);
        }
        // Set the matching field's value from the data value
        for (AnnotatedField field : fields) {
            try {
                field.setValueFrom(dataValue, obj);
                return obj;
            } catch (Asn1UnexpectedTagException expected) {
                // not a match
            }
        }

        throw new Asn1DecodingException(
                "No options of CHOICE " + containerClass.getName() + " matched");
    }

    private static <T> T parseSequence(BerDataValue container, Class<T> containerClass)
            throws Asn1DecodingException {
        return parseSequence(container, containerClass, false);
    }

    private static <T> T parseSequence(BerDataValue container, Class<T> containerClass,
            boolean isUnencodedContainer) throws Asn1DecodingException {
        List<AnnotatedField> fields = getAnnotatedFields(containerClass);
        Collections.sort(
                fields, (f1, f2) -> f1.getAnnotation().index() - f2.getAnnotation().index());
        // Check that there are no fields with the same index
        if (fields.size() > 1) {
            AnnotatedField lastField = null;
            for (AnnotatedField field : fields) {
                if ((lastField != null)
                        && (lastField.getAnnotation().index() == field.getAnnotation().index())) {
                    throw new Asn1DecodingException(
                            "Fields have the same index: " + containerClass.getName()
                                    + "." + lastField.getField().getName()
                                    + " and ." + field.getField().getName());
                }
                lastField = field;
            }
        }

        // Instantiate the container object / result
        T t;
        try {
            t = containerClass.getConstructor().newInstance();
        } catch (IllegalArgumentException | ReflectiveOperationException e) {
            throw new Asn1DecodingException("Failed to instantiate " + containerClass.getName(), e);
        }

        // Parse fields one by one. A complication is that there may be optional fields.
        int nextUnreadFieldIndex = 0;
        BerDataValueReader elementsReader = container.contentsReader();
        while (nextUnreadFieldIndex < fields.size()) {
            BerDataValue dataValue;
            try {
                // if this is the first field of an unencoded container then the entire contents of
                // the container should be used when assigning to this field.
                if (isUnencodedContainer && nextUnreadFieldIndex == 0) {
                    dataValue = container;
                } else {
                    dataValue = elementsReader.readDataValue();
                }
            } catch (BerDataValueFormatException e) {
                throw new Asn1DecodingException("Malformed data value", e);
            }
            if (dataValue == null) {
                break;
            }

            for (int i = nextUnreadFieldIndex; i < fields.size(); i++) {
                AnnotatedField field = fields.get(i);
                try {
                    if (field.isOptional()) {
                        // Optional field -- might not be present and we may thus be trying to set
                        // it from the wrong tag.
                        try {
                            field.setValueFrom(dataValue, t);
                            nextUnreadFieldIndex = i + 1;
                            break;
                        } catch (Asn1UnexpectedTagException e) {
                            // This field is not present, attempt to use this data value for the
                            // next / iteration of the loop
                            continue;
                        }
                    } else {
                        // Mandatory field -- if we can't set its value from this data value, then
                        // it's an error
                        field.setValueFrom(dataValue, t);
                        nextUnreadFieldIndex = i + 1;
                        break;
                    }
                } catch (Asn1DecodingException e) {
                    throw new Asn1DecodingException(
                            "Failed to parse " + containerClass.getName()
                                    + "." + field.getField().getName(),
                            e);
                }
            }
        }

        return t;
    }

    // NOTE: This method returns List rather than Set because ASN.1 SET_OF does require uniqueness
    // of elements -- it's an unordered collection.
    @SuppressWarnings("unchecked")
    private static <T> List<T> parseSetOf(BerDataValue container, Class<T> elementClass)
            throws Asn1DecodingException {
        List<T> result = new ArrayList<>();
        BerDataValueReader elementsReader = container.contentsReader();
        while (true) {
            BerDataValue dataValue;
            try {
                dataValue = elementsReader.readDataValue();
            } catch (BerDataValueFormatException e) {
                throw new Asn1DecodingException("Malformed data value", e);
            }
            if (dataValue == null) {
                break;
            }
            T element;
            if (ByteBuffer.class.equals(elementClass)) {
                element = (T) dataValue.getEncodedContents();
            } else if (Asn1OpaqueObject.class.equals(elementClass)) {
                element = (T) new Asn1OpaqueObject(dataValue.getEncoded());
            } else {
                element = parse(dataValue, elementClass);
            }
            result.add(element);
        }
        return result;
    }

    private static Asn1Type getContainerAsn1Type(Class<?> containerClass)
            throws Asn1DecodingException {
        Asn1Class containerAnnotation = containerClass.getDeclaredAnnotation(Asn1Class.class);
        if (containerAnnotation == null) {
            throw new Asn1DecodingException(
                    containerClass.getName() + " is not annotated with "
                            + Asn1Class.class.getName());
        }

        switch (containerAnnotation.type()) {
            case CHOICE:
            case SEQUENCE:
            case UNENCODED_CONTAINER:
                return containerAnnotation.type();
            default:
                throw new Asn1DecodingException(
                        "Unsupported ASN.1 container annotation type: "
                                + containerAnnotation.type());
        }
    }

    private static Class<?> getElementType(Field field)
            throws Asn1DecodingException, ClassNotFoundException {
        String type = field.getGenericType().getTypeName();
        int delimiterIndex =  type.indexOf('<');
        if (delimiterIndex == -1) {
            throw new Asn1DecodingException("Not a container type: " + field.getGenericType());
        }
        int startIndex = delimiterIndex + 1;
        int endIndex = type.indexOf('>', startIndex);
        // TODO: handle comma?
        if (endIndex == -1) {
            throw new Asn1DecodingException("Not a container type: " + field.getGenericType());
        }
        String elementClassName = type.substring(startIndex, endIndex);
        return Class.forName(elementClassName);
    }

    private static final class AnnotatedField {
        private final Field mField;
        private final Asn1Field mAnnotation;
        private final Asn1Type mDataType;
        private final Asn1TagClass mTagClass;
        private final int mBerTagClass;
        private final int mBerTagNumber;
        private final Asn1Tagging mTagging;
        private final boolean mOptional;

        public AnnotatedField(Field field, Asn1Field annotation) throws Asn1DecodingException {
            mField = field;
            mAnnotation = annotation;
            mDataType = annotation.type();

            Asn1TagClass tagClass = annotation.cls();
            if (tagClass == Asn1TagClass.AUTOMATIC) {
                if (annotation.tagNumber() != -1) {
                    tagClass = Asn1TagClass.CONTEXT_SPECIFIC;
                } else {
                    tagClass = Asn1TagClass.UNIVERSAL;
                }
            }
            mTagClass = tagClass;
            mBerTagClass = BerEncoding.getTagClass(mTagClass);

            int tagNumber;
            if (annotation.tagNumber() != -1) {
                tagNumber = annotation.tagNumber();
            } else if ((mDataType == Asn1Type.CHOICE) || (mDataType == Asn1Type.ANY)) {
                tagNumber = -1;
            } else {
                tagNumber = BerEncoding.getTagNumber(mDataType);
            }
            mBerTagNumber = tagNumber;

            mTagging = annotation.tagging();
            if (((mTagging == Asn1Tagging.EXPLICIT) || (mTagging == Asn1Tagging.IMPLICIT))
                    && (annotation.tagNumber() == -1)) {
                throw new Asn1DecodingException(
                        "Tag number must be specified when tagging mode is " + mTagging);
            }

            mOptional = annotation.optional();
        }

        public Field getField() {
            return mField;
        }

        public Asn1Field getAnnotation() {
            return mAnnotation;
        }

        public boolean isOptional() {
            return mOptional;
        }

        public int getBerTagClass() {
            return mBerTagClass;
        }

        public int getBerTagNumber() {
            return mBerTagNumber;
        }

        public void setValueFrom(BerDataValue dataValue, Object obj) throws Asn1DecodingException {
            int readTagClass = dataValue.getTagClass();
            if (mBerTagNumber != -1) {
                int readTagNumber = dataValue.getTagNumber();
                if ((readTagClass != mBerTagClass) || (readTagNumber != mBerTagNumber)) {
                    throw new Asn1UnexpectedTagException(
                            "Tag mismatch. Expected: "
                            + BerEncoding.tagClassAndNumberToString(mBerTagClass, mBerTagNumber)
                            + ", but found "
                            + BerEncoding.tagClassAndNumberToString(readTagClass, readTagNumber));
                }
            } else {
                if (readTagClass != mBerTagClass) {
                    throw new Asn1UnexpectedTagException(
                            "Tag mismatch. Expected class: "
                            + BerEncoding.tagClassToString(mBerTagClass)
                            + ", but found "
                            + BerEncoding.tagClassToString(readTagClass));
                }
            }

            if (mTagging == Asn1Tagging.EXPLICIT) {
                try {
                    dataValue = dataValue.contentsReader().readDataValue();
                } catch (BerDataValueFormatException e) {
                    throw new Asn1DecodingException(
                            "Failed to read contents of EXPLICIT data value", e);
                }
            }

            BerToJavaConverter.setFieldValue(obj, mField, mDataType, dataValue);
        }
    }

    private static class Asn1UnexpectedTagException extends Asn1DecodingException {
        private static final long serialVersionUID = 1L;

        public Asn1UnexpectedTagException(String message) {
            super(message);
        }
    }

    private static String oidToString(ByteBuffer encodedOid) throws Asn1DecodingException {
        if (!encodedOid.hasRemaining()) {
            throw new Asn1DecodingException("Empty OBJECT IDENTIFIER");
        }

        // First component encodes the first two nodes, X.Y, as X * 40 + Y, with 0 <= X <= 2
        long firstComponent = decodeBase128UnsignedLong(encodedOid);
        int firstNode = (int) Math.min(firstComponent / 40, 2);
        long secondNode = firstComponent - firstNode * 40;
        StringBuilder result = new StringBuilder();
        result.append(Long.toString(firstNode)).append('.')
                .append(Long.toString(secondNode));

        // Each consecutive node is encoded as a separate component
        while (encodedOid.hasRemaining()) {
            long node = decodeBase128UnsignedLong(encodedOid);
            result.append('.').append(Long.toString(node));
        }

        return result.toString();
    }

    private static long decodeBase128UnsignedLong(ByteBuffer encoded) throws Asn1DecodingException {
        if (!encoded.hasRemaining()) {
            return 0;
        }
        long result = 0;
        while (encoded.hasRemaining()) {
            if (result > Long.MAX_VALUE >>> 7) {
                throw new Asn1DecodingException("Base-128 number too large");
            }
            int b = encoded.get() & 0xff;
            result <<= 7;
            result |= b & 0x7f;
            if ((b & 0x80) == 0) {
                return result;
            }
        }
        throw new Asn1DecodingException(
                "Truncated base-128 encoded input: missing terminating byte, with highest bit not"
                        + " set");
    }

    private static BigInteger integerToBigInteger(ByteBuffer encoded) {
        if (!encoded.hasRemaining()) {
            return BigInteger.ZERO;
        }
        return new BigInteger(ByteBufferUtils.toByteArray(encoded));
    }

    private static int integerToInt(ByteBuffer encoded) throws Asn1DecodingException {
        BigInteger value = integerToBigInteger(encoded);
        if (value.compareTo(BigInteger.valueOf(Integer.MIN_VALUE)) < 0
            || value.compareTo(BigInteger.valueOf(Integer.MAX_VALUE)) > 0) {
            throw new Asn1DecodingException(
                String.format("INTEGER cannot be represented as int: %1$d (0x%1$x)", value));
        }
        return value.intValue();
    }

    private static long integerToLong(ByteBuffer encoded) throws Asn1DecodingException {
        BigInteger value = integerToBigInteger(encoded);
        if (value.compareTo(BigInteger.valueOf(Long.MIN_VALUE)) < 0
                || value.compareTo(BigInteger.valueOf(Long.MAX_VALUE)) > 0) {
            throw new Asn1DecodingException(
                String.format("INTEGER cannot be represented as long: %1$d (0x%1$x)", value));
        }
        return value.longValue();
    }

    private static List<AnnotatedField> getAnnotatedFields(Class<?> containerClass)
            throws Asn1DecodingException {
        Field[] declaredFields = containerClass.getDeclaredFields();
        List<AnnotatedField> result = new ArrayList<>(declaredFields.length);
        for (Field field : declaredFields) {
            Asn1Field annotation = field.getDeclaredAnnotation(Asn1Field.class);
            if (annotation == null) {
                continue;
            }
            if (Modifier.isStatic(field.getModifiers())) {
                throw new Asn1DecodingException(
                        Asn1Field.class.getName() + " used on a static field: "
                                + containerClass.getName() + "." + field.getName());
            }

            AnnotatedField annotatedField;
            try {
                annotatedField = new AnnotatedField(field, annotation);
            } catch (Asn1DecodingException e) {
                throw new Asn1DecodingException(
                        "Invalid ASN.1 annotation on "
                                + containerClass.getName() + "." + field.getName(),
                        e);
            }
            result.add(annotatedField);
        }
        return result;
    }

    private static final class BerToJavaConverter {
        private BerToJavaConverter() {}

        public static void setFieldValue(
                Object obj, Field field, Asn1Type type, BerDataValue dataValue)
                        throws Asn1DecodingException {
            try {
                switch (type) {
                    case SET_OF:
                    case SEQUENCE_OF:
                        if (Asn1OpaqueObject.class.equals(field.getType())) {
                            field.set(obj, convert(type, dataValue, field.getType()));
                        } else {
                            field.set(obj, parseSetOf(dataValue, getElementType(field)));
                        }
                        return;
                    default:
                        field.set(obj, convert(type, dataValue, field.getType()));
                        break;
                }
            } catch (ReflectiveOperationException e) {
                throw new Asn1DecodingException(
                        "Failed to set value of " + obj.getClass().getName()
                                + "." + field.getName(),
                        e);
            }
        }

        private static final byte[] EMPTY_BYTE_ARRAY = new byte[0];

        @SuppressWarnings("unchecked")
        public static <T> T convert(
                Asn1Type sourceType,
                BerDataValue dataValue,
                Class<T> targetType) throws Asn1DecodingException {
            if (ByteBuffer.class.equals(targetType)) {
                return (T) dataValue.getEncodedContents();
            } else if (byte[].class.equals(targetType)) {
                ByteBuffer resultBuf = dataValue.getEncodedContents();
                if (!resultBuf.hasRemaining()) {
                    return (T) EMPTY_BYTE_ARRAY;
                }
                byte[] result = new byte[resultBuf.remaining()];
                resultBuf.get(result);
                return (T) result;
            } else if (Asn1OpaqueObject.class.equals(targetType)) {
                return (T) new Asn1OpaqueObject(dataValue.getEncoded());
            }
            ByteBuffer encodedContents = dataValue.getEncodedContents();
            switch (sourceType) {
                case INTEGER:
                    if ((int.class.equals(targetType)) || (Integer.class.equals(targetType))) {
                        return (T) Integer.valueOf(integerToInt(encodedContents));
                    } else if ((long.class.equals(targetType)) || (Long.class.equals(targetType))) {
                        return (T) Long.valueOf(integerToLong(encodedContents));
                    } else if (BigInteger.class.equals(targetType)) {
                        return (T) integerToBigInteger(encodedContents);
                    }
                    break;
                case OBJECT_IDENTIFIER:
                    if (String.class.equals(targetType)) {
                        return (T) oidToString(encodedContents);
                    }
                    break;
                case UTC_TIME:
                case GENERALIZED_TIME:
                    if (String.class.equals(targetType)) {
                        return (T) new String(ByteBufferUtils.toByteArray(encodedContents));
                    }
                    break;
                case BOOLEAN:
                    // A boolean should be encoded in a single byte with a value of 0 for false and
                    // any non-zero value for true.
                    if (boolean.class.equals(targetType)) {
                        if (encodedContents.remaining() != 1) {
                            throw new Asn1DecodingException(
                                    "Incorrect encoded size of boolean value: "
                                            + encodedContents.remaining());
                        }
                        boolean result;
                        if (encodedContents.get() == 0) {
                            result = false;
                        } else {
                            result = true;
                        }
                        return (T) new Boolean(result);
                    }
                    break;
                case SEQUENCE:
                {
                    Asn1Class containerAnnotation =
                            targetType.getDeclaredAnnotation(Asn1Class.class);
                    if ((containerAnnotation != null)
                            && (containerAnnotation.type() == Asn1Type.SEQUENCE)) {
                        return parseSequence(dataValue, targetType);
                    }
                    break;
                }
                case CHOICE:
                {
                    Asn1Class containerAnnotation =
                            targetType.getDeclaredAnnotation(Asn1Class.class);
                    if ((containerAnnotation != null)
                            && (containerAnnotation.type() == Asn1Type.CHOICE)) {
                        return parseChoice(dataValue, targetType);
                    }
                    break;
                }
                default:
                    break;
            }

            throw new Asn1DecodingException(
                    "Unsupported conversion: ASN.1 " + sourceType + " to " + targetType.getName());
        }
    }
}
