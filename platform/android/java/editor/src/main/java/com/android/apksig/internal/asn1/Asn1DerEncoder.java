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

import com.android.apksig.internal.asn1.ber.BerEncoding;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Encoder of ASN.1 structures into DER-encoded form.
 *
 * <p>Structure is described to the encoder by providing a class annotated with {@link Asn1Class},
 * containing fields annotated with {@link Asn1Field}.
 */
public final class Asn1DerEncoder {
    private Asn1DerEncoder() {}

    /**
     * Returns the DER-encoded form of the provided ASN.1 structure.
     *
     * @param container container to be encoded. The container's class must meet the following
     *        requirements:
     *        <ul>
     *        <li>The class must be annotated with {@link Asn1Class}.</li>
     *        <li>Member fields of the class which are to be encoded must be annotated with
     *            {@link Asn1Field} and be public.</li>
     *        </ul>
     *
     * @throws Asn1EncodingException if the input could not be encoded
     */
    public static byte[] encode(Object container) throws Asn1EncodingException {
        Class<?> containerClass = container.getClass();
        Asn1Class containerAnnotation = containerClass.getDeclaredAnnotation(Asn1Class.class);
        if (containerAnnotation == null) {
            throw new Asn1EncodingException(
                    containerClass.getName() + " not annotated with " + Asn1Class.class.getName());
        }

        Asn1Type containerType = containerAnnotation.type();
        switch (containerType) {
            case CHOICE:
                return toChoice(container);
            case SEQUENCE:
                return toSequence(container);
            case UNENCODED_CONTAINER:
                return toSequence(container, true);
            default:
                throw new Asn1EncodingException("Unsupported container type: " + containerType);
        }
    }

    private static byte[] toChoice(Object container) throws Asn1EncodingException {
        Class<?> containerClass = container.getClass();
        List<AnnotatedField> fields = getAnnotatedFields(container);
        if (fields.isEmpty()) {
            throw new Asn1EncodingException(
                    "No fields annotated with " + Asn1Field.class.getName()
                            + " in CHOICE class " + containerClass.getName());
        }

        AnnotatedField resultField = null;
        for (AnnotatedField field : fields) {
            Object fieldValue = getMemberFieldValue(container, field.getField());
            if (fieldValue != null) {
                if (resultField != null) {
                    throw new Asn1EncodingException(
                            "Multiple non-null fields in CHOICE class " + containerClass.getName()
                                    + ": " + resultField.getField().getName()
                                    + ", " + field.getField().getName());
                }
                resultField = field;
            }
        }

        if (resultField == null) {
            throw new Asn1EncodingException(
                    "No non-null fields in CHOICE class " + containerClass.getName());
        }

        return resultField.toDer();
    }

    private static byte[] toSequence(Object container) throws Asn1EncodingException {
        return toSequence(container, false);
    }

    private static byte[] toSequence(Object container, boolean omitTag)
            throws Asn1EncodingException {
        Class<?> containerClass = container.getClass();
        List<AnnotatedField> fields = getAnnotatedFields(container);
        Collections.sort(
                fields, (f1, f2) -> f1.getAnnotation().index() - f2.getAnnotation().index());
        if (fields.size() > 1) {
            AnnotatedField lastField = null;
            for (AnnotatedField field : fields) {
                if ((lastField != null)
                        && (lastField.getAnnotation().index() == field.getAnnotation().index())) {
                    throw new Asn1EncodingException(
                            "Fields have the same index: " + containerClass.getName()
                                    + "." + lastField.getField().getName()
                                    + " and ." + field.getField().getName());
                }
                lastField = field;
            }
        }

        List<byte[]> serializedFields = new ArrayList<>(fields.size());
        int contentLen = 0;
        for (AnnotatedField field : fields) {
            byte[] serializedField;
            try {
                serializedField = field.toDer();
            } catch (Asn1EncodingException e) {
                throw new Asn1EncodingException(
                        "Failed to encode " + containerClass.getName()
                                + "." + field.getField().getName(),
                        e);
            }
            if (serializedField != null) {
                serializedFields.add(serializedField);
                contentLen += serializedField.length;
            }
        }

        if (omitTag) {
            byte[] unencodedResult = new byte[contentLen];
            int index = 0;
            for (byte[] serializedField : serializedFields) {
                System.arraycopy(serializedField, 0, unencodedResult, index, serializedField.length);
                index += serializedField.length;
            }
            return unencodedResult;
        } else {
            return createTag(
                    BerEncoding.TAG_CLASS_UNIVERSAL, true, BerEncoding.TAG_NUMBER_SEQUENCE,
                    serializedFields.toArray(new byte[0][]));
        }
    }

    private static byte[] toSetOf(Collection<?> values, Asn1Type elementType) throws Asn1EncodingException {
        return toSequenceOrSetOf(values, elementType, true);
    }

    private static byte[] toSequenceOf(Collection<?> values, Asn1Type elementType) throws Asn1EncodingException {
        return toSequenceOrSetOf(values, elementType, false);
    }

    private static byte[] toSequenceOrSetOf(Collection<?> values, Asn1Type elementType, boolean toSet)
            throws Asn1EncodingException {
        List<byte[]> serializedValues = new ArrayList<>(values.size());
        for (Object value : values) {
            serializedValues.add(JavaToDerConverter.toDer(value, elementType, null));
        }
        int tagNumber;
        if (toSet) {
            if (serializedValues.size() > 1) {
                Collections.sort(serializedValues, ByteArrayLexicographicComparator.INSTANCE);
            }
            tagNumber = BerEncoding.TAG_NUMBER_SET;
        } else {
            tagNumber = BerEncoding.TAG_NUMBER_SEQUENCE;
        }
        return createTag(
                BerEncoding.TAG_CLASS_UNIVERSAL, true, tagNumber,
                serializedValues.toArray(new byte[0][]));
    }

    /**
     * Compares two bytes arrays based on their lexicographic order. Corresponding elements of the
     * two arrays are compared in ascending order. Elements at out of range indices are assumed to
     * be smaller than the smallest possible value for an element.
     */
    private static class ByteArrayLexicographicComparator implements Comparator<byte[]> {
            private static final ByteArrayLexicographicComparator INSTANCE =
                    new ByteArrayLexicographicComparator();

            @Override
            public int compare(byte[] arr1, byte[] arr2) {
                int commonLength = Math.min(arr1.length, arr2.length);
                for (int i = 0; i < commonLength; i++) {
                    int diff = (arr1[i] & 0xff) - (arr2[i] & 0xff);
                    if (diff != 0) {
                        return diff;
                    }
                }
                return arr1.length - arr2.length;
            }
    }

    private static List<AnnotatedField> getAnnotatedFields(Object container)
            throws Asn1EncodingException {
        Class<?> containerClass = container.getClass();
        Field[] declaredFields = containerClass.getDeclaredFields();
        List<AnnotatedField> result = new ArrayList<>(declaredFields.length);
        for (Field field : declaredFields) {
            Asn1Field annotation = field.getDeclaredAnnotation(Asn1Field.class);
            if (annotation == null) {
                continue;
            }
            if (Modifier.isStatic(field.getModifiers())) {
                throw new Asn1EncodingException(
                        Asn1Field.class.getName() + " used on a static field: "
                                + containerClass.getName() + "." + field.getName());
            }

            AnnotatedField annotatedField;
            try {
                annotatedField = new AnnotatedField(container, field, annotation);
            } catch (Asn1EncodingException e) {
                throw new Asn1EncodingException(
                        "Invalid ASN.1 annotation on "
                                + containerClass.getName() + "." + field.getName(),
                        e);
            }
            result.add(annotatedField);
        }
        return result;
    }

    private static byte[] toInteger(int value) {
        return toInteger((long) value);
    }

    private static byte[] toInteger(long value) {
        return toInteger(BigInteger.valueOf(value));
    }

    private static byte[] toInteger(BigInteger value) {
        return createTag(
                BerEncoding.TAG_CLASS_UNIVERSAL, false, BerEncoding.TAG_NUMBER_INTEGER,
                value.toByteArray());
    }

    private static byte[] toBoolean(boolean value) {
        // A boolean should be encoded in a single byte with a value of 0 for false and any non-zero
        // value for true.
        byte[] result = new byte[1];
        if (value == false) {
            result[0] = 0;
        } else {
            result[0] = 1;
        }
        return createTag(BerEncoding.TAG_CLASS_UNIVERSAL, false, BerEncoding.TAG_NUMBER_BOOLEAN, result);
    }

    private static byte[] toOid(String oid) throws Asn1EncodingException {
        ByteArrayOutputStream encodedValue = new ByteArrayOutputStream();
        String[] nodes = oid.split("\\.");
        if (nodes.length < 2) {
            throw new Asn1EncodingException(
                    "OBJECT IDENTIFIER must contain at least two nodes: " + oid);
        }
        int firstNode;
        try {
            firstNode = Integer.parseInt(nodes[0]);
        } catch (NumberFormatException e) {
            throw new Asn1EncodingException("Node #1 not numeric: " + nodes[0]);
        }
        if ((firstNode > 6) || (firstNode < 0)) {
            throw new Asn1EncodingException("Invalid value for node #1: " + firstNode);
        }

        int secondNode;
        try {
            secondNode = Integer.parseInt(nodes[1]);
        } catch (NumberFormatException e) {
            throw new Asn1EncodingException("Node #2 not numeric: " + nodes[1]);
        }
        if ((secondNode >= 40) || (secondNode < 0)) {
            throw new Asn1EncodingException("Invalid value for node #2: " + secondNode);
        }
        int firstByte = firstNode * 40 + secondNode;
        if (firstByte > 0xff) {
            throw new Asn1EncodingException(
                    "First two nodes out of range: " + firstNode + "." + secondNode);
        }

        encodedValue.write(firstByte);
        for (int i = 2; i < nodes.length; i++) {
            String nodeString = nodes[i];
            int node;
            try {
                node = Integer.parseInt(nodeString);
            } catch (NumberFormatException e) {
                throw new Asn1EncodingException("Node #" + (i + 1) + " not numeric: " + nodeString);
            }
            if (node < 0) {
                throw new Asn1EncodingException("Invalid value for node #" + (i + 1) + ": " + node);
            }
            if (node <= 0x7f) {
                encodedValue.write(node);
                continue;
            }
            if (node < 1 << 14) {
                encodedValue.write(0x80 | (node >> 7));
                encodedValue.write(node & 0x7f);
                continue;
            }
            if (node < 1 << 21) {
                encodedValue.write(0x80 | (node >> 14));
                encodedValue.write(0x80 | ((node >> 7) & 0x7f));
                encodedValue.write(node & 0x7f);
                continue;
            }
            throw new Asn1EncodingException("Node #" + (i + 1) + " too large: " + node);
        }

        return createTag(
                BerEncoding.TAG_CLASS_UNIVERSAL, false, BerEncoding.TAG_NUMBER_OBJECT_IDENTIFIER,
                encodedValue.toByteArray());
    }

    private static Object getMemberFieldValue(Object obj, Field field)
            throws Asn1EncodingException {
        try {
            return field.get(obj);
        } catch (ReflectiveOperationException e) {
            throw new Asn1EncodingException(
                    "Failed to read " + obj.getClass().getName() + "." + field.getName(), e);
        }
    }

    private static final class AnnotatedField {
        private final Field mField;
        private final Object mObject;
        private final Asn1Field mAnnotation;
        private final Asn1Type mDataType;
        private final Asn1Type mElementDataType;
        private final Asn1TagClass mTagClass;
        private final int mDerTagClass;
        private final int mDerTagNumber;
        private final Asn1Tagging mTagging;
        private final boolean mOptional;

        public AnnotatedField(Object obj, Field field, Asn1Field annotation)
                throws Asn1EncodingException {
            mObject = obj;
            mField = field;
            mAnnotation = annotation;
            mDataType = annotation.type();
            mElementDataType = annotation.elementType();

            Asn1TagClass tagClass = annotation.cls();
            if (tagClass == Asn1TagClass.AUTOMATIC) {
                if (annotation.tagNumber() != -1) {
                    tagClass = Asn1TagClass.CONTEXT_SPECIFIC;
                } else {
                    tagClass = Asn1TagClass.UNIVERSAL;
                }
            }
            mTagClass = tagClass;
            mDerTagClass = BerEncoding.getTagClass(mTagClass);

            int tagNumber;
            if (annotation.tagNumber() != -1) {
                tagNumber = annotation.tagNumber();
            } else if ((mDataType == Asn1Type.CHOICE) || (mDataType == Asn1Type.ANY)) {
                tagNumber = -1;
            } else {
                tagNumber = BerEncoding.getTagNumber(mDataType);
            }
            mDerTagNumber = tagNumber;

            mTagging = annotation.tagging();
            if (((mTagging == Asn1Tagging.EXPLICIT) || (mTagging == Asn1Tagging.IMPLICIT))
                    && (annotation.tagNumber() == -1)) {
                throw new Asn1EncodingException(
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

        public byte[] toDer() throws Asn1EncodingException {
            Object fieldValue = getMemberFieldValue(mObject, mField);
            if (fieldValue == null) {
                if (mOptional) {
                    return null;
                }
                throw new Asn1EncodingException("Required field not set");
            }

            byte[] encoded = JavaToDerConverter.toDer(fieldValue, mDataType, mElementDataType);
            switch (mTagging) {
                case NORMAL:
                    return encoded;
                case EXPLICIT:
                    return createTag(mDerTagClass, true, mDerTagNumber, encoded);
                case IMPLICIT:
                    int originalTagNumber = BerEncoding.getTagNumber(encoded[0]);
                    if (originalTagNumber == 0x1f) {
                        throw new Asn1EncodingException("High-tag-number form not supported");
                    }
                    if (mDerTagNumber >= 0x1f) {
                        throw new Asn1EncodingException(
                                "Unsupported high tag number: " + mDerTagNumber);
                    }
                    encoded[0] = BerEncoding.setTagNumber(encoded[0], mDerTagNumber);
                    encoded[0] = BerEncoding.setTagClass(encoded[0], mDerTagClass);
                    return encoded;
                default:
                    throw new RuntimeException("Unknown tagging mode: " + mTagging);
            }
        }
    }

    private static byte[] createTag(
            int tagClass, boolean constructed, int tagNumber, byte[]... contents) {
        if (tagNumber >= 0x1f) {
            throw new IllegalArgumentException("High tag numbers not supported: " + tagNumber);
        }
        // tag class & number fit into the first byte
        byte firstIdentifierByte =
                (byte) ((tagClass << 6) | (constructed ? 1 << 5 : 0) | tagNumber);

        int contentsLength = 0;
        for (byte[] c : contents) {
            contentsLength += c.length;
        }
        int contentsPosInResult;
        byte[] result;
        if (contentsLength < 0x80) {
            // Length fits into one byte
            contentsPosInResult = 2;
            result = new byte[contentsPosInResult + contentsLength];
            result[0] = firstIdentifierByte;
            result[1] = (byte) contentsLength;
        } else {
            // Length is represented as multiple bytes
            // The low 7 bits of the first byte represent the number of length bytes (following the
            // first byte) in which the length is in big-endian base-256 form
            if (contentsLength <= 0xff) {
                contentsPosInResult = 3;
                result = new byte[contentsPosInResult + contentsLength];
                result[1] = (byte) 0x81; // 1 length byte
                result[2] = (byte) contentsLength;
            } else if (contentsLength <= 0xffff) {
                contentsPosInResult = 4;
                result = new byte[contentsPosInResult + contentsLength];
                result[1] = (byte) 0x82; // 2 length bytes
                result[2] = (byte) (contentsLength >> 8);
                result[3] = (byte) (contentsLength & 0xff);
            } else if (contentsLength <= 0xffffff) {
                contentsPosInResult = 5;
                result = new byte[contentsPosInResult + contentsLength];
                result[1] = (byte) 0x83; // 3 length bytes
                result[2] = (byte) (contentsLength >> 16);
                result[3] = (byte) ((contentsLength >> 8) & 0xff);
                result[4] = (byte) (contentsLength & 0xff);
            } else {
                contentsPosInResult = 6;
                result = new byte[contentsPosInResult + contentsLength];
                result[1] = (byte) 0x84; // 4 length bytes
                result[2] = (byte) (contentsLength >> 24);
                result[3] = (byte) ((contentsLength >> 16) & 0xff);
                result[4] = (byte) ((contentsLength >> 8) & 0xff);
                result[5] = (byte) (contentsLength & 0xff);
            }
            result[0] = firstIdentifierByte;
        }
        for (byte[] c : contents) {
            System.arraycopy(c, 0, result, contentsPosInResult, c.length);
            contentsPosInResult += c.length;
        }
        return result;
    }

    private static final class JavaToDerConverter {
        private JavaToDerConverter() {}

        public static byte[] toDer(Object source, Asn1Type targetType, Asn1Type targetElementType)
                throws Asn1EncodingException {
            Class<?> sourceType = source.getClass();
            if (Asn1OpaqueObject.class.equals(sourceType)) {
                ByteBuffer buf = ((Asn1OpaqueObject) source).getEncoded();
                byte[] result = new byte[buf.remaining()];
                buf.get(result);
                return result;
            }

            if ((targetType == null) || (targetType == Asn1Type.ANY)) {
                return encode(source);
            }

            switch (targetType) {
                case OCTET_STRING:
                case BIT_STRING:
                    byte[] value = null;
                    if (source instanceof ByteBuffer) {
                        ByteBuffer buf = (ByteBuffer) source;
                        value = new byte[buf.remaining()];
                        buf.slice().get(value);
                    } else if (source instanceof byte[]) {
                        value = (byte[]) source;
                    }
                    if (value != null) {
                        return createTag(
                                BerEncoding.TAG_CLASS_UNIVERSAL,
                                false,
                                BerEncoding.getTagNumber(targetType),
                                value);
                    }
                    break;
                case INTEGER:
                    if (source instanceof Integer) {
                        return toInteger((Integer) source);
                    } else if (source instanceof Long) {
                        return toInteger((Long) source);
                    } else if (source instanceof BigInteger) {
                        return toInteger((BigInteger) source);
                    }
                    break;
                case BOOLEAN:
                    if (source instanceof Boolean) {
                        return toBoolean((Boolean) (source));
                    }
                    break;
                case UTC_TIME:
                case GENERALIZED_TIME:
                    if (source instanceof String) {
                        return createTag(BerEncoding.TAG_CLASS_UNIVERSAL, false,
                                BerEncoding.getTagNumber(targetType), ((String) source).getBytes());
                    }
                    break;
                case OBJECT_IDENTIFIER:
                    if (source instanceof String) {
                        return toOid((String) source);
                    }
                    break;
                case SEQUENCE:
                {
                    Asn1Class containerAnnotation =
                            sourceType.getDeclaredAnnotation(Asn1Class.class);
                    if ((containerAnnotation != null)
                            && (containerAnnotation.type() == Asn1Type.SEQUENCE)) {
                        return toSequence(source);
                    }
                    break;
                }
                case CHOICE:
                {
                    Asn1Class containerAnnotation =
                            sourceType.getDeclaredAnnotation(Asn1Class.class);
                    if ((containerAnnotation != null)
                            && (containerAnnotation.type() == Asn1Type.CHOICE)) {
                        return toChoice(source);
                    }
                    break;
                }
                case SET_OF:
                    return toSetOf((Collection<?>) source, targetElementType);
                case SEQUENCE_OF:
                    return toSequenceOf((Collection<?>) source, targetElementType);
                default:
                    break;
            }

            throw new Asn1EncodingException(
                    "Unsupported conversion: " + sourceType.getName() + " to ASN.1 " + targetType);
        }
    }
    /** ASN.1 DER-encoded {@code NULL}. */
    public static final Asn1OpaqueObject ASN1_DER_NULL =
            new Asn1OpaqueObject(new byte[] {BerEncoding.TAG_NUMBER_NULL, 0});
}
