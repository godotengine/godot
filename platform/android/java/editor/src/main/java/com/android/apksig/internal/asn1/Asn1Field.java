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

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface Asn1Field {
    /** Index used to order fields in a container. Required for fields of SEQUENCE containers. */
    public int index() default 0;

    public Asn1TagClass cls() default Asn1TagClass.AUTOMATIC;

    public Asn1Type type();

    /** Tagging mode. Default: NORMAL. */
    public Asn1Tagging tagging() default Asn1Tagging.NORMAL;

    /** Tag number. Required when IMPLICIT and EXPLICIT tagging mode is used.*/
    public int tagNumber() default -1;

    /** {@code true} if this field is optional. Ignored for fields of CHOICE containers. */
    public boolean optional() default false;

    /** Type of elements. Used only for SET_OF or SEQUENCE_OF. */
    public Asn1Type elementType() default Asn1Type.ANY;
}
