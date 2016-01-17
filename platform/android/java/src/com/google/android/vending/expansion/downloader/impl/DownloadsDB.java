/*
 * Copyright (C) 2012 The Android Open Source Project
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

package com.google.android.vending.expansion.downloader.impl;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteDoneException;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteStatement;
import android.provider.BaseColumns;
import android.util.Log;

public class DownloadsDB {
    private static final String DATABASE_NAME = "DownloadsDB";
    private static final int DATABASE_VERSION = 7;
    public static final String LOG_TAG = DownloadsDB.class.getName();
    final SQLiteOpenHelper mHelper;
    SQLiteStatement mGetDownloadByIndex;
    SQLiteStatement mUpdateCurrentBytes;
    private static DownloadsDB mDownloadsDB;
    long mMetadataRowID = -1;
    int mVersionCode = -1;
    int mStatus = -1;
    int mFlags;

    static public synchronized DownloadsDB getDB(Context paramContext) {
        if (null == mDownloadsDB) {
            return new DownloadsDB(paramContext);
        }
        return mDownloadsDB;
    }

    private SQLiteStatement getDownloadByIndexStatement() {
        if (null == mGetDownloadByIndex) {
            mGetDownloadByIndex = mHelper.getReadableDatabase().compileStatement(
                    "SELECT " + BaseColumns._ID + " FROM "
                            + DownloadColumns.TABLE_NAME + " WHERE "
                            + DownloadColumns.INDEX + " = ?");
        }
        return mGetDownloadByIndex;
    }

    private SQLiteStatement getUpdateCurrentBytesStatement() {
        if (null == mUpdateCurrentBytes) {
            mUpdateCurrentBytes = mHelper.getReadableDatabase().compileStatement(
                    "UPDATE " + DownloadColumns.TABLE_NAME + " SET " + DownloadColumns.CURRENTBYTES
                            + " = ?" +
                            " WHERE " + DownloadColumns.INDEX + " = ?");
        }
        return mUpdateCurrentBytes;
    }

    private DownloadsDB(Context paramContext) {
        this.mHelper = new DownloadsContentDBHelper(paramContext);
        final SQLiteDatabase sqldb = mHelper.getReadableDatabase();
        // Query for the version code, the row ID of the metadata (for future
        // updating) the status and the flags
        Cursor cur = sqldb.rawQuery("SELECT " +
                MetadataColumns.APKVERSION + "," +
                BaseColumns._ID + "," +
                MetadataColumns.DOWNLOAD_STATUS + "," +
                MetadataColumns.FLAGS +
                " FROM "
                + MetadataColumns.TABLE_NAME + " LIMIT 1", null);
        if (null != cur && cur.moveToFirst()) {
            mVersionCode = cur.getInt(0);
            mMetadataRowID = cur.getLong(1);
            mStatus = cur.getInt(2);
            mFlags = cur.getInt(3);
            cur.close();
        }
        mDownloadsDB = this;
    }

    protected DownloadInfo getDownloadInfoByFileName(String fileName) {
        final SQLiteDatabase sqldb = mHelper.getReadableDatabase();
        Cursor itemcur = null;
        try {
            itemcur = sqldb.query(DownloadColumns.TABLE_NAME, DC_PROJECTION,
                    DownloadColumns.FILENAME + " = ?",
                    new String[] {
                        fileName
                    }, null, null, null);
            if (null != itemcur && itemcur.moveToFirst()) {
                return getDownloadInfoFromCursor(itemcur);
            }
        } finally {
            if (null != itemcur)
                itemcur.close();
        }
        return null;
    }

    public long getIDForDownloadInfo(final DownloadInfo di) {
        return getIDByIndex(di.mIndex);
    }

    public long getIDByIndex(int index) {
        SQLiteStatement downloadByIndex = getDownloadByIndexStatement();
        downloadByIndex.clearBindings();
        downloadByIndex.bindLong(1, index);
        try {
            return downloadByIndex.simpleQueryForLong();
        } catch (SQLiteDoneException e) {
            return -1;
        }
    }

    public void updateDownloadCurrentBytes(final DownloadInfo di) {
        SQLiteStatement downloadCurrentBytes = getUpdateCurrentBytesStatement();
        downloadCurrentBytes.clearBindings();
        downloadCurrentBytes.bindLong(1, di.mCurrentBytes);
        downloadCurrentBytes.bindLong(2, di.mIndex);
        downloadCurrentBytes.execute();
    }

    public void close() {
        this.mHelper.close();
    }

    protected static class DownloadsContentDBHelper extends SQLiteOpenHelper {
        DownloadsContentDBHelper(Context paramContext) {
            super(paramContext, DATABASE_NAME, null, DATABASE_VERSION);
        }

        private String createTableQueryFromArray(String paramString,
                String[][] paramArrayOfString) {
            StringBuilder localStringBuilder = new StringBuilder();
            localStringBuilder.append("CREATE TABLE ");
            localStringBuilder.append(paramString);
            localStringBuilder.append(" (");
            int i = paramArrayOfString.length;
            for (int j = 0;; j++) {
                if (j >= i) {
                    localStringBuilder
                            .setLength(localStringBuilder.length() - 1);
                    localStringBuilder.append(");");
                    return localStringBuilder.toString();
                }
                String[] arrayOfString = paramArrayOfString[j];
                localStringBuilder.append(' ');
                localStringBuilder.append(arrayOfString[0]);
                localStringBuilder.append(' ');
                localStringBuilder.append(arrayOfString[1]);
                localStringBuilder.append(',');
            }
        }

        /**
         * These two arrays must match and have the same order. For every Schema
         * there must be a corresponding table name.
         */
        static final private String[][][] sSchemas = {
                DownloadColumns.SCHEMA, MetadataColumns.SCHEMA
        };

        static final private String[] sTables = {
                DownloadColumns.TABLE_NAME, MetadataColumns.TABLE_NAME
        };

        /**
         * Goes through all of the tables in sTables and drops each table if it
         * exists. Altered to no longer make use of reflection.
         */
        private void dropTables(SQLiteDatabase paramSQLiteDatabase) {
            for (String table : sTables) {
                try {
                    paramSQLiteDatabase.execSQL("DROP TABLE IF EXISTS " + table);
                } catch (Exception localException) {
                    localException.printStackTrace();
                }
            }
        }

        /**
         * Goes through all of the tables in sTables and creates a database with
         * the corresponding schema described in sSchemas. Altered to no longer
         * make use of reflection.
         */
        public void onCreate(SQLiteDatabase paramSQLiteDatabase) {
            int numSchemas = sSchemas.length;
            for (int i = 0; i < numSchemas; i++) {
                try {
                    String[][] schema = (String[][]) sSchemas[i];
                    paramSQLiteDatabase.execSQL(createTableQueryFromArray(
                            sTables[i], schema));
                } catch (Exception localException) {
                    while (true)
                        localException.printStackTrace();
                }
            }
        }

        public void onUpgrade(SQLiteDatabase paramSQLiteDatabase,
                int paramInt1, int paramInt2) {
            Log.w(DownloadsContentDBHelper.class.getName(),
                    "Upgrading database from version " + paramInt1 + " to "
                            + paramInt2 + ", which will destroy all old data");
            dropTables(paramSQLiteDatabase);
            onCreate(paramSQLiteDatabase);
        }
    }

    public static class MetadataColumns implements BaseColumns {
        public static final String APKVERSION = "APKVERSION";
        public static final String DOWNLOAD_STATUS = "DOWNLOADSTATUS";
        public static final String FLAGS = "DOWNLOADFLAGS";

        public static final String[][] SCHEMA = {
                {
                        BaseColumns._ID, "INTEGER PRIMARY KEY"
                },
                {
                        APKVERSION, "INTEGER"
                }, {
                        DOWNLOAD_STATUS, "INTEGER"
                },
                {
                        FLAGS, "INTEGER"
                }
        };
        public static final String TABLE_NAME = "MetadataColumns";
        public static final String _ID = "MetadataColumns._id";
    }

    public static class DownloadColumns implements BaseColumns {
        public static final String INDEX = "FILEIDX";
        public static final String URI = "URI";
        public static final String FILENAME = "FN";
        public static final String ETAG = "ETAG";

        public static final String TOTALBYTES = "TOTALBYTES";
        public static final String CURRENTBYTES = "CURRENTBYTES";
        public static final String LASTMOD = "LASTMOD";

        public static final String STATUS = "STATUS";
        public static final String CONTROL = "CONTROL";
        public static final String NUM_FAILED = "FAILCOUNT";
        public static final String RETRY_AFTER = "RETRYAFTER";
        public static final String REDIRECT_COUNT = "REDIRECTCOUNT";

        public static final String[][] SCHEMA = {
                {
                        BaseColumns._ID, "INTEGER PRIMARY KEY"
                },
                {
                        INDEX, "INTEGER UNIQUE"
                }, {
                        URI, "TEXT"
                },
                {
                        FILENAME, "TEXT UNIQUE"
                }, {
                        ETAG, "TEXT"
                },
                {
                        TOTALBYTES, "INTEGER"
                }, {
                        CURRENTBYTES, "INTEGER"
                },
                {
                        LASTMOD, "INTEGER"
                }, {
                        STATUS, "INTEGER"
                },
                {
                        CONTROL, "INTEGER"
                }, {
                        NUM_FAILED, "INTEGER"
                },
                {
                        RETRY_AFTER, "INTEGER"
                }, {
                        REDIRECT_COUNT, "INTEGER"
                }
        };
        public static final String TABLE_NAME = "DownloadColumns";
        public static final String _ID = "DownloadColumns._id";
    }

    private static final String[] DC_PROJECTION = {
            DownloadColumns.FILENAME,
            DownloadColumns.URI, DownloadColumns.ETAG,
            DownloadColumns.TOTALBYTES, DownloadColumns.CURRENTBYTES,
            DownloadColumns.LASTMOD, DownloadColumns.STATUS,
            DownloadColumns.CONTROL, DownloadColumns.NUM_FAILED,
            DownloadColumns.RETRY_AFTER, DownloadColumns.REDIRECT_COUNT,
            DownloadColumns.INDEX
    };

    private static final int FILENAME_IDX = 0;
    private static final int URI_IDX = 1;
    private static final int ETAG_IDX = 2;
    private static final int TOTALBYTES_IDX = 3;
    private static final int CURRENTBYTES_IDX = 4;
    private static final int LASTMOD_IDX = 5;
    private static final int STATUS_IDX = 6;
    private static final int CONTROL_IDX = 7;
    private static final int NUM_FAILED_IDX = 8;
    private static final int RETRY_AFTER_IDX = 9;
    private static final int REDIRECT_COUNT_IDX = 10;
    private static final int INDEX_IDX = 11;

    /**
     * This function will add a new file to the database if it does not exist.
     * 
     * @param di DownloadInfo that we wish to store
     * @return the row id of the record to be updated/inserted, or -1
     */
    public boolean updateDownload(DownloadInfo di) {
        ContentValues cv = new ContentValues();
        cv.put(DownloadColumns.INDEX, di.mIndex);
        cv.put(DownloadColumns.FILENAME, di.mFileName);
        cv.put(DownloadColumns.URI, di.mUri);
        cv.put(DownloadColumns.ETAG, di.mETag);
        cv.put(DownloadColumns.TOTALBYTES, di.mTotalBytes);
        cv.put(DownloadColumns.CURRENTBYTES, di.mCurrentBytes);
        cv.put(DownloadColumns.LASTMOD, di.mLastMod);
        cv.put(DownloadColumns.STATUS, di.mStatus);
        cv.put(DownloadColumns.CONTROL, di.mControl);
        cv.put(DownloadColumns.NUM_FAILED, di.mNumFailed);
        cv.put(DownloadColumns.RETRY_AFTER, di.mRetryAfter);
        cv.put(DownloadColumns.REDIRECT_COUNT, di.mRedirectCount);
        return updateDownload(di, cv);
    }

    public boolean updateDownload(DownloadInfo di, ContentValues cv) {
        long id = di == null ? -1 : getIDForDownloadInfo(di);
        try {
            final SQLiteDatabase sqldb = mHelper.getWritableDatabase();
            if (id != -1) {
                if (1 != sqldb.update(DownloadColumns.TABLE_NAME,
                        cv, DownloadColumns._ID + " = " + id, null)) {
                    return false;
                }
            } else {
                return -1 != sqldb.insert(DownloadColumns.TABLE_NAME,
                        DownloadColumns.URI, cv);
            }
        } catch (android.database.sqlite.SQLiteException ex) {
            ex.printStackTrace();
        }
        return false;
    }

    public int getLastCheckedVersionCode() {
        return mVersionCode;
    }

    public boolean isDownloadRequired() {
        final SQLiteDatabase sqldb = mHelper.getReadableDatabase();
        Cursor cur = sqldb.rawQuery("SELECT Count(*) FROM "
                + DownloadColumns.TABLE_NAME + " WHERE "
                + DownloadColumns.STATUS + " <> 0", null);
        try {
            if (null != cur && cur.moveToFirst()) {
                return 0 == cur.getInt(0);
            }
        } finally {
            if (null != cur)
                cur.close();
        }
        return true;
    }

    public int getFlags() {
        return mFlags;
    }

    public boolean updateFlags(int flags) {
        if (mFlags != flags) {
            ContentValues cv = new ContentValues();
            cv.put(MetadataColumns.FLAGS, flags);
            if (updateMetadata(cv)) {
                mFlags = flags;
                return true;
            } else {
                return false;
            }
        } else {
            return true;
        }
    };

    public boolean updateStatus(int status) {
        if (mStatus != status) {
            ContentValues cv = new ContentValues();
            cv.put(MetadataColumns.DOWNLOAD_STATUS, status);
            if (updateMetadata(cv)) {
                mStatus = status;
                return true;
            } else {
                return false;
            }
        } else {
            return true;
        }
    };

    public boolean updateMetadata(ContentValues cv) {
        final SQLiteDatabase sqldb = mHelper.getWritableDatabase();
        if (-1 == this.mMetadataRowID) {
            long newID = sqldb.insert(MetadataColumns.TABLE_NAME,
                    MetadataColumns.APKVERSION, cv);
            if (-1 == newID)
                return false;
            mMetadataRowID = newID;
        } else {
            if (0 == sqldb.update(MetadataColumns.TABLE_NAME, cv,
                    BaseColumns._ID + " = " + mMetadataRowID, null))
                return false;
        }
        return true;
    }

    public boolean updateMetadata(int apkVersion, int downloadStatus) {
        ContentValues cv = new ContentValues();
        cv.put(MetadataColumns.APKVERSION, apkVersion);
        cv.put(MetadataColumns.DOWNLOAD_STATUS, downloadStatus);
        if (updateMetadata(cv)) {
            mVersionCode = apkVersion;
            mStatus = downloadStatus;
            return true;
        } else {
            return false;
        }
    };

    public boolean updateFromDb(DownloadInfo di) {
        final SQLiteDatabase sqldb = mHelper.getReadableDatabase();
        Cursor cur = null;
        try {
            cur = sqldb.query(DownloadColumns.TABLE_NAME, DC_PROJECTION,
                    DownloadColumns.FILENAME + "= ?",
                    new String[] {
                        di.mFileName
                    }, null, null, null);
            if (null != cur && cur.moveToFirst()) {
                setDownloadInfoFromCursor(di, cur);
                return true;
            }
            return false;
        } finally {
            if (null != cur) {
                cur.close();
            }
        }
    }

    public void setDownloadInfoFromCursor(DownloadInfo di, Cursor cur) {
        di.mUri = cur.getString(URI_IDX);
        di.mETag = cur.getString(ETAG_IDX);
        di.mTotalBytes = cur.getLong(TOTALBYTES_IDX);
        di.mCurrentBytes = cur.getLong(CURRENTBYTES_IDX);
        di.mLastMod = cur.getLong(LASTMOD_IDX);
        di.mStatus = cur.getInt(STATUS_IDX);
        di.mControl = cur.getInt(CONTROL_IDX);
        di.mNumFailed = cur.getInt(NUM_FAILED_IDX);
        di.mRetryAfter = cur.getInt(RETRY_AFTER_IDX);
        di.mRedirectCount = cur.getInt(REDIRECT_COUNT_IDX);
    }

    public DownloadInfo getDownloadInfoFromCursor(Cursor cur) {
        DownloadInfo di = new DownloadInfo(cur.getInt(INDEX_IDX),
                cur.getString(FILENAME_IDX), this.getClass().getPackage()
                        .getName());
        setDownloadInfoFromCursor(di, cur);
        return di;
    }

    public DownloadInfo[] getDownloads() {
        final SQLiteDatabase sqldb = mHelper.getReadableDatabase();
        Cursor cur = null;
        try {
            cur = sqldb.query(DownloadColumns.TABLE_NAME, DC_PROJECTION, null,
                    null, null, null, null);
            if (null != cur && cur.moveToFirst()) {
                DownloadInfo[] retInfos = new DownloadInfo[cur.getCount()];
                int idx = 0;
                do {
                    DownloadInfo di = getDownloadInfoFromCursor(cur);
                    retInfos[idx++] = di;
                } while (cur.moveToNext());
                return retInfos;
            }
            return null;
        } finally {
            if (null != cur) {
                cur.close();
            }
        }
    }

}
