
package com.android.godot;
import java.util.HashMap;

import android.content.res.AssetManager;
import java.io.InputStream;
import java.io.IOException;

// Wrapper for native library

public class GodotIO {


	AssetManager am;


	/// FILES

	public int last_file_id=1;

	class AssetData {


		public boolean eof=false;
		public String path;
		public InputStream is;
		public int len;
		public int pos;
	}


	HashMap<Integer,AssetData> streams;


	public int file_open(String path,boolean write) {

		System.out.printf("file_open: Attempt to Open %s\n",path);

		if (write)
			return -1;


		AssetData ad = new AssetData();

		try {
			ad.is = am.open(path);

		} catch (Exception e) {

			System.out.printf("Exception on file_open: %s\n",e);
			return -1;
		}

		try {
			ad.len=ad.is.available();
		} catch (Exception e) {

			System.out.printf("Exception availabling on file_open: %s\n",e);
			return -1;
		}

		ad.path=path;
		ad.pos=0;
		++last_file_id;
		streams.put(last_file_id,ad);

		return last_file_id;
	}
	public int file_get_size(int id) {

		if (!streams.containsKey(id)) {
			System.out.printf("file_get_size: Invalid file id: %d\n",id);
			return -1;
		}

		return streams.get(id).len;

	}
	public void file_seek(int id,int bytes) {

		if (!streams.containsKey(id)) {
			System.out.printf("file_get_size: Invalid file id: %d\n",id);
			return;
		}
		//seek sucks
		AssetData ad = streams.get(id);
		if (bytes>ad.len)
			bytes=ad.len;
		if (bytes<0)
			bytes=0;

		try {

		if (bytes > (int)ad.pos) {
			int todo=bytes-(int)ad.pos;
			while(todo>0) {
				todo-=ad.is.skip(todo);
			}
			ad.pos=bytes;
		} else if (bytes<(int)ad.pos) {

			ad.is=am.open(ad.path);

			ad.pos=bytes;
			int todo=bytes;
			while(todo>0) {
				todo-=ad.is.skip(todo);
			}
		}

		ad.eof=false;
		} catch (IOException e) {

			System.out.printf("Exception on file_seek: %s\n",e);
			return;
		}


	}

	public int file_tell(int id) {

		if (!streams.containsKey(id)) {
			System.out.printf("file_read: Can't tell eof for invalid file id: %d\n",id);
			return 0;
		}

		AssetData ad = streams.get(id);
		return ad.pos;
	}
	public boolean file_eof(int id) {

		if (!streams.containsKey(id)) {
			System.out.printf("file_read: Can't check eof for invalid file id: %d\n",id);
			return false;
		}

		AssetData ad = streams.get(id);
		return ad.eof;
	}

	public byte[] file_read(int id, int bytes) {

		if (!streams.containsKey(id)) {
			System.out.printf("file_read: Can't read invalid file id: %d\n",id);
			return new byte[0];
		}


		AssetData ad = streams.get(id);

		if (ad.pos + bytes > ad.len) {

			bytes=ad.len-ad.pos;
			ad.eof=true;
		}


		if (bytes==0) {

			return new byte[0];
		}



		byte[] buf1=new byte[bytes];
		int r=0;
		try {
			r = ad.is.read(buf1);
		} catch (IOException e) {

			System.out.printf("Exception on file_read: %s\n",e);
			return new byte[bytes];
		}

		if (r==0) {
			return new byte[0];
		}

		ad.pos+=r;

		if (r<bytes) {

			byte[] buf2=new byte[r];
			for(int i=0;i<r;i++)
				buf2[i]=buf1[i];
			return buf2;
		} else {

			return buf1;
		}

	}

	public void file_close(int id) {

		if (!streams.containsKey(id)) {
			System.out.printf("file_close: Can't close invalid file id: %d\n",id);
			return;
		}

		streams.remove(id);

	}


	/// DIRECTORIES


	class AssetDir {

		public String[] files;
		public int current;
	}

	public int last_dir_id=1;

	HashMap<Integer,AssetDir> dirs;

	public int dir_open(String path) {

		AssetDir ad = new AssetDir();
		ad.current=0;

		try {
			ad.files = am.list(path);
		} catch (IOException e) {

			System.out.printf("Exception on dir_open: %s\n",e);
			return -1;
		}

		++last_dir_id;
		dirs.put(last_dir_id,ad);

		return last_dir_id;

	}

	public String dir_next(int id) {

		if (!dirs.containsKey(id)) {
			System.out.printf("dir_next: invalid dir id: %d\n",id);
			return "";
		}

		AssetDir ad = dirs.get(id);
		if (ad.current>=ad.files.length)
			return "";
		String r = ad.files[ad.current];
		ad.current++;
		return r;

	}

	public void dir_close(int id) {

		if (!dirs.containsKey(id)) {
			System.out.printf("dir_close: invalid dir id: %d\n",id);
			return;
		}

		dirs.remove(id);
	}


	GodotIO(AssetManager p_am) {

		am=p_am;
		streams=new HashMap<Integer,AssetData>();
		dirs=new HashMap<Integer,AssetDir>();


	}


}
