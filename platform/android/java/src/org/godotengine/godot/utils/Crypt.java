package org.godotengine.godot.utils;

import java.security.MessageDigest;
import java.util.Random;

public class Crypt {

	public static String md5(String input){
        try {
            // Create MD5 Hash
            MessageDigest digest = java.security.MessageDigest.getInstance("MD5");
            digest.update(input.getBytes());
            byte messageDigest[] = digest.digest();
            
            // Create Hex String
            StringBuffer hexString = new StringBuffer();
            for (int i=0; i<messageDigest.length; i++)
                hexString.append(Integer.toHexString(0xFF & messageDigest[i]));
            return hexString.toString();
            
        } catch (Exception e) {
            e.printStackTrace();
        }   
        return "";
	}
	
	public static String createRandomHash(){
		return md5(Long.toString(createRandomLong()));
	}
	
	public static long createAbsRandomLong(){
		return Math.abs(createRandomLong());
	}
	
	public static long createRandomLong(){
		Random r = new Random();
		return r.nextLong();
	}
}
