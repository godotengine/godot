#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t; cin>>t;
	while(t--){
	   string s;
	   cin>>s;
	   bool h=0;
	   int c=0;
	   for(int i=0; i<s.size();i++){
	       if(s[i]=='a' || s[i]=='e'|| s[i]=='i'|| s[i]=='o' || s[i]=='u'){
	           c++;
	           if(c>=3){
                cout<<"Happy"<<endl;
                h=1;
                break;
	           }
	           }
                    else{
                        c=0;

                    }
       }
         if(!h){
          cout<<"Sad"<<endl;
    }
	}
    return 0;
}
