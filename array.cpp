#include <iostream>
#include <conio.h>
using namespace std;
main ()
{
char nama_brg[20];
int jmlh_beli=0,tot_beli=0,x,y,kembali,saldo,nama;
awal:
long int harga=0,hrg_brg=0,total=0,bayar=0,diskon;
cout<<"masukkan jumlah barang yang di beli"<<endl;
cin>>y;
x=1;	
      while (x<=y){
cout<<"\nmasukkan nama barang"<<x<<":";
cin>>nama_brg;
cout<<"masukkan harga barang :";
	cin>>harga;
	cout<<"masukkan jumlah pembeli : ";
cin>>jmlh_beli;

hrg_brg=harga*jmlh_beli;
       total=total+hrg_brg;
           tot_beli+=jmlh_beli;
	   x++;
}
if (total>=50000){
	diskon =0.3*total;
}
else{
diskon=0;
	}
	cout<<"uang yang di bayarkan :\n";
	cin>>saldo;
	bayar = total-diskon;
kembali= saldo-bayar;
	cout<<"total beli\n"<<tot_beli<<endl;
cout<<"total pembelian\n"<<total<<endl;
	cout<<" mendapat potongan diskon sebesar :\n"<<diskon<<endl;
cout<<"total yang harus di bayar\n="<<bayar<<endl;
	cout<<"kemabalian anda sebesar\n"<<kembali<<endl;
getch();	
goto awal;
}
