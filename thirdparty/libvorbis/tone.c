#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

void usage(){
  fprintf(stderr,"tone <frequency_Hz>,[<amplitude>] [<frequency_Hz>,[<amplitude>]...]\n");
  exit(1);
}

int main (int argc,char *argv[]){
  int i,j;
  double *f;
  double *amp;

  if(argc<2)usage();

  f=alloca(sizeof(*f)*(argc-1));
  amp=alloca(sizeof(*amp)*(argc-1));

  i=0;
  while(argv[i+1]){
    char *pos=strchr(argv[i+1],',');

    f[i]=atof(argv[i+1]);
    if(pos)
      amp[i]=atof(pos+1)*32767.f;
    else
      amp[i]=32767.f;

    fprintf(stderr,"%g Hz, %g amp\n",f[i],amp[i]);

    i++;
  }

  for(i=0;i<44100*10;i++){
    float val=0;
    int ival;
    for(j=0;j<argc-1;j++)
      val+=amp[j]*sin(i/44100.f*f[j]*2*M_PI);
    ival=rint(val);

    if(ival>32767.f)ival=32767.f;
    if(ival<-32768.f)ival=-32768.f;

    fprintf(stdout,"%c%c%c%c",
            (char)(ival&0xff),
            (char)((ival>>8)&0xff),
            (char)(ival&0xff),
            (char)((ival>>8)&0xff));
  }
  return(0);
}

