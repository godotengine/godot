% Attempts to diagnose AEC problems from recorded samples
%
% out = echo_diagnostic(rec_file, play_file, out_file, tail_length)
%
% Computes the full matrix inversion to cancel echo from the 
% recording 'rec_file' using the far end signal 'play_file' using 
% a filter length of 'tail_length'. The output is saved to 'out_file'.
function out = echo_diagnostic(rec_file, play_file, out_file, tail_length)

F=fopen(rec_file,'rb');
rec=fread(F,Inf,'short');
fclose (F);
F=fopen(play_file,'rb');
play=fread(F,Inf,'short');
fclose (F);

rec = [rec; zeros(1024,1)];
play = [play; zeros(1024,1)];

N = length(rec);
corr = real(ifft(fft(rec).*conj(fft(play))));
acorr = real(ifft(fft(play).*conj(fft(play))));

[a,b] = max(corr);

if b > N/2
      b = b-N;
end
printf ("Far end to near end delay is %d samples\n", b);
if (b > .3*tail_length)
      printf ('This is too much delay, try delaying the far-end signal a bit\n');
else if (b < 0)
      printf ('You have a negative delay, the echo canceller has no chance to cancel anything!\n');
   else
      printf ('Delay looks OK.\n');
      end
   end
end
N2 = round(N/2);
corr1 = real(ifft(fft(rec(1:N2)).*conj(fft(play(1:N2)))));
corr2 = real(ifft(fft(rec(N2+1:end)).*conj(fft(play(N2+1:end)))));

[a,b1] = max(corr1);
if b1 > N2/2
      b1 = b1-N2;
end
[a,b2] = max(corr2);
if b2 > N2/2
      b2 = b2-N2;
end
drift = (b1-b2)/N2;
printf ('Drift estimate is %f%% (%d samples)\n', 100*drift, b1-b2);
if abs(b1-b2) < 10
   printf ('A drift of a few (+-10) samples is normal.\n');
else
   if abs(b1-b2) < 30
      printf ('There may be (not sure) excessive clock drift. Is the capture and playback done on the same soundcard?\n');
   else
      printf ('Your clock is drifting! No way the AEC will be able to do anything with that. Most likely, you''re doing capture and playback from two different cards.\n');
      end
   end
end
acorr(1) = .001+1.00001*acorr(1);
AtA = toeplitz(acorr(1:tail_length));
bb = corr(1:tail_length);
h = AtA\bb;

out = (rec - filter(h, 1, play));

F=fopen(out_file,'w');
fwrite(F,out,'short');
fclose (F);
