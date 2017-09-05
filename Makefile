CC=gcc
LD=gcc
OBJS=qscale.o libqscale.o

CFLAGS=-Wall -O2 -msse3 -fPIC
LDFLAGS=-lm -ljpeg

qscale: qscale.o libqscale.o

# Perl stuff
qscale_wrap.c: qscale.i
	swig -perl5 qscale.i
qscale_wrap.o: qscale_wrap.c
	$(CC) $(CFLAGS) $(shell perl -MExtUtils::Embed -e ccopts) -c $< -o $@
qscale.so: libqscale.o qscale_wrap.o
	$(LD) $(LDFLAGS) -shared libqscale.o qscale_wrap.o -o $@

clean:
	$(RM) qscale $(OBJS) qscale_wrap.o qscale_wrap.c qscale.pm qscale.so

.PHONY: clean
