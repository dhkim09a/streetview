
.PHONY: all clean

CC = g++

PWD = `pwd`

CFLAGS = -O2 -Wall -I../openSURF/src `pkg-config --cflags opencv`
LDFLAGS = ../openSURF/libsurf.so -lpthread `pkg-config --libs opencv` 

OBJS = main.o

PROG = dbmaker.elf

all: $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o
	rm -f $(PROG)
