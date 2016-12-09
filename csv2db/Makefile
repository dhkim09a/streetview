
.PHONY: all clean

CC = gcc

PWD = `pwd`

CFLAGS = -O3 -g -ggdb -Wall

OBJS = main.o

PROG = csv2db.elf

all: $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o
	rm -f $(PROG)
