
.PHONY: all clean

CC = gcc

CFLAGS = -O2 -Wall -g $(shell pkg-config --cflags opencv)
LDFLAGS = $(shell pkg-config --libs opencv)

CFLAGS += $(addprefix -D,$(CONFIG))

PROG = pick_geolocation.elf

OBJ = pick_geolocation.o

all: $(PROG)

$(PROG): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CCBIN) $(CFLAGS) -c $<

clean:
	rm -f *.o
	rm -f *.elf

