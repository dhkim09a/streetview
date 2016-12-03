
.PHONY: all clean

CONFIG_MK = config.mk

include $(CONFIG_MK)

CCBIN = g++-4.9
CC = nvcc -ccbin $(CCBIN)

CUFLAGS = -O0 --use_fast_math -lineinfo --maxrregcount=$(REG) -D_FORCE_INLINES
CFLAGS = -O2 -Wall -I../openSURF/src -g \
		 $(shell pkg-config --cflags opencv)
CFLAGS += -Wno-unused-function
LDFLAGS = $(shell pkg-config --libs opencv) \
		  ../openSURF/libsurf.so

CFLAGS += $(addprefix -D,$(CONFIG))

PROG = curator.elf

OBJS = db_loader.o main.o message.o net.o \
			 search.o cpu_search.o gpu_search.o \

all: $(PROG)

$(PROG): $(OBJS)
	$(CC) -g -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

%.o: %.cu 
	$(CC) $(CUFLAGS) $(addprefix -Xcompiler ,$(CFLAGS)) -c $<

%.o: %.c
	$(CCBIN) -mavx $(CFLAGS) -c $<

%.o: %.cpp
	$(CCBIN) -mavx $(CFLAGS) -c $<

clean:
	rm -f *.o
	rm -f *.elf

