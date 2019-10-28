#!/bin/bash

#CPUS='0 1 2 4 8 16 32'
#GPUS='0 1 2 3 4 5 6 7 8'
CPUS='24'
GPUS='0 1 2 3 4'
LOG="$(date | tr ' ' '_').log"

for cpu in $CPUS; do for gpu in $GPUS; do
	if [ $cpu -eq 0 ] && [ $gpu -eq 0 ]; then
		continue
	fi

	sed -i -e 's/NUMCPU=[0-9]\+/NUMCPU='"$cpu"'/' config.mk
	sed -i -e 's/NUMGPU=[0-9]\+/NUMGPU='"$gpu"'/' config.mk

	make clean > /dev/null 2> /dev/null
	make -j > /dev/null 2> /dev/null
	
	for i in {1..1}; do
		./curator.elf ../valparaiso.db Street-Art-Valparaiso-1000x658.jpg
	done \
		| grep --line-buffered search \
		| unbuffer -p tail -n +2 \
		| awk '{ total += $3; count++; print count ": " $3 " ms"; fflush() } END { print '"$cpu"'" cpus, "'"$gpu"'" gpus average: " total/count " ms" }'

	echo "-----"

done done | unbuffer -p tee $LOG;
