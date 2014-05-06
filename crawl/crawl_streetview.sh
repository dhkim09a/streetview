#!/bin/bash

#config
LIMIT=2400    #number of pictures
AREA=1000     #meters
INTERVAL=10   #meters
SIZE=640x640  #640x640 is maximum
DIR=dir$1_$2
TEMP_FILE=$DIR/.temp

#constants
LAT_DPM="`echo "scale=6; 1/111133" | bc`" #y
LNG_DPM="`echo "scale=6; 1/93156" | bc`" #x
DAY_SEC="`echo "24 * 3600" | bc`"

ITER="`echo "($AREA / $INTERVAL) + 1" | bc`"
ITER="`echo "$ITER * $ITER * 3" | bc`"

if [ $# -ne 2 ];then
	echo "Usage: $0 [latitude] [longitude]"
	exit
fi

LAT_FROM="`echo "scale=6; $1 - (($AREA / 2) * $LAT_DPM)" | bc`"
LAT_TO="`echo "scale=6; $1 + (($AREA / 2) * $LAT_DPM)" | bc`"
LNG_FROM="`echo "scale=6; $2 - (($AREA / 2) * $LNG_DPM)" | bc`"
LNG_TO="`echo "scale=6; $2 + (($AREA / 2) * $LNG_DPM)" | bc`"
echo "Latitude from $LAT_FROM to $LAT_TO"
echo "Longitude from $LNG_FROM to $LNG_TO"

if [ -e $TEMP_FILE ];then
	COMPLETED="`cat $TEMP_FILE`"
else
	COMPLETED=0
fi

echo "Download `echo "$ITER - $COMPLETED" | bc` streetview snapshots"
echo "It will take `echo "($ITER - $COMPLETED) / $LIMIT" | bc` days"

#if [ $ITER -gt $LIMIT ]; then
#	echo "Error: You can crawl at most $LIMIT pictures"
#	exit
#fi

mkdir -p $DIR
echo "Latitude: from $LAT_FROM to $LAT_TO" >> $DIR/README
echo "Longitude: from $LNG_FROM to $LNG_TO" >> $DIR/README

while true;
do
	echo -n "Do you want to run with this configuration? (y/n) "
	read answer

	if [ $answer = y ];then
		break;
	elif [ $answer = n ];then
		exit
	fi
done

cnt=0
for i in `seq 0 $INTERVAL $AREA`
do
	lat="`echo "scale=6; $1 + (($i - ($AREA / 2)) * $LAT_DPM)" | bc`"
	for j in `seq 0 $INTERVAL $AREA`
	do
		lng="`echo "scale=6; $2 + (($j - ($AREA / 2)) * $LNG_DPM)" | bc`"
		for headling in {0..240..120}
		do
			if [ $cnt -ge $COMPLETED ];then
				`wget "http://maps.googleapis.com/maps/api/streetview?size=$SIZE&location=$lat,$lng&heading=$headling&pitch=0&sensor=false&fov=120" -O "$DIR/sv_$lat$$_$lng$$_$headling.jpg"`
				cnt="`echo "$cnt + 1" | bc`"
				echo "$cnt" > $TEMP_FILE
				progress="`echo "scale=2; ($cnt * 100) / $ITER" | bc`"
				echo "$progress%"
				if [ `echo "$cnt % $LIMIT" | bc` -eq 0 ];then
					echo "`date +"%D %r"`: sleep for a day"
					`sleep $DAY_SEC`
				fi
			else
				cnt="`echo "$cnt + 1" | bc`"
			fi
		done
	done
done
