#!/bin/bash

if [ $# -ne 1 ];then
	echo "Usage: $0 [street-view image]"
	exit
fi

#config
PICKER=pick_geolocation.elf
LIMIT=25000    #number of pictures
SIZE=640x640  #640x640 is maximum
DIR=dir_$1
TEMP_FILE=$DIR/.temp
GEOLO_FILE=$DIR/geolo.txt
KEY_FILE=key.txt
QUERY_INTERVAL=0 #sec
RETRY_INTERVAL=10 #sec

#constants
DAY_SEC="`echo "24 * 3600" | bc`"
OK_MSG="200 OK"

if [ -e $TEMP_FILE ];then
	COMPLETED="`cat $TEMP_FILE`"
else
	COMPLETED=0
fi

if [ -e $KEY_FILE ];then
	echo "You have Google API Key"
	KEY="`cat $KEY_FILE`"
	KEY="&key=$KEY"
else
	echo "Warning: $KEY_FILE not found"
	KEY=""
fi

#if [ $ITER -gt $LIMIT ]; then
#	echo "Error: You can crawl at most $LIMIT pictures"
#	exit
#fi

# prepare directory
mkdir -p $DIR
cp -f $1 $DIR

# make and run PICKER
make $PICKER
./$PICKER $1 $GEOLO_FILE

ITER="`cat $GEOLO_FILE | wc -l`"
ITER="`echo "$ITER * 3" | bc`"

echo "Download `echo "$ITER - $COMPLETED" | bc` streetview snapshots"
echo "It will take `echo "($ITER - $COMPLETED) / $LIMIT" | bc` days"

while true;
do
	echo -n "Do you want to run with this configuration? (y/n) "
	read answer

	if [ $answer = y ];then
		break;
	elif [ $answer = n ];then
		rm -rf $DIR
		exit
	fi
done

# start crawl
cnt=0
progress=0
state="LAT"
LATLNG="`cat $GEOLO_FILE`"
for i in $LATLNG
do
	if [ $state = "LAT" ];then
		lat=$i
		state="LNG"
	elif [ $state = "LNG" ];then
		lng=$i
		for headling in {0..240..120}
		do
			if [ $cnt -ge $COMPLETED ];then
				result=""
				date=""
				while true;
				do
					date="`date +"%D %T"`"
					result=$OK_MSG
					result="`wget "http://maps.googleapis.com/maps/api/streetview?size=$SIZE&location=$lat,$lng&heading=$headling&pitch=0&sensor=false&fov=120$KEY" -O "$DIR/sv_$lat$$_$lng$$_$headling.jpg" 2>&1`"

					if [ `echo $result | grep -c "$OK_MSG"` -gt 0 ];then
						sleep $QUERY_INTERVAL
						break
					else
						echo -ne "\033[1K"
						echo -ne "Retrying since $date for every $RETRY_INTERVAL secs. ($progress%,$cnt/$ITER)"
						sleep $RETRY_INTERVAL
					fi
				done

				cnt="`echo "$cnt + 1" | bc`"
				echo -ne "$cnt" > $TEMP_FILE
				progress="`echo "scale=2; ($cnt * 100) / $ITER" | bc`"
				echo -ne "\033[2K"
				echo -ne "\r$progress% ($cnt / $ITER)"
			else
				cnt="`echo "$cnt + 1" | bc`"
			fi
		done
		state="LAT"
	fi
done
echo ""
