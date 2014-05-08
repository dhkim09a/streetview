#!/bin/bash

HASHTHRESH=500

hashthreshinc=$HASHTHRESH
DIR=$1
HASHLIST=""
HASHSHOUT=""

if [ $# -ne 1 ];then
	echo "Usage: $0 [directory]"
	exit
fi

if [[ ! -e $DIR ]];then
	echo "$DIR: no such directory found"
	exit
fi

TOTAL=`ls -1 $DIR | wc -l`

#Generate md5 hash list
cnt=0
echo "Generating MD5 hash... "
for i in `ls $DIR`
do
	HASHSHORT="$HASHSHORT `md5sum $DIR/$i`"
	cnt="`echo "$cnt + 1" | bc`"
	if [ $cnt -gt $hashthreshinc ];then
		HASHLIST="$HASHLIST $HASHSHORT"
		HASHSHORT=""
		hashthreshinc="`echo "$hashthreshinc + $HASHTHRESH" | bc`"
	fi
	echo -ne "\033[1K"
	echo -ne "\r$cnt/$TOTAL"
done
HASHLIST="$HASHLIST $HASHSHORT"
echo " ... done"

echo "Removing duplicants... "
cnt=0
rmvd=0
statei="HASH"
statej="HASH"
for i in $HASHLIST
do
	if [ $statei = "HASH" ];then
		md5i=$i
		#		echo "md5i=$md5i"
		statei="FILE"
	elif [ $statei = "FILE" ];then
		filei=$i
		#		echo "filei=$filei"

		if [ -e $filei ];then
			for j in $HASHLIST
			do
				if [ $statej = "HASH" ];then
					md5j=$j
					#				echo "md5j=$md5j"
					statej="FILE"
				elif [ $statej = "FILE" ];then
					filej=$j
					#				echo "filej=$filej"

					if [ $md5i = $md5j ] && [ $filei != $filej ] && [ -e $filej ];then
						cmp -s $filei $filej
						if [[ $? -eq 0 ]];then
							rm $filej
							rmvd="`echo "$rmvd + 1" | bc`"
						fi
					fi
					statej="HASH"
				fi
			done
		fi

		statei="HASH"
		cnt="`echo "$cnt + 1" | bc`"
		echo -ne "\033[1K"
		echo -ne "\r$cnt/$TOTAL ($rmvd removed)"
	fi
done
echo " ... done"
echo "Removed $rmvd/$TOTAL duplications"
