#!/bin/bash

export PATH=$PATH:/home/dhkim/cs572

gen_test=gen_test.sh
run_test=run_test.sh
root_dir="$PWD"
result_dir=$root_dir/result

show_help() {
	echo >&2 "Usage: $0 image_dir input_csv number_of_test_samples category_keywords"
}

save_result() {
	#echo "1: $1, 2: $2, 3: $3"
	echo "$2" > $1
}

if [ ! $# -eq 4 ]; then
	show_help
	exit
fi

img_dir=$root_dir/$1
input_csv=$2
test_no=$3
keywords=$4

for i in $input_csv; do
	if [ ! -f $i ]; then
		echo >&2 "$i does not exist!"
		exit
	fi
	tmp="$tmp `readlink -f $i`"
done
input_csv=$tmp

for i in $input_csv; do
	tmp=`basename $i`
	model=${tmp%.*}
	models="$models $model"

	mkdir -p $root_dir/$model

	cd $root_dir/$model

	$gen_test $i $test_no "$keywords"

	ifs_save=$IFS
	IFS=$'\n'
	echo "$run_test"
	for line in `find ! -path . -type d -exec $run_test {} \\;`; do
		IFS=$ifs_save
		save_result $line
		IFS=$'\n'
	done
	IFS=$ifs_save
done

for m in $models; do
	for t in `find $root_dir/$m -maxdepth 1 -type f`; do
		name=`basename $t`
		mkdir -p $result_dir/$name
		if [ ! -f $result_dir/$name/target.* ]; then
			img=`ls $img_dir/$name.*`
			cp $img_dir/$name.* $result_dir/$name/target.${img##*.}
		fi
		img=`cat $t`
		cp $img_dir/$img $result_dir/$name/$m.${img##*.}
	done
done
