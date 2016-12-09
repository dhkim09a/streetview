#!/bin/bash

export PATH=$PATH:/home/dhkim/cs572/streetview/csv2db

csv2db=csv2db.elf
root_dir="$PWD"

show_help() {
	echo >&2 "Usage: $0 input_csv number_of_test_samples category_keywords"
}

if [ ! $# -eq 3 ]; then
	show_help
	exit
fi

input_csv=$1
test_no=$2
keywords=$3

for i in $input_csv; do
	if [ ! -f $i ]; then
		echo >&2 "$i does not exist!"
		exit
	fi
	tmp="$tmp `readlink -f $i`"
done
input_csv=$tmp

for keyword in $keywords; do
	echo "[[$keyword]]"

	out_dir="$root_dir/$keyword"
	testvecnames=""

	if [ -d $out_dir ]; then
		echo >&2 "$out_dir alread exists!"
		continue
	fi

	mkdir -p $out_dir

	for i in $input_csv; do 
		cnt="0"
		id="0"
		regex_vecs=""
		temp=""
		while true; do
			if [ $cnt -eq $test_no ]; then
				break;
			fi

			testvecname="${keyword}__${id}"
			testvec=`cat $i | grep "\\<$testvecname\\>"`

			if [ ! -z "$testvec" ]; then
				if [ -z "$regex_vecs" ]; then
					regex_vecs="\\<\\($testvecname"
				else
					regex_vecs="$regex_vecs\\|$testvecname"
				fi
				echo "$testvec" > $out_dir/${testvecname}_`basename $i`
				((cnt++))

				temp="$temp $testvecname"
			fi
			((id++))
		done

		if [ -z "$testvecnames" ]; then
			testvecnames="$temp complementary"
		fi

		if [ ! -z "$regex_vecs" ]; then
			regex_vecs="$regex_vecs\\)\\>"
		fi

		cat $i | grep "^$keyword" | grep -v "$regex_vecs" > $out_dir/complementary_`basename $i`
	done

	# Conver them to DB

	cd $out_dir

	for v in $testvecnames; do
		targets=`ls $v*.csv`

		$csv2db `echo "$targets" | sed "s/\\(\\<\\(\\w\\|\\.\\)\\+\\>\\)/ -f \\1/g"`
	done
done
