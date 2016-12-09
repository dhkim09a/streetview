#!/bin/bash

export PATH=$PATH:/home/dhkim/cs572/streetview/app

curator=curator.elf
complementary="complementary"

show_help() {
	echo >&2 "Usage: $0 directory"
}

if [ ! $# -eq "1" ]; then
	show_help
	exit
fi

dir=$1

if [ ! -d $dir ]; then
	echo >&2 "$dir does not exist!"
	exit
fi

cd $dir

targets=`ls *.db | grep -v "$complementary"`
complementary=`ls *.db | grep "$complementary"`

for t in $targets; do
	name=`echo $t | sed "s/^[0-9]\\+_[0-9]\\+_\\(\\w\\+__[0-9]\\+\\).*/\\1/g"`
	echo -n "$name "
	$curator $complementary $t
done
