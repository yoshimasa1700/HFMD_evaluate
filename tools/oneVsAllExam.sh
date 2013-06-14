#!/bin/zsh

classfilename=$1
folderfilename=$2

cat ${classfilename}　| while read line
do
	classname+=${line}
done

for i in `seq 1 ${#classname[@]}`
do
	echo ${classname[$i-1]}
done

