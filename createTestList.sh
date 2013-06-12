#!/bin/zsh

basename=$1
number=$2
echo "${basename}_${number}"
babasename=${basename}_${number}

if [ $# -ge 1 ]; then
    echo "60" > testDataList.txt
    for i in `seq 1 60`
    do
	numberedname=${babasename}_1_${i}_
	echo ${numberedname}crop.png ${numberedname}depthcrop.png nodata ${basename} 0 0 0 EOL >> testDataList.txt
	
    done
    mv ./testDataList.txt ./dataset/${basename}/${babasename}
else
    echo "usage : ./createNegList.sh [basename *example bottle_1]"
    exit 1
fi
