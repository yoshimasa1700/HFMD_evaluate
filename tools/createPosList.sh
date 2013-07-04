#!/bin/zsh

basename=$1
echo "${basename}"
classFolderList=(`echo ${basename}/*`)

for cFolder in ${classFolderList}
do
    if [ -d $cFolder ]; then
	#classes+=${cFolder##*/}
	echo ${cFolder##*/}

	instanceFolderList=(`echo ${cFolder}/*`)

	for iFolder in ${instanceFolderList}
	do
	    echo $iFolder
	    
	    fileList=(`echo ${iFolder}/*`)

	    rm ${iFolder}/dataList2.0.txt
	    touch ${iFolder}/dataList2.0.txt
	    
	    num=0

	    for f in ${fileList}
	    do
		if [ ! -d $f ]; then
		    
		    case ${f##*/} in
			*_crop*)
			    bName=${f##*/}
			    bbName=${bName%_*}
			    echo ${bbName}
			    angle=`expr ${bbName##*_} \* 6`
			    echo ${angle}
			    echo ${bName%_*}_crop.png ${bName%_*}_depthcrop.png ${bName%_*}_maskcrop.png $angle >> ${iFolder}/dataList2.0.txt
		    esac

		    num=`expr $num + 1`
		fi
	    done

	    cat ${iFolder}/dataList2.0.txt | sed "1i\\${num}"  > ${iFolder}/dataList2.0.txt
	    #sed -i '1s/^/${num}}\n/' ${iFolder}/dataList2.0.txt

	done
    fi
done

# if [ $# -ge 1 ]; then
#     echo "60" > dataList.txt
#     for i in `seq 1 60`
#     do
# 	numberedname=${babasename}_1_${i}_
# 	echo ${numberedname}crop.png ${numberedname}depthcrop.png >> negDataList.txt
# 	mv ./negDataList.txt ./dataset/${basename}/${babasename}
#     done
# else
#     echo "usage : ./createNegList.sh [basename *example bottle_1]"
#     exit 1
# fi
