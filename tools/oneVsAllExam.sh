#!/bin/zsh

classfilename=$1
folderfilename=$2

cat ${classfilename}| while read line
do
	classname+=(${line})
done

cat ${folderfilename}| while read line
do
    foldername+=(`echo "${line}"`)
done

for i in `seq 1 ${#classname[@]}`
do
    #実験フォルダの準備
	./tools/changeConfig ./config.xml stride 1
    ./tools/setupExperiment.sh ${classname[$i]}VsAll
    echo ${classname[$i]}
    num=`expr ${#foldername[@]} / 2`

    posfolder=0
    negfolder=0

    for j in `seq 1 $num`
    do

	k=`expr $j\*2`
	if [ "${classname[$i]}" = "${foldername[$k-1]}" ]; then
	    echo ${foldername[$k-1]}_${foldername[$k]}
	    posfolder=`expr $posfolder + 1`
	fi
	if [ "${classname[$i]}" != "${foldername[$k-1]}" ]; then
	    echo ${foldername[$k-1]}_${foldername[$k]}
	    negfolder=`expr $negfolder + 1`
	fi
    done
    
    echo "$posfolder"　> trainData.txt
    echo "$negfolder" > negDataFolderList.txt
    for j in `seq 1 $num`
    do
	
	k=`expr $j\*2`
	if [ "${classname[$i]}" = "${foldername[$k-1]}" ]; then
	    echo ${foldername[$k-1]}_${foldername[$k]} >> trainData.txt
	    
	fi
	if [ "${classname[$i]}" != "${foldername[$k-1]}" ]; then
	    echo ${foldername[$k-1]}_${foldername[$k]} >> negDataFolderList.txt
    
	fi
    done
    mv ../HFMD_evaluate/tainData.txt ../${classname[$i]}VsAll/dataset
    mv ../HFMD_evaluate/negDataFolderList.txt ../${classname[$i]}VsAll/negdata

    ../${classname[$i]}VsAll/learning
	../${classname[$i]}VsAll/tools/changeConfig.sh ./config.xml stride 5
    ../${classname[$i]}VsAll/objectPoseEstimation
done


