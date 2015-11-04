epsilon=70000,200000,3000000,7000000
pca=2,4,6,8,10,20
numclusters=2,5,10,20,50,80,100,150
for i in $(echo $epsilon | sed "s/,/ /g")
do
    for j in $(echo $pca | sed "s/,/ /g")
    do
        for k in $(echo $numclusters | sed "s/,/ /g")
        do
    # call your procedure/other scripts here below
    ./spark-submit --class "PCA" --master local[4] --driver-memory 12G PCANormal/target/scala-2.10/pca_2.10-1.0.jar $j $k $i > "outputpca/$j$k$i"
    # echo "$i
        done
    done
done
