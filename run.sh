#Orig
make clean && make -j 56
if [ $1 == "rd" ]
then
    data="Radial"
elif [ $1 == "s" ]
then
    data="Shells"
elif [ $1 == "ra" ]
then
    data="Random"
fi
echo "/home/inspur/pac20/Data/$data 56 $2"
./Demo /home/inspur/pac20/Data/$data 56 $2

./verify /home/inspur/pac20/Data/$data.ref.$2 /home/inspur/pac20/Data/$data.bin.$2
# ./verify ../data/$data.int.$2 ../data/$data.bin.$2

#/home/inspur/pac20/Data/Random 56 1