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
./verify /home/inspur/pac20/old_Data/$data.ref.$2 /home/inspur/pac20/Data/$data.bin.$2
