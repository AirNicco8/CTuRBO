#!/bin/bash
#echo 'Data: '$1' - Split: '$2' - Transformed: '$3
#to create table 2

instances="2 5 15 19 28"
t_bounds="0 60 120 300"
sol_quality="0 10 20"
evals="50 100 200"

for Y in $t_bounds #time bound
    do
        for X in $sol_quality # max evals (sol quality)
            do
                for Z in $instances #instanceId
                    do
                        if [ $3 -eq 1 ]
                        then
                            python3 main.py --data $1 --fix_instance $Z --max_time $Y --max_mem 180 --sol_q $X --trust_regions 2 --max_evals 150 --load_gp --freeze_gp --split $2 --transform --csv
                        else
                            python3 main.py --data $1 --fix_instance $Z --max_time $Y --max_mem 180 --sol_q $X --trust_regions 2 --max_evals 150 --load_gp --freeze_gp --split $2 --csv
                        fi
                    done
           done
    done


