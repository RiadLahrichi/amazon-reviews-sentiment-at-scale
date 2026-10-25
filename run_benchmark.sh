#!/usr/bin/env bash
# Runs train_one.py for every size with both engines, then draws the chart.
#
#   bash run_benchmark.sh                # the default sizes below
#   bash run_benchmark.sh 10000 all      # your own sizes, e.g. for a quick test

# sizes from the command line, or these defaults
sizes="${*:-10000 100000 500000 1000000 all}"

mkdir -p results
# fresh table, train_one.py adds a row per run
echo "engine,size,status,train_rows,seconds,memory_gb,macro_f1" > results/benchmark.csv

for size in $sizes; do
    for engine in sklearn spark; do
        echo "== $engine on $size reviews"
        # stop any run that takes more than 2 hours
        timeout 2h python train_one.py --engine "$engine" --size "$size"
        exit_code=$?

        # if the run was killed it couldn't save its row, so do it here
        # exit code 124 = timeout, 137 = killed by Linux (out of memory)
        if [ $exit_code -eq 124 ]; then
            echo "$engine,$size,timed out,,,," >> results/benchmark.csv
        elif [ $exit_code -ne 0 ]; then
            echo "$engine,$size,crashed or ran out of memory,,,," >> results/benchmark.csv
        fi
    done
done

python plot_benchmark.py
