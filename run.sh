#!/bin/bash

run__help() {
  echo "help"
}

run__download() {
    mkdir data
    # Middlebury
    rm -Rf data/middlebury
    wget https://vision.middlebury.edu/flow/data/comp/zip/other-gray-twoframes.zip
    unzip -qq other-gray-twoframes.zip "other-data-gray/**" -d data/middlebury/
    rm other-gray-twoframes.zip
    wget https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip
    unzip -qq other-gt-flow.zip "other-gt-flow/**" -d data/middlebury/
    rm other-gt-flow.zip
}

run__installpipdependencies() {
    pip install -r requirements.txt
}

run__restart() {
    rm -Rf results
    run
}

run() {
    mkdir results
    # Middlebury
    mkdir results/middlebury
    for dir in data/middlebury/other-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury/other-data-gray/$input/frame10.png
            frame11=data/middlebury/other-data-gray/$input/frame11.png
            ground_truth=data/middlebury/other-gt-flow/$input/flow10.flo

            if test -f "$ground_truth"; then
                mkdir results/middlebury/$input
                
                # Generate ground truth           
                normalizing=$(./bin/color_flow $ground_truth results/middlebury/$input/flow10.png | grep -Eo '^max motion: [[:digit:]]+([.][[:digit:]]+)?' | grep -Eo '[[:digit:]]+([.][[:digit:]]+)?$')
                ./bin/color_flow $ground_truth results/middlebury/$input/flow10.png $normalizing
                echo "optical flow will be normalize by ${normalizing}"
    
                # FOTO
                if [ ! -f "results/middlebury/$input/.out.foto.sucess" ]; then
                    python3 main.py $frame10 $frame11 --ground-truth=$ground_truth --out=results/middlebury/$input/foto.flo --save-benchmark=results/middlebury/$input/foto.benchmark.txt \
                        --epsilon=0.1 --Nt=6
                    ./bin/color_flow results/middlebury/$input/foto.flo results/middlebury/$input/foto.png $normalizing
                    touch results/middlebury/$input/.out.foto.sucess
                fi

            fi
        fi
    done
}

if [ "$1" = "help" ]; then
    run__help
elif [ "$1" = "download" ]; then
    run__download
elif [ "$1" = "install" ]; then
    run__installpipdependencies
elif [ "$1" = "restart" ]; then
    run__restart
else
    run
fi


