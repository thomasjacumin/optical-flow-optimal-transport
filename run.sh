#!/bin/bash

run__help() {
  echo "help"
}

run__download() {
    mkdir data
    # Middlebury-1
    rm -Rf data/middlebury-1
    wget https://vision.middlebury.edu/flow/data/comp/zip/eval-gray-twoframes.zip
    unzip -qq eval-gray-twoframes.zip "eval-data-gray/**" -d data/middlebury-1/
    rm eval-gray-twoframes.zip
    # Middlebury-2
    rm -Rf data/middlebury-2
    wget https://vision.middlebury.edu/flow/data/comp/zip/other-gray-twoframes.zip
    unzip -qq other-gray-twoframes.zip "other-data-gray/**" -d data/middlebury-2/
    rm other-gray-twoframes.zip
    wget https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip
    unzip -qq other-gt-flow.zip "other-gt-flow/**" -d data/middlebury-2/
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
    # Middlebury-1
    mkdir results/middlebury-1
    for dir in data/middlebury-1/eval-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury-1/eval-data-gray/$input/frame10.png
            frame11=data/middlebury-1/eval-data-gray/$input/frame11.png

            mkdir results/middlebury-1/$input

            # # Gennert and Negahdaripour
            # if [ ! -f "results/middlebury-1/$input/.out.gn.sucess" ]; then
            #     python3 main.py $frame10 $frame11 \
            #         --out=results/middlebury-1/$input/gn.flo --save-benchmark=results/middlebury-1/$input/gn.benchmark.txt \
            #         --save-reconstruction=results/middlebury-1/$input/gn.rec.png \
            #         --save-lum=results/middlebury-1/$input/gn.lum.png \
            #         --algo=GN --alpha=0.1 --lambda=0.2 --normalize
            #     ./bin/color_flow results/middlebury-1/$input/gn.flo results/middlebury-1/$input/gn.png
            #     touch results/middlebury-1/$input/.out.gn.sucess
            # fi

            # FOTO
            if [ ! -f "results/middlebury-1/$input/.out.foto.sucess" ]; then
                python3 main.py $frame10 $frame11 \
                    --out=results/middlebury-1/$input/foto.flo --save-benchmark=results/middlebury-1/$input/foto.benchmark.txt \
                    --save-reconstruction=results/middlebury-1/$input/foto.rec.png \
                    --save-lum=results/middlebury-1/$input/foto.lum.png \
                    --algo=foto --epsilon=0.5 --Nt=2 --max-it=1000 --normalize
                ./bin/color_flow results/middlebury-1/$input/foto.flo results/middlebury-1/$input/foto.png
                touch results/middlebury-1/$input/.out.foto.sucess
            fi
        fi
    done

    # Middlebury-2
    mkdir results/middlebury-2
    for dir in data/middlebury-2/other-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury-2/other-data-gray/$input/frame10.png
            frame11=data/middlebury-2/other-data-gray/$input/frame11.png
            ground_truth=data/middlebury-2/other-gt-flow/$input/flow10.flo

            if test -f "$ground_truth"; then
                mkdir results/middlebury-2/$input
                
                # Generate ground truth           
                normalizing=$(./bin/color_flow $ground_truth results/middlebury-2/$input/flow10.png | grep -Eo '^max motion: [[:digit:]]+([.][[:digit:]]+)?' | grep -Eo '[[:digit:]]+([.][[:digit:]]+)?$')
                ./bin/color_flow $ground_truth results/middlebury-2/$input/flow10.png $normalizing
                echo "optical flow will be normalize by ${normalizing}"
    
                # Gennert and Negahdaripour
                if [ ! -f "results/middlebury-2/$input/.out.gn.sucess" ]; then
                    python3 main.py $frame10 $frame11 --ground-truth=$ground_truth \
                        --out=results/middlebury-2/$input/gn.flo --save-benchmark=results/middlebury-2/$input/gn.benchmark.txt \
                        --save-reconstruction=results/middlebury-2/$input/gn.rec.png \
                        --save-lum=results/middlebury-2/$input/gn.lum.png \
                        --algo=GN --alpha=0.1 --lambda=0.2
                    ./bin/color_flow results/middlebury-2/$input/gn.flo results/middlebury-2/$input/gn.png # $normalizing
                    touch results/middlebury-2/$input/.out.gnhs.sucess
                fi

                # FOTO
                if [ ! -f "results/middlebury-2/$input/.out.foto.sucess" ]; then
                    python3 main.py $frame10 $frame11 --ground-truth=$ground_truth \
                        --out=results/middlebury-2/$input/foto.flo --save-benchmark=results/middlebury-2/$input/foto.benchmark.txt \
                        --save-reconstruction=results/middlebury-2/$input/foto.rec.png \
                        --save-lum=results/middlebury-2/$input/foto.lum.png \
                        --algo=foto --epsilon=0.1 --Nt=4
                    ./bin/color_flow results/middlebury-2/$input/foto.flo results/middlebury-2/$input/foto.png # $normalizing
                    touch results/middlebury-2/$input/.out.foto.sucess
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


