#!/bin/bash

run__download() {
    rm -Rf data
    mkdir data

    # Middlebury-1
    rm -Rf data/middlebury-1
    wget https://vision.middlebury.edu/flow/data/comp/zip/eval-gray-twoframes.zip
    unzip -qq eval-gray-twoframes.zip "eval-data-gray/**" -d data/middlebury-1/
    rm eval-gray-twoframes.zip

    run__resizedataset
    run__createlumdataset
    run__normalizedataset
}

run__resizedataset() {
    echo 'Resizing datasets'
    for dir in data/middlebury-1/eval-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury-1/eval-data-gray/$input/frame10.png
            frame11=data/middlebury-1/eval-data-gray/$input/frame11.png
            
            magick $frame10 -resize 50% $frame10
            magick $frame11 -resize 50% $frame11
        fi
    done
}

run__createlumdataset() {
    RANDOM=12345  # Set seed
    echo 'Adding random artifical illumination'
    mkdir data/middlebury-1-lum
    mkdir data/middlebury-1-lum/eval-data-gray
    for dir in data/middlebury-1/eval-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury-1/eval-data-gray/$input/frame10.png
            frame11=data/middlebury-1/eval-data-gray/$input/frame11.png

            mkdir data/middlebury-1-lum/eval-data-gray/$input
            cp $frame10 data/middlebury-1-lum/eval-data-gray/$input/frame10.png
            python3 bin/create_lum_dataset.py $frame11 data/middlebury-1-lum/eval-data-gray/$input/frame11.png $RANDOM
        fi
    done
}

run__normalizedataset() {
    echo 'Normalizing datasets'
    for dir in data/middlebury-1/eval-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury-1/eval-data-gray/$input/frame10.png
            frame11=data/middlebury-1/eval-data-gray/$input/frame11.png
            
            python3 bin/normalize_image.py $frame10 $frame11 $frame10 $frame11
        fi
    done
    for dir in data/middlebury-1-lum/eval-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury-1-lum/eval-data-gray/$input/frame10.png
            frame11=data/middlebury-1-lum/eval-data-gray/$input/frame11.png
            
            python3 bin/normalize_image.py $frame10 $frame11 $frame10 $frame11
        fi
    done
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

            # compute difference
            python3 data_diff.py $frame10 $frame11 results/middlebury-1/$input/diff.png

            # Gennert and Negahdaripour
            if [ ! -f "results/middlebury-1/$input/.out.gn.sucess" ]; then
                python3 main.py $frame10 $frame11 \
                    --out=results/middlebury-1/$input/gn.flo --save-benchmark=results/middlebury-1/$input/gn.benchmark.txt \
                    --save-reconstruction=results/middlebury-1/$input/gn.rec.png \
                    --save-lum=results/middlebury-1/$input/gn.lum.png \
                    --algo=GN --alpha=0.1 --lambda=0.2 --normalize
                ./bin/color_flow results/middlebury-1/$input/gn.flo results/middlebury-1/$input/gn.png
                touch results/middlebury-1/$input/.out.gn.sucess
            fi

            # FOTO
            if [ ! -f "results/middlebury-1/$input/.out.foto.sucess" ]; then
                python3 main.py $frame10 $frame11 \
                    --out=results/middlebury-1/$input/foto.flo --save-benchmark=results/middlebury-1/$input/foto.benchmark.txt \
                    --save-reconstruction=results/middlebury-1/$input/foto.rec.png \
                    --save-lum=results/middlebury-1/$input/foto.lum.png \
                    --algo=foto --r=1 --convergence-tol=0.01 --reg-epsilon=1e-2 --Nt=16 --max-it=200 --normalize
                ./bin/color_flow results/middlebury-1/$input/foto.flo results/middlebury-1/$input/foto.png
                touch results/middlebury-1/$input/.out.foto.sucess
            fi
        fi
    done

    # Middlebury-1-lum
    mkdir results/middlebury-1-lum
    for dir in data/middlebury-1-lum/eval-data-gray/*; do
        if [ -d "$dir" ]; then
            input=${dir##*/}
            frame10=data/middlebury-1-lum/eval-data-gray/$input/frame10.png
            frame11=data/middlebury-1-lum/eval-data-gray/$input/frame11.png

            mkdir results/middlebury-1-lum/$input

            # compute difference
            python3 data_diff.py $frame10 $frame11 results/middlebury-1-lum/$input/diff.png

            # Gennert and Negahdaripour
            if [ ! -f "results/middlebury-1-lum/$input/.out.gn.sucess" ]; then
                python3 main.py $frame10 $frame11 \
                    --out=results/middlebury-1-lum/$input/gn.flo --save-benchmark=results/middlebury-1-lum/$input/gn.benchmark.txt \
                    --save-reconstruction=results/middlebury-1-lum/$input/gn.rec.png \
                    --save-lum=results/middlebury-1-lum/$input/gn.lum.png \
                    --algo=GN --alpha=0.1 --lambda=0.2 --normalize
                ./bin/color_flow results/middlebury-1-lum/$input/gn.flo results/middlebury-1-lum/$input/gn.png
                touch results/middlebury-1-lum/$input/.out.gn.sucess
            fi

            # FOTO
            if [ ! -f "results/middlebury-1-lum/$input/.out.foto.sucess" ]; then
                python3 main.py $frame10 $frame11 \
                    --out=results/middlebury-1-lum/$input/foto.flo --save-benchmark=results/middlebury-1-lum/$input/foto.benchmark.txt \
                    --save-reconstruction=results/middlebury-1-lum/$input/foto.rec.png \
                    --save-lum=results/middlebury-1-lum/$input/foto.lum.png \
                    --algo=foto --r=1 --convergence-tol=0.01 --reg-epsilon=1e-2 --Nt=16 --max-it=200 --normalize
                ./bin/color_flow results/middlebury-1-lum/$input/foto.flo results/middlebury-1-lum/$input/foto.png
                touch results/middlebury-1-lum/$input/.out.foto.sucess
            fi
        fi
    done
}

if [ "$1" = "download" ]; then
    run__download
elif [ "$1" = "install" ]; then
    run__installpipdependencies
elif [ "$1" = "restart" ]; then
    run__restart
else
    run
fi