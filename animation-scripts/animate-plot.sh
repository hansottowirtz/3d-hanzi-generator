#!/bin/bash

# Usage:
# bash animation-scripts/animate-plot.sh --t-purpose smoothen_curve

set -euxo pipefail
dir="out/anim/$(date +%s)"
mkdir -p $dir

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

N=4
open_sem $N

task() {
  # sleep 0.5; echo "$1";
  p=$(python -c "print('%05d' % ((10 + $1) * 1000))")
  python src/main.py --out-scad $dir/model-$p.scad --settings examples/hao.yml --parts strokes --t $* --out-debug-plot $dir/output-$p.png
  # /Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD --imgsize 2048,2048 --autocenter --camera 60,-80,150,0,0,20 --colorscheme=Nature2 -o $dir/output-$p.png $dir/model-$p.scad
  # convert $dir/output-$p.png -trim $dir/output-$p.png;
}

for x in $(seq 0.0 0.01 1.0); do
  run_with_lock task $x $@
done
wait
ffmpeg \
  -framerate 20 \
  -pattern_type glob \
  -i "$dir/output-*.png" \
  -vf scale=2048:-1 \
  $dir/result.webm
# convert -delay 20 -loop 0 $dir/output-*.png $dir/result.gif
