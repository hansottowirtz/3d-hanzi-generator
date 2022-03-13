#!/bin/bash
# from https://stackoverflow.com/a/53187369
ffmpeg -i $1 -i $2 -filter_complex "[0][1]scale2ref='oh*mdar':'if(lt(main_h,ih),ih,main_h)'[0s][1s];
  [1s][0s]scale2ref='oh*mdar':'if(lt(main_h,ih),ih,main_h)'[1s][0s];
  [0s][1s]hstack,setsar=1" $3
