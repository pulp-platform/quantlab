#!/bin/bash

# Usage:
# 
#   `bash receive.sh archivename.arxv1.arxv2...arxvN`
# 
# where `arxv1`, `arxv2`, ..., `arxvN` are compression extensions
# (e.g., `tar`, `gz`). Therefore, `${1}` is the archive file name.
# 

# get buffer folder (where received files have been stored)
IFS="."
name=( ${1} )
unset IFS
recv_buffer="receive_${name[0]}"
N=$(find ${recv_buffer} -iregex '^.*slice.*\.gz$' | wc -l)

# verify signatures of slices
slicesig_send="${recv_buffer}/send_signatures.md5"
slicesig_recv="${recv_buffer}/recv_signatures.md5"
for i in $(seq 0 1 $((${N} - 1)))
do
  slice="${recv_buffer}/slice${i}.gz" 
  md5sum ${slice} >> ${slicesig_recv}
done
sed -i 's/receive_//g' ${slicesig_recv}

slicesig_diff=$(cmp -s ${slicesig_recv} ${slicesig_send}; echo $?)
if [ ${slicesig_diff} -ne 0 ]; then
  echo "Verification of slices signatures failed! The slices will not be unpacked."
else
  # recompose archive
  for i in $(seq 0 1 $((${N} - 1)))
  do
    slice="${recv_buffer}/slice${i}.gz" 
    gunzip <${slice} >> ${1}
  done
  # check if complete transmission was correct
  archivesig_send="${recv_buffer}/${recv_buffer/receive_/send_}.md5"
  archivesig_recv="${recv_buffer}/${recv_buffer}.md5"
  md5sum ${1} > ${archivesig_recv}
  archivesig_diff=$(cmp -s ${archivesig_send} ${archivesig_recv}; echo $?)
  if [ ${archivesig_diff} -ne 0 ]; then
    echo "Verification of archive signature failed! The local archive will be erased."
    rm ${1}
  fi
fi

