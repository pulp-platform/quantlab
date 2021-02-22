#!/bin/bash

# Usage:
# 
#   `bash receive.sh archivename.tar.gz`
# 
# `${1}` is the archive file name, `archivename.tar.gz`.
# 

# get buffer folder (where received files have been stored)
IFS="."
ARCHIVENAME=( ${1} )
RECV_BUFFER="receive_${ARCHIVENAME[0]}"
unset IFS
N=$(find ${RECV_BUFFER} -iregex '^.*slice.*\.gz$' | wc -l)

# verify signatures of slices
SLICESIG_SEND="${recv_buffer}/send_signatures.md5"
SLICESIG_RECV="${recv_buffer}/recv_signatures.md5"
for i in $(seq 0 1 $((${N} - 1)))
do
  SLICE="${RECV_BUFFER}/slice${i}.gz" 
  md5sum ${SLICE} >> ${SLICESIG_RECV}
done
sed -i 's/receive_//g' ${SLICESIG_RECV}

SLICESIG_DIFF=$(cmp -s ${SLICESIG_RECV} ${SLICESIG_SEND}; echo $?)
if [ ${SLICESIG_DIFF} -ne 0 ]; then
  echo "Verification of slices signatures failed! The slices will not be unpacked."
else
  # recompose archive
  for i in $(seq 0 1 $((${N} - 1)))
  do
    SLICE="${RECV_BUFFER}/slice${i}.gz" 
    gunzip <${SLICE} >> ${1}
  done
  # check if complete transmission was correct
  ARCHIVESIG_SEND="${RECV_BUFFER}/${RECV_BUFFER/receive_/send_}.md5"
  ARCHIVESIG_RECV="${RECV_BUFFER}/${RECV_BUFFER}.md5"
  md5sum ${1} > ${ARCHIVESIG_RECV}
  ARCHIVESIG_DIFF=$(cmp -s ${ARCHIVESIG_RECV} ${ARCHIVESIG_SEND}; echo $?)
  if [ ${ARCHIVESIG_DIFF} -ne 0 ]; then
    echo "Verification of archive signature failed! The local archive will be erased."
    rm ${1}
  fi
fi

