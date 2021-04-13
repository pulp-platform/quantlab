#!/bin/bash

# Usage:
# 
#   `bash send.sh archivename.arxv1.arxv2...arxvN YY`
# 
# where `arxv1`, `arxv2`, ..., `arxvN` are compression extensions
# (e.g., `tar`, `gz`). Therefore, `${1}` is the archive file name.
# YY (${2}) is the size of each slice in GigaBytes.
#

# create buffer folder for files to be sent (slices)
IFS="."
name=( ${1} )
send_buffer="send_${name[0]}"
mkdir -p ${send_buffer}
unset IFS
echo ${send_buffer}

# sign archive (using MD5)
archivesig_send="${send_buffer}/${send_buffer}.md5"
md5sum ${1} > ${archivesig_send}

# compute slices number and size
IFS=" "
archive_props=( $(wc -c ${1}) )
fs=${archive_props[0]}            # file size (in bytes)
unset IFS
bs=$((1024 * 4))                  # 4kB (page size)
bc=$((${2} * (1024 / 4) * 1024))  # blocks per slice
ss=$((${bs} * ${bc}))             # slice size (in bytes) 
N=$(((${fs} - 1) / ${ss} + 1))    # number of slices

# slice input file and sign each slice (using MD5)
slicesig_send="${send_buffer}/send_signatures.md5"
for i in $(seq 0 1 $((${N} - 1)))
do
  slice="${send_buffer}/slice${i}.gz"
  sadd=$((${i} * ${ss}))
  eadd=$(($((${sadd} + ${ss} > ${fs})) ? ${fs} : $((${sadd} + ${ss}))))
  Nb=$(((${eadd} - ${sadd} - 1) / ${bs} + 1))  # number of blocks in current slice
  dd if=${1} bs=${bs} count=${Nb} skip=$((${i} * ${bc})) status=progress | gzip > ${slice}
  md5sum ${slice} >> ${slicesig_send}
done
sed -i 's/send_//g' ${slicesig_send}

# send files to CINECA machines
# for details, see: https://wiki.u-gov.it/confluence/display/SCAIUS/Rsync
remote_user='mspallan'                            # username
remote_host='login.galileo.cineca.it'             # host machine
remote_work='/gpfs/work/cin_powerdam_4/mspallan'  # working directory
recv_buffer="${send_buffer/send_/receive_}"

ssh "${remote_user}@${remote_host}" "mkdir -p ${remote_work}/${recv_buffer}"
for f in $(ls ${send_buffer})
do
  rsync -ravzHS --progress "${send_buffer}/${f}" "${remote_user}@${remote_host}:${remote_work}/${recv_buffer}"
done

