#!/bin/bash

# Usage:
# 
#   `bash send.sh archivename.tar.gz n`
# 
# `${1}` is the archive file name, `archivename.tar.gz`.
# `${2}` is the size of each slice (in GigaBytes).
#

# create buffer folder for files to be sent ("slices")
IFS="."
ARCHIVENAME=( ${1} )
SEND_BUFFER="send_${ARCHIVENAME[0]}"
mkdir -p ${SEND_BUFFER}
unset IFS

# sign archive (using MD5)
ARCHIVESIG_SEND="${SEND_BUFFER}/${SEND_BUFFER}.md5"
md5sum ${1} > ${ARCHIVESIG_SEND}

# compute slices number and size
IFS=" "
ARCHIVE_PROPERTIES=( $(wc -c ${1}) )
FS=${ARCHIVE_PROPERTIES[0]}       # file size (in bytes)
unset IFS
BS=$((1024 * 4))                  # 4kB (page size)
BC=$((${2} * (1024 / 4) * 1024))  # blocks per slice
SS=$((${BS} * ${BC}))             # slice size (in bytes) 
N=$(((${FS} - 1) / ${SS} + 1))    # number of slices

# slice input file and sign each slice (using MD5)
SLICESIG_SEND="${SEND_BUFFER}/send_signatures.md5"
for i in $(seq 0 1 $((${N} - 1)))
do
  SLICENAME="${SEND_BUFFER}/slice${i}.gz"
  SADDR=$((${i} * ${SS}))
  EADDR=$(($((${SADDR} + ${SS} > ${FS})) ? ${FS} : $((${SADDR} + ${SS}))))
  NB=$(((${EADDR} - ${SADDR} - 1) / ${BS} + 1))  # number of blocks in current slice
  dd if=${1} bs=${BS} count=${NB} skip=$((${i} * ${BC})) status=progress | gzip > ${SLICENAME}
  md5sum ${SLICENAME} >> ${SLICESIG_SEND}
done
sed -i 's/send_//g' ${SLICESIG_SEND}

# send files to CINECA machines
# for details, see: https://wiki.u-gov.it/confluence/display/SCAIUS/Rsync
REMOTE_USER='mspallan'                            # username
REMOTE_HOST='login.galileo.cineca.it'             # host machine
REMOTE_WORK='/gpfs/work/cin_powerdam_4/mspallan'  # target working directory
RECV_BUFFER="${SEND_BUFFER/send_/receive_}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_WORK}/${RECV_BUFFER}"
for f in $(ls ${SEND_BUFFER})
do
  rsync -ravzHS --progress "${SEND_BUFFER}/${f}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_WORK}/${RECV_BUFFER}"
done

