#!/usr/bin/env bash

PROBLEM=${1} # TODO: check that this is alphanumeric

PATH_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PATH_QUANTLAB_HOME=$(cd $(dirname ${PATH_CFG}) && pwd)

# retrieve (hard) systems folder
PATH_STORAGE_CFG=${PATH_CFG}/storage_cfg.json
PYTHON_READ_STORAGE_CFG_DATA="import sys; import json; fp = open('${PATH_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
PATH_HARD_SYSTEMS_PKG_4DATA=$(python -c "${PYTHON_READ_STORAGE_CFG_DATA}")/$(basename ${PATH_QUANTLAB_HOME})/systems

if [ ! -d "${PATH_HARD_SYSTEMS_PKG_4DATA}" ]
then
    echo "[QuantLab] QuantLab data storage is not properly configured!"
else

    # set up problem package
    PATH_PROBLEM_PKG=${PATH_QUANTLAB_HOME}/systems/${PROBLEM}
    if [ -d "${PATH_PROBLEM_PKG}" ] && [ -f "${PATH_PROBLEM_PKG}/__init__.py" ]
    then
        echo "[QuantLab] It seems that the QuantLab package for problem ${PROBLEM} has already been created..."
    else
        mkdir ${PATH_PROBLEM_PKG}
        touch ${PATH_PROBLEM_PKG}/__init__.py
        mkdir ${PATH_PROBLEM_PKG}/utils
        touch ${PATH_PROBLEM_PKG}/utils/__init__.py
        mkdir ${PATH_PROBLEM_PKG}/utils/statistics
        touch ${PATH_PROBLEM_PKG}/utils/statistics/__init__.py
        touch ${PATH_PROBLEM_PKG}/utils/statistics/taskstatistic.py
    fi

    # create (hard) data folder
    PATH_HARD_PROBLEM_PKG_DATA=${PATH_HARD_SYSTEMS_PKG_4DATA}/${PROBLEM}/data
    mkdir -p ${PATH_HARD_PROBLEM_PKG_DATA}
    echo "[QuantLab] Remember to prepare the data for problem ${PROBLEM} at <$(dirname ${PATH_HARD_PROBLEM_PKG_DATA})>."

    # link (soft) data folder to (hard) data folder
    ln -s ${PATH_HARD_PROBLEM_PKG_DATA} ${PATH_PROBLEM_PKG}/data

fi
