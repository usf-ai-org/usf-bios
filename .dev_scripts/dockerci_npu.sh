#!/bin/bash
HF_HOME_DIR=/hf_cache
CODE_DIR=$PWD
USF_DEBUG=True
echo "$USER"
gpus='0,1 2,3'
is_get_file_lock=false
CI_COMMAND=${CI_COMMAND:-bash .dev_scripts/ci_container_test.sh python tests/run.py --parallel 2 --run_config tests/run_config.yaml}
echo "ci command: $CI_COMMAND"
PR_CHANGED_FILES="${PR_CHANGED_FILES:-}"
echo "PR modified files: $PR_CHANGED_FILES"
PR_CHANGED_FILES=${PR_CHANGED_FILES//[ ]/#}
echo "PR_CHANGED_FILES: $PR_CHANGED_FILES"
idx=0
for gpu in $gpus
do
 exec {lock_fd}>"/tmp/gpu$gpu" || exit 1
 flock -n "$lock_fd" || { echo "WARN: gpu $gpu is in use!" >&2; idx=$((idx+1)); continue; }
 echo "get gpu lock $gpu"

 let is_get_file_lock=true

 # 
 export CI_TEST=True
 export TEST_LEVEL=$TEST_LEVEL
 export HF_HOME=${HF_HOME:-$HF_HOME_DIR}
 export HF_ENDPOINT=$HF_ENDPOINT
 export HUB_DATASET_ENDPOINT=$HUB_DATASET_ENDPOINT
 export TEST_ACCESS_TOKEN_CITEST=$TEST_ACCESS_TOKEN_CITEST
 export TEST_ACCESS_TOKEN_SDKDEV=$TEST_ACCESS_TOKEN_SDKDEV
 export USF_ENVIRONMENT='ci'
 export TEST_UPLOAD_MS_TOKEN=$TEST_UPLOAD_MS_TOKEN
 export MODEL_TAG_URL=$MODEL_TAG_URL
 export HF_TOKEN=$HF_TOKEN
 export PR_CHANGED_FILES=$PR_CHANGED_FILES
 export CUDA_VISIBLE_DEVICES=$gpu

 if [ "$USF_DEBUG" == "True" ]; then
 export USF_DEBUG=True
 echo 'debugging'
 fi

 # 
 cd $CODE_DIR
 eval $CI_COMMAND

 if [ $? -ne 0 ]; then
 echo "Running test case failed, please check the log!"
 exit -1
 fi
 break
done

if [ "$is_get_file_lock" = false ] ; then
 echo 'No free GPU!'
 exit 1
fi
